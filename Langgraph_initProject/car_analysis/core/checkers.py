
"""Checker functions with retry logic for car analysis"""

import asyncio
from .models import CarAnalysisState


async def price_research_checker(state: CarAnalysisState) -> CarAnalysisState:
    """Check if price research was successful"""

    MAX_PRICE_RESEARCH_RETRIES = 3

    research = state.get("price_research", {})
    retries = state.get("retries", {})
    analysis_errors = state.get("analysis_errors", [])

    # Check for successful research with adequate data (increased minimum for better accuracy)
    if research.get("success") and research.get("sample_count", 0) >= 5:
        return {
            **state,
            "research_ok": True,
            "dbg_logs": ["Price research validation passed"]
        }

    # Retry logic
    retry_count = retries.get("price_research", 0)
    if retry_count >= MAX_PRICE_RESEARCH_RETRIES:
        # Permanent failure - record error
        error_msg = f"TAVILY_SEARCH_FAILED: Unable to fetch market data after {MAX_PRICE_RESEARCH_RETRIES} attempts. {research.get('error', 'Unknown search error')}"
        analysis_errors.append(error_msg)

        return {
            **state,
            "research_ok": False,
            "research_final": True,
            "failed_permanently": True,
            "analysis_errors": analysis_errors,
            "dbg_logs": [f"Price research permanently failed: {error_msg}"]
        }

    retries["price_research"] = retry_count + 1
    await asyncio.sleep(0.2)  # Brief backoff

    print(f"   🔄 Retrying price research: attempt {retry_count + 1}/{MAX_PRICE_RESEARCH_RETRIES}")

    return {
        **state,
        "research_ok": False,
        "research_final": False,
        "retries": retries,
        "analysis_errors": analysis_errors,
        "dbg_logs": [f"Price research retry {retry_count + 1}/{MAX_PRICE_RESEARCH_RETRIES}"]
    }


async def comparison_checker(state: CarAnalysisState) -> CarAnalysisState:
    """Check if price comparison was successful"""
    comparison = state.get("price_comparison", {})
    retries = state.get("retries", {})

    if comparison.get("success") and "verdict_category" in comparison:
        return {"comparison_ok": True, "dbg_logs": ["Price comparison validation passed"]}

    # Retry logic
    retry_count = retries.get("price_comparison", 0)
    if retry_count >= 3:
        return {
            "comparison_ok": False,
            "comparison_final": True,
            "dbg_logs": ["Price comparison failed after max retries"]
        }

    retries["price_comparison"] = retry_count + 1
    await asyncio.sleep(0.1)

    return {
        "comparison_ok": False,
        "comparison_final": False,
        "retries": retries,
        "dbg_logs": [f"Price comparison retry {retry_count + 1}"]
    }


async def scoring_checker(state: CarAnalysisState) -> CarAnalysisState:
    """Check if deal scoring was successful"""
    scoring = state.get("deal_score", {})
    retries = state.get("retries", {})

    if scoring.get("success") and "score" in scoring:
        return {"scoring_ok": True, "dbg_logs": ["Deal scoring validation passed"]}

    # Retry logic
    retry_count = retries.get("deal_scoring", 0)
    if retry_count >= 3:
        return {
            "scoring_ok": False,
            "scoring_final": True,
            "dbg_logs": ["Deal scoring failed after max retries"]
        }

    retries["deal_scoring"] = retry_count + 1
    await asyncio.sleep(0.1)

    return {
        "scoring_ok": False,
        "scoring_final": False,
        "retries": retries,
        "dbg_logs": [f"Deal scoring retry {retry_count + 1}"]
    }

def verdict_score(score: int) -> int:
    if score >= 90: return 5
    if score >= 75: return 4
    if score >= 60: return 3
    if score >= 40: return 2
    return 1

async def join_scores(state: CarAnalysisState) -> CarAnalysisState:
    """Join node to ensure both scoring branches have completed.

    Returns a flag `awaiting_scores` indicating whether to keep waiting.
    """
    # Ensure both deal_score and llm_opinion are present before proceeding
    deal_score = state.get("deal_score", {})
    llm_opinion = state.get("llm_opinion", {})

    has_deal = bool(deal_score.get("score") or deal_score.get("success"))
    has_llm = bool(llm_opinion.get("score") or llm_opinion.get("verdict"))

    if not (has_deal and has_llm):
        # Small backoff to avoid a tight loop
        await asyncio.sleep(0.05)
        return {
            **state,
            "awaiting_scores": True,
            "dbg_logs": ["Waiting for parallel scoring branches to complete"]
        }

    return {
        **state,
        "awaiting_scores": False,
        "dbg_logs": ["Both scoring branches present"]
    }

def score_disagreement_checker(state: CarAnalysisState) -> CarAnalysisState:
    """Check for disagreement between rule-based and LLM scoring"""

    MAX_LLM_RETRIES = 2

    rule_score = state.get("deal_score", {}).get("score", 0)
    llm_score = state.get("llm_opinion", {}).get("score", 0)
    retries = state.get("retries", {})
    analysis_errors = state.get("analysis_errors", [])

    # At this stage we expect both scores to be present because of the join node
    if not rule_score or not llm_score:
        print("   ⚠️  Missing scores at score_check (unexpected)")
        return {
            **state,
            "llm_retry": False,
            "score_disagree_retry": False,
            "retries": retries,
            "analysis_errors": analysis_errors,
        }

    rule_level = verdict_score(rule_score)
    llm_level = verdict_score(llm_score)

    score_diff = abs(rule_score - llm_score)
    level_diff = abs(rule_level - llm_level)
    print(f"   🔍 Disagreement check: Rule={rule_score}, LLM={llm_score}, Diff={score_diff}")

    # Major disagreement definition
    major_disagreement = level_diff >= 2 or score_diff >= 30

    # Strategy: Retry LLM first; if still major disagreement after retries, then refresh research
    MAX_RESEARCH_REFRESH_RETRIES = 1
    if major_disagreement:
        llm_retries = retries.get("llm_opinion", 0)
        if llm_retries < MAX_LLM_RETRIES:
            retries["llm_opinion"] = llm_retries + 1
            print(f"   🔄 Major disagreement - retrying LLM opinion {llm_retries + 1}/{MAX_LLM_RETRIES}")
            return {
                **state,
                "awaiting_scores": False,
                "llm_retry": True,
                "score_disagree_retry": False,
                "retries": retries,
                "analysis_errors": analysis_errors,
            }
        else:
            # LLM retries exhausted — consider refreshing market research, with cap
            prev = retries.get("score_disagreement", 0)
            if prev < MAX_RESEARCH_REFRESH_RETRIES:
                retries["score_disagreement"] = prev + 1
                print("   🔁 LLM retries exhausted — falling back to price research")
                return {
                    **state,
                    "awaiting_scores": False,
                    "llm_retry": False,
                    "score_disagree_retry": True,
                    "retries": retries,
                    "analysis_errors": analysis_errors,
                }
            else:
                # Give up after refresh attempt — mark permanently failed with explanation
                err = (
                    f"DISAGREEMENT_PERSISTENT: Rule-based score ({rule_score}) and LLM score ({llm_score}) "
                    f"disagree after LLM retries ({MAX_LLM_RETRIES}) and research refresh ({MAX_RESEARCH_REFRESH_RETRIES})"
                )
                analysis_errors.append(err)
                print(f"   ❌ {err}")
                return {
                    **state,
                    "awaiting_scores": False,
                    "llm_retry": False,
                    "score_disagree_retry": False,
                    "failed_permanently": True,
                    "retries": retries,
                    "analysis_errors": analysis_errors,
                }

    # Within acceptable range — proceed to report
    print("   ✅ Scores within acceptable range")
    return {
        **state,
        "awaiting_scores": False,
        "llm_retry": False,
        "score_disagree_retry": False,
        "retries": retries,
        "analysis_errors": analysis_errors,
    }
