"""Market pricing agent using Tavily research, scoring, and LLM opinion."""

from __future__ import annotations

from typing import Any, Dict

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)
from car_analysis.core.rag_enhanced_workers import (
    price_research_worker,
    price_comparison_worker,
    deal_scoring_worker,
    llm_opinion_worker,
)

from .condition import _basic_car_context


async def market_price_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Market analysis using Tavily research + scoring agents."""

    logs = log_agent_start("market_price", payload=_basic_car_context(state))
    working_state: Dict[str, Any] = {**state, **logs}

    try:
        research_update = await price_research_worker(working_state)
        working_state.update(research_update)

        comparison_update = await price_comparison_worker(working_state)
        working_state.update(comparison_update)

        scoring_update = await deal_scoring_worker(working_state)
        working_state.update(scoring_update)

        llm_update = await llm_opinion_worker(working_state)
        working_state.update(llm_update)

        price_research = working_state.get("price_research", {})
        price_comparison = working_state.get("price_comparison", {})
        deal_score = working_state.get("deal_score", {})
        llm_opinion = working_state.get("llm_opinion", {})

        summary = {
            "success": bool(price_research.get("success")) and bool(price_comparison.get("success")),
            "market_median": price_comparison.get("market_median"),
            "price_delta": price_comparison.get("price_delta"),
            "price_delta_pct": price_comparison.get("price_delta_pct"),
            "deal_category": price_comparison.get("verdict_category"),
            "rule_score": deal_score.get("score"),
            "rule_verdict": deal_score.get("verdict"),
            "llm_score": llm_opinion.get("score"),
            "llm_verdict": llm_opinion.get("verdict"),
        }

        return {
            **logs,
            **research_update,
            **comparison_update,
            **scoring_update,
            **llm_update,
            "market_analysis": summary,
            **log_agent_complete(
                "market_price",
                payload={
                    "market_median": summary.get("market_median"),
                    "deal_category": summary.get("deal_category"),
                },
            ),
        }
    except Exception as exc:
        return {
            **logs,
            "market_analysis": {"success": False, "error": str(exc)},
            **log_agent_error("market_price", exc),
        }
