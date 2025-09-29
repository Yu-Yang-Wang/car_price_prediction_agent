"""Consistency/cross-check agent: lets agents question each other.

It compares Market vs Residual vs CarsXE (if present) and produces a
structured list of issues with suggested actions. Optionally asks an LLM to
summarize inconsistencies for human-friendly notes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)

try:
    from car_analysis.nodes.tools import get_llm  # optional
except Exception:  # pragma: no cover
    get_llm = None  # type: ignore

from .condition import _basic_car_context


def _issues_numeric(state: CarAnalysisState) -> List[Dict[str, Any]]:
    """Rule-based checks across agents."""
    issues: List[Dict[str, Any]] = []

    market = state.get("market_analysis", {}) or {}
    residual = state.get("residual_analysis", {}) or {}
    carsxe = (state.get("rag_insights", {}) or {}).get("carsxe", {}) or {}

    median = market.get("market_median")
    price_delta_pct = market.get("price_delta_pct")
    residual_price = residual.get("predicted_price")
    carsxe_avg = None
    # try to extract a simple average price from carsxe payload if present
    raw = carsxe.get("raw")
    if isinstance(raw, dict):
        for k in ("averageMarketPrice", "average_market_price", "average_price", "average"):
            if k in raw and isinstance(raw[k], (int, float)):
                carsxe_avg = float(raw[k])
                break

    # 1) Large disagreement between rule and LLM scores
    rule_score = market.get("rule_score")
    llm_score = market.get("llm_score")
    if isinstance(rule_score, (int, float)) and isinstance(llm_score, (int, float)):
        if abs(rule_score - llm_score) >= 25:
            issues.append({
                "type": "score_disagreement",
                "severity": "medium",
                "details": f"Rule {rule_score} vs LLM {llm_score}",
                "action": "Revisit market comps or regenerate LLM opinion"
            })

    # 2) Residual value inconsistent with market median
    if isinstance(residual_price, (int, float)) and isinstance(median, (int, float)) and median > 0:
        gap_pct = (residual_price - median) / median * 100
        if abs(gap_pct) >= 20:
            issues.append({
                "type": "residual_vs_market",
                "severity": "medium",
                "details": f"Residual {residual_price:,.0f} vs market {median:,.0f} ({gap_pct:+.1f}%)",
                "action": "Check model features or market comps; verify mileage normalization"
            })

    # 3) CarsXE vs market median disagreement
    if isinstance(carsxe_avg, (int, float)) and isinstance(median, (int, float)) and median > 0:
        gap_pct = (carsxe_avg - median) / median * 100
        if abs(gap_pct) >= 15:
            issues.append({
                "type": "carsxe_vs_market",
                "severity": "low",
                "details": f"CarsXE {carsxe_avg:,.0f} vs market {median:,.0f} ({gap_pct:+.1f}%)",
                "action": "Use blended reference or prefer source with higher reliability"
            })

    # 4) Large price delta but high LLM score
    if isinstance(price_delta_pct, (int, float)) and isinstance(llm_score, (int, float)):
        if price_delta_pct > 15 and llm_score >= 75:
            issues.append({
                "type": "delta_high_but_llm_positive",
                "severity": "low",
                "details": f"Delta {price_delta_pct:+.1f}% with LLM score {llm_score}",
                "action": "Explain rationale (rare trim, options) or lower LLM score"
            })

    return issues


async def consistency_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Cross-check outputs across agents; optional LLM critique."""

    logs = log_agent_start("consistency", payload=_basic_car_context(state))

    try:
        issues = _issues_numeric(state)

        critique = None
        if get_llm is not None:
            try:
                llm = get_llm()
                car = state.get("current_car", {})
                market = state.get("market_analysis", {})
                residual = state.get("residual_analysis", {})
                prompt = (
                    "You are a car pricing QA assistant. Given the car info and agent outputs, "
                    "list inconsistencies and suggest concise fixes. Keep it under 120 words.\n\n"
                    f"Car: {car}\nMarket: {market}\nResidual: {residual}\nIssues: {issues}"
                )
                resp = await llm.ainvoke([("user", prompt)])
                critique = getattr(resp, "content", None) or str(resp)
            except Exception:  # keep robust
                critique = None

        report = {
            "success": True,
            "issues": issues,
            "llm_critique": critique,
        }

        return {
            **logs,
            "consistency_report": report,
            **log_agent_complete("consistency", payload={"issue_count": len(issues)})
        }

    except Exception as exc:
        return {
            **logs,
            "consistency_report": {"success": False, "error": str(exc)},
            **log_agent_error("consistency", exc),
        }

