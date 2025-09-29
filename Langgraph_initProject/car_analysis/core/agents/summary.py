"""Summary agent consolidating outputs."""

from __future__ import annotations

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
)
try:
    from car_analysis.nodes.tools import get_llm  # optional
except Exception:  # pragma: no cover
    get_llm = None  # type: ignore
from .condition import _basic_car_context


async def summary_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Consolidate agent outputs into a summary skeleton."""

    logs = log_agent_start("summary", payload=_basic_car_context(state))

    condition = state.get("condition_report", {}) or {}
    market = state.get("market_analysis", {}) or {}
    residual = state.get("residual_analysis", {}) or {}
    news = state.get("news_analysis", {}) or {}
    insights = state.get("rag_insights", {}) or {}

    summary = {
        "success": True,
        "highlights": {
            "condition": condition.get("condition_flags"),
            "market": {
                "deal_category": market.get("deal_category"),
                "rule_score": market.get("rule_score"),
                "llm_score": market.get("llm_score"),
            },
            "residual": residual.get("predicted_price"),
        },
        "news": news,
        "external_insights": insights,
        "analysis_text": (insights.get("vector", {}) or {}).get("enhanced_analysis"),
    }

    # Build a baseline, source-aware markdown that includes conflicts
    carsxe = (insights.get("carsxe", {}) or {}).get("raw") if isinstance(insights, dict) else None
    carsxe_avg = None
    if isinstance(carsxe, dict):
        for k in ("averageMarketPrice", "average_market_price", "average_price", "average"):
            if k in carsxe and isinstance(carsxe[k], (int, float)):
                carsxe_avg = float(carsxe[k])
                break

    issues = (state.get("consistency_report", {}) or {}).get("issues", [])
    early_brief = (state.get("early_rag", {}) or {}).get("brief")
    vector_brief = (insights.get("vector", {}) or {}).get("retrieved_info")

    def _fmt_currency(x):
        try:
            return f"${float(x):,.0f}"
        except Exception:
            return str(x)

    baseline_lines = []
    car = state.get("current_car", {}) or {}
    baseline_lines.append(f"# {car.get('year')} {car.get('make')} {car.get('model')}")
    baseline_lines.append("")
    baseline_lines.append("## Sources")
    baseline_lines.append(f"- Paid: {_fmt_currency(car.get('price_paid'))}")
    if market:
        baseline_lines.append(f"- Market median: {_fmt_currency(market.get('market_median'))} (Δ {_fmt_currency(market.get('price_delta'))}, {market.get('price_delta_pct', 0):+.1f}%), verdict: {market.get('deal_category')}")
        baseline_lines.append(f"- Rule score: {market.get('rule_score')} / LLM score: {market.get('llm_score')}")
    if residual.get("predicted_price"):
        baseline_lines.append(f"- Residual model: {_fmt_currency(residual.get('predicted_price'))}")
    if carsxe_avg is not None:
        baseline_lines.append(f"- CarsXE avg: {_fmt_currency(carsxe_avg)}")
    baseline_lines.append("")
    if early_brief:
        baseline_lines.append("## Early context")
        baseline_lines.append(early_brief)
        baseline_lines.append("")
    if vector_brief:
        baseline_lines.append("## Retrieved evidence")
        baseline_lines.append(str(vector_brief)[:800])
        baseline_lines.append("")
    if issues:
        baseline_lines.append("## Conflicts & cross-check")
        for it in issues:
            baseline_lines.append(f"- [{it.get('severity')}] {it.get('type')}: {it.get('details')} → {it.get('action')}")
        baseline_lines.append("")
    # simple synthesis
    baseline_lines.append("## Synthesis")
    if market.get("price_delta_pct") is not None:
        delta = market.get("price_delta_pct")
        if abs(delta) >= 15:
            direction = "overpriced" if delta > 0 else "undervalued"
            baseline_lines.append(f"Overall the deal looks {direction} by {delta:+.1f}% vs market median.")
        else:
            baseline_lines.append("Overall the deal is close to market median.")
    baseline_md = "\n".join(baseline_lines)

    # Optional LLM refinement of Markdown
    refined = None
    if get_llm is not None:
        try:
            llm = get_llm()
            prompt = (
                "Act as a senior car pricing analyst. Rewrite the following summary into a concise Markdown report with sections: Inputs, Sources, Conflicts, Synthesis, Recommendation. "
                "Explain disagreements (market vs residual vs CarsXE vs LLM) and give a short rationale. Do not invent numbers.\n\n"
                f"Baseline:\n{baseline_md}\n"
            )
            resp = await llm.ainvoke([("user", prompt)])
            refined = getattr(resp, "content", None) or str(resp)
        except Exception:
            refined = None

    # write analysis_text into summary for orchestrator to pick up
    summary["analysis_text"] = refined or baseline_md

    return {
        **logs,
        "summary_report": summary,
        "markdown_refined": refined,
        **log_agent_complete("summary", payload={"deal_category": market.get("deal_category")}),
    }
