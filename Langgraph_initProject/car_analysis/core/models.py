"""Data models for car analysis"""

from __future__ import annotations

from typing import Optional, TypedDict, Any, List, Dict
from typing_extensions import Annotated
from dataclasses import dataclass


def merge_dict(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(a or {})
    out.update(b or {})
    return out


def append_list(a: list[Any] | None, b: list[Any] | None) -> list[Any]:
    return list(a or []) + list(b or [])


@dataclass
class Car:
    """Car data structure"""
    make: str
    model: str
    year: int
    mileage: int
    price_paid: float
    index: int  # Position in PDF
    raw_text: str = ""


class CarAnalysisState(TypedDict, total=False):
    """State for car analysis workflow"""
    # Core fields
    cars: Annotated[list[dict[str, Any]], append_list]
    current_car: dict[str, Any]
    car_index: int

    # Primary worker / agent outputs
    price_research: dict[str, Any]
    price_comparison: dict[str, Any]
    # Allow safe overwrites even if concurrent branches re-enter the scorer
    deal_score: Annotated[dict[str, Any], merge_dict]

    # Final analysis
    car_reports: Annotated[list[dict[str, Any]], append_list]
    llm_opinion: Annotated[Optional[dict[str, Any]], merge_dict]  # LLM 输出的打分、判断
    score_disagree_retry: Optional[bool]   # 是否触发 retry

    # Multi-agent enrichments
    condition_report: Annotated[dict[str, Any], merge_dict]
    market_analysis: Annotated[dict[str, Any], merge_dict]
    residual_analysis: Annotated[dict[str, Any], merge_dict]
    news_analysis: Annotated[dict[str, Any], merge_dict]
    rag_insights: Annotated[dict[str, Any], merge_dict]
    summary_report: Annotated[dict[str, Any], merge_dict]

    # Error tracking
    analysis_errors: list[str]  # Track all errors encountered
    failed_permanently: bool    # Mark if analysis failed after max retries

    # Shared fields for LangGraph
    retries: Annotated[dict[str, int], merge_dict]
    dbg_logs: Annotated[list[str], append_list]
    agent_logs: Annotated[List[Dict[str, Any]], append_list]

    # Validation and routing
    research_ok: bool
    research_final: bool
    comparison_ok: bool
    comparison_final: bool
    scoring_ok: bool
    scoring_final: bool
