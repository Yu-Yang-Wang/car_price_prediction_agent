"""Car Analysis Package - Multi-Car Price Analysis with LangGraph and Tavily"""

from .core.models import Car, CarAnalysisState
from .core.graph import build_car_analysis_graph, build_single_car_graph
from .core.orchestrator import process_single_car, generate_final_report
from .core.agents import (
    condition_agent,
    market_price_agent,
    residual_value_agent,
    news_policy_agent,
    carsxe_agent,
    rag_vector_agent,
    summary_agent,
)

try:  # optional heavy dependency
    from .utils.pdf_extractor import extract_cars_from_pdf  # type: ignore
except Exception:  # pragma: no cover - optional
    extract_cars_from_pdf = None  # type: ignore

__all__ = [
    "Car",
    "CarAnalysisState",
    "extract_cars_from_pdf",
    "build_car_analysis_graph",
    "build_single_car_graph",
    "process_single_car",
    "generate_final_report",
    "condition_agent",
    "market_price_agent",
    "residual_value_agent",
    "news_policy_agent",
    "carsxe_agent",
    "rag_vector_agent",
    "summary_agent",
]
