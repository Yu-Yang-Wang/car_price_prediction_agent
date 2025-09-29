"""LangGraph workflow definition for car analysis"""

from langgraph.graph import StateGraph
from langgraph.constants import START, END

from .models import CarAnalysisState
from ..utils.pdf_extractor import extract_cars_from_pdf
from .agents import (
    condition_agent,
    market_price_agent,
    residual_value_agent,
    news_policy_agent,
    carsxe_agent,
    rag_vector_agent,
    summary_agent,
)
from .rag_enhanced_workers import save_analysis_to_database
from .orchestrator import aggregate_car_reports


def build_car_analysis_graph():
    """Build the LangGraph workflow for complete PDF car analysis"""

    # Create the graph
    workflow = StateGraph(CarAnalysisState)

    # Add nodes
    workflow.add_node("extract_cars", extract_cars_from_pdf)
    workflow.add_node("generate_report", aggregate_car_reports)

    # Set entry point
    workflow.set_entry_point("extract_cars")

    # Simple workflow for PDF processing
    workflow.add_edge("extract_cars", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def build_single_car_graph():
    """Build the LangGraph workflow for single car analysis with multi-agent nodes."""

    workflow = StateGraph(CarAnalysisState)

    workflow.add_node("condition_agent", condition_agent)
    workflow.add_node("market_agent", market_price_agent)
    workflow.add_node("residual_agent", residual_value_agent)
    workflow.add_node("early_rag_agent", early_rag_agent)
    workflow.add_node("news_agent", news_policy_agent)
    workflow.add_node("carsxe_agent", carsxe_agent)
    workflow.add_node("rag_agent", rag_vector_agent)
    workflow.add_node("consistency_agent", consistency_agent)
    workflow.add_node("summary_agent", summary_agent)
    workflow.add_node("save_to_database", save_analysis_to_database)
    workflow.add_node("generate_report", aggregate_car_reports)

    workflow.set_entry_point("condition_agent")

    # condition completes first
    workflow.add_edge("condition_agent", "market_agent")
    workflow.add_edge("condition_agent", "residual_agent")
    workflow.add_edge("condition_agent", "news_agent")
    workflow.add_edge("condition_agent", "carsxe_agent")
    workflow.add_edge("condition_agent", "early_rag_agent")

    # fan-in before rag/summary once all parallel agents done
    workflow.add_conditional_edges(
        "market_agent",
        lambda state: "carsxe_agent" if state.get("market_analysis") else "market_agent",
        {"carsxe_agent": "carsxe_agent", "market_agent": "market_agent"}
    )
    workflow.add_conditional_edges(
        "residual_agent",
        lambda state: "carsxe_agent" if state.get("residual_analysis") else "residual_agent",
        {"carsxe_agent": "carsxe_agent", "residual_agent": "residual_agent"}
    )
    workflow.add_conditional_edges(
        "news_agent",
        lambda state: "carsxe_agent",
        {"carsxe_agent": "carsxe_agent"}
    )

    workflow.add_conditional_edges(
        "carsxe_agent",
        lambda state: (
            "consistency_agent"
            if all(state.get(k) is not None for k in ("market_analysis", "residual_analysis", "news_analysis"))
            else "carsxe_agent"
        ),
        {"consistency_agent": "consistency_agent", "carsxe_agent": "carsxe_agent"}
    )

    workflow.add_edge("consistency_agent", "rag_agent")
    workflow.add_edge("rag_agent", "summary_agent")
    workflow.add_edge("summary_agent", "save_to_database")
    workflow.add_edge("save_to_database", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()
