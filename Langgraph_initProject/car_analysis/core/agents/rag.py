"""RAG vector retrieval agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)
from car_analysis.rag.rag_system import RAGSystem
from .condition import _basic_car_context


logger = logging.getLogger(__name__)

_RAG_SYSTEM: Optional[RAGSystem] = None


def _get_rag_system() -> Optional[RAGSystem]:
    global _RAG_SYSTEM
    if _RAG_SYSTEM is not None:
        return _RAG_SYSTEM
    try:
        _RAG_SYSTEM = RAGSystem()
        logger.info("RAG system initialised for vector insights")
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning("RAG system unavailable: %s", exc)
        _RAG_SYSTEM = None
    return _RAG_SYSTEM


async def rag_vector_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Retrieve similar cases and enhanced analysis via RAG system."""

    logs = log_agent_start("rag_vector", payload=_basic_car_context(state))
    rag_system = _get_rag_system()

    if rag_system is None:
        return {
            **logs,
            "rag_insights": {
                "vector": {
                    "success": False,
                    "error": "RAG system unavailable",
                }
            },
            **log_agent_complete("rag_vector", payload={"available": False}),
        }

    car = state.get("current_car", {}) or {}
    market = state.get("market_analysis", {}) or {}

    analysis_context_parts = []
    if market.get("market_median"):
        analysis_context_parts.append(
            f"Market median price ${market['market_median']:,.0f}"
        )
    if market.get("deal_category"):
        analysis_context_parts.append(f"Deal category {market['deal_category']}")
    analysis_context = ". ".join(analysis_context_parts) or None

    try:
        similar_cases = rag_system.find_similar_cases(car)
        enhanced = rag_system.enhance_car_analysis(car, analysis_context=analysis_context)
        return {
            **logs,
            "rag_insights": {
                "vector": {
                    "success": True,
                    "similar_cases": similar_cases.get("similar_cases"),
                    "cases_analysis": similar_cases.get("analysis"),
                    "enhanced_analysis": enhanced.get("enhanced_analysis"),
                    "retrieved_info": enhanced.get("retrieved_info"),
                    "confidence": enhanced.get("rag_confidence"),
                }
            },
            **log_agent_complete(
                "rag_vector",
                payload={"cases_found": len(similar_cases.get("similar_cases", []))},
            ),
        }
    except Exception as exc:
        return {
            **logs,
            "rag_insights": {
                "vector": {
                    "success": False,
                    "error": str(exc),
                }
            },
            **log_agent_error("rag_vector", exc),
        }
