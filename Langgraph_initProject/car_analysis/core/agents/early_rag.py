"""Early RAG agent: lightweight retrieval to support mid-pipeline agents.

Does NOT call LLM. It retrieves brief context from the vector store and stores
it under `state['early_rag']`, so Market/Residual agents can reference it in
their prompts or explanations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)
from car_analysis.rag.embeddings import EmbeddingManager
from car_analysis.rag.vector_store import VectorStoreManager
from .condition import _basic_car_context


def _format_brief(similar: List[Dict[str, Any]], knowledge: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if similar:
        lines.append("Similar cases (top 3):")
        for item in similar[:3]:
            md = item.get("metadata", {})
            year = md.get("year")
            make = md.get("make")
            model = md.get("model")
            price = md.get("price_paid")
            sim = item.get("similarity", 0)
            lines.append(f"- {year} {make} {model}, paid ${price:,.0f} (sim {sim:.2f})")
    if knowledge:
        lines.append("Knowledge snippets (top 2):")
        for item in knowledge[:2]:
            md = item.get("metadata", {})
            title = md.get("title") or md.get("category") or "entry"
            sim = item.get("similarity", 0)
            doc = (item.get("document") or "").strip()
            if len(doc) > 140:
                doc = doc[:140] + "…"
            lines.append(f"- {title} (sim {sim:.2f}): {doc}")
    return "\n".join(lines) if lines else ""


async def early_rag_agent(state: CarAnalysisState) -> CarAnalysisState:
    logs = log_agent_start("early_rag", payload=_basic_car_context(state))

    try:
        car = state.get("current_car", {}) or {}

        embedding_manager = EmbeddingManager()
        vector_manager = VectorStoreManager(embedding_manager=embedding_manager)

        # Build a simple text query and also try car-based similarity
        make = car.get("make", "")
        model = car.get("model", "")
        year = car.get("year", 0)
        query_text = f"{year} {make} {model} used car pricing factors"

        # Cross-collection search (knowledge + analyses)
        search = vector_manager.semantic_search(query_text, collections=["knowledge", "analyses"], limit=5)
        knowledge_items = (search.get("knowledge") or []) + (search.get("analyses") or [])

        # Similar cars by embedding
        similar_cars = vector_manager.search_similar_cars(query_car=car, limit=5, similarity_threshold=0.6)

        brief = _format_brief(similar_cars, knowledge_items)
        early = {
            "success": True,
            "similar_cars": similar_cars,
            "knowledge_items": knowledge_items,
            "brief": brief,
        }

        return {
            **logs,
            "early_rag": early,
            **log_agent_complete("early_rag", payload={"has_brief": bool(brief)})
        }

    except Exception as exc:
        return {
            **logs,
            "early_rag": {"success": False, "error": str(exc)},
            **log_agent_error("early_rag", exc),
        }

