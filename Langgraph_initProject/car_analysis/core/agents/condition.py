"""Condition assessment agent."""

from __future__ import annotations

from typing import Any, Dict

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)


def _basic_car_context(state: CarAnalysisState) -> Dict[str, Any]:
    car = state.get("current_car", {}) or {}
    return {
        "year": car.get("year"),
        "make": car.get("make"),
        "model": car.get("model"),
        "mileage": car.get("mileage"),
        "price_paid": car.get("price_paid"),
    }


async def condition_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Basic condition assessment from raw car metadata."""

    logs = log_agent_start("condition", payload=_basic_car_context(state))

    car = state.get("current_car", {}) or {}
    try:
        accident_text = car.get("condition") or car.get("accident_history") or "Unknown"
        clean_title = car.get("clean_title", 0)
        report = {
            "success": True,
            "summary": f"{car.get('year')} {car.get('make')} {car.get('model')}",
            "mileage": car.get("mileage"),
            "condition_flags": {
                "accident_history": accident_text,
                "clean_title": bool(clean_title),
            },
            "raw": car,
        }
        return {
            "condition_report": report,
            **logs,
            **log_agent_complete("condition", payload={"clean_title": bool(clean_title)})
        }
    except Exception as exc:
        return {
            "condition_report": {"success": False, "error": str(exc)},
            **logs,
            **log_agent_error("condition", exc),
        }
