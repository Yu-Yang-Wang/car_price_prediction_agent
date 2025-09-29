"""CarsXE market valuation agent."""

from __future__ import annotations

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)
from car_analysis.tools.carsxe_api import carsxe_client
from .condition import _basic_car_context


async def carsxe_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Optional CarsXE valuation agent."""

    logs = log_agent_start("carsxe_market", payload=_basic_car_context(state))
    car = state.get("current_car", {}) or {}

    if not carsxe_client.available:
        return {
            **logs,
            "rag_insights": {
                "carsxe": {
                    "success": False,
                    "error": "CarsXE API disabled",
                }
            },
            **log_agent_complete("carsxe_market", payload={"available": False}),
        }

    try:
        response = carsxe_client.fetch_market_value_by_trim(
            make=car.get("make"),
            model=car.get("model"),
            year=car.get("year"),
            mileage=car.get("mileage"),
        )
        return {
            **logs,
            "rag_insights": {
                "carsxe": {
                    "success": True,
                    "raw": response,
                }
            },
            **log_agent_complete("carsxe_market", payload={"success": True}),
        }
    except Exception as exc:
        return {
            **logs,
            "rag_insights": {
                "carsxe": {
                    "success": False,
                    "error": str(exc),
                }
            },
            **log_agent_error("carsxe_market", exc),
        }
