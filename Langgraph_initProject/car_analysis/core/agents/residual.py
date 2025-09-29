"""Residual value prediction agent using the ML predictor tool."""

from __future__ import annotations

from typing import Any, Dict

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
    log_agent_error,
)
from car_analysis.tools.ml_predictor import ml_predictor
from .condition import _basic_car_context


async def residual_value_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Residual / resale value projection using the ML predictor tool."""

    logs = log_agent_start("residual_value", payload=_basic_car_context(state))
    car = state.get("current_car", {}) or {}

    try:
        if not ml_predictor.available:
            raise RuntimeError("ML predictor unavailable; ensure joblib model is loaded")

        features = {
            "year": car.get("year"),
            "mileage": car.get("mileage"),
            "hp": car.get("hp") or car.get("horsepower"),
            "engine_displacement": car.get("engine_displacement") or car.get("engine_liters"),
            "fuel_type": car.get("fuel_type"),
            "transmission": car.get("transmission"),
            "is_v_engine": car.get("is_v_engine"),
            "clean_title": car.get("clean_title"),
        }

        result = ml_predictor.predict_price(features)
        report = {
            "success": True,
            "predicted_price": result.get("predicted_price"),
            "features_used": result.get("features_used"),
        }

        return {
            **logs,
            "residual_analysis": report,
            **log_agent_complete(
                "residual_value",
                payload={"predicted_price": result.get("predicted_price")},
            ),
        }
    except Exception as exc:
        return {
            **logs,
            "residual_analysis": {"success": False, "error": str(exc)},
            **log_agent_error("residual_value", exc, payload=car),
        }
