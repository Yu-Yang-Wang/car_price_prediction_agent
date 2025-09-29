"""Placeholder news/policy agent."""

from __future__ import annotations

from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agent_logging import (
    log_agent_start,
    log_agent_complete,
)
from .condition import _basic_car_context


async def news_policy_agent(state: CarAnalysisState) -> CarAnalysisState:
    """Stub agent for news/policy impact (placeholder for future integration)."""

    logs = log_agent_start("news_policy", payload=_basic_car_context(state))

    report = {
        "success": False,
        "error": "News/policy analysis not yet implemented",
    }
    return {
        **logs,
        "news_analysis": report,
        **log_agent_complete("news_policy", payload={"implemented": False}),
    }
