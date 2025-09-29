"""Structured logging helpers for multi-agent workflow.

Each helper returns a partial state update compatible with LangGraph state
reducers. Usage pattern inside an agent function:

```
from car_analysis.core.agent_logging import log_agent_start, log_agent_complete

state = {**state, **log_agent_start("market_price", {"make": make})}
...
return {
    "market_analysis": analysis_result,
    **log_agent_complete("market_price", summary="Fetched 12 comps"),
}
```
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def _make_entry(agent: str, event: str, message: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent": agent,
        "event": event,
        "message": message,
        "payload": payload or {},
    }


def _append_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "agent_logs": [entry],
        "dbg_logs": [f"[{entry['agent']}] {entry['event']}: {entry['message']}"],
    }


def log_agent_start(agent: str, payload: Optional[Dict[str, Any]] = None, message: str = "start") -> Dict[str, Any]:
    entry = _make_entry(agent, "start", message, payload)
    return _append_entry(entry)


def log_agent_complete(agent: str, payload: Optional[Dict[str, Any]] = None, message: str = "complete") -> Dict[str, Any]:
    entry = _make_entry(agent, "complete", message, payload)
    return _append_entry(entry)


def log_agent_error(agent: str, error: Exception, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entry = _make_entry(agent, "error", str(error), payload)
    return _append_entry(entry)

