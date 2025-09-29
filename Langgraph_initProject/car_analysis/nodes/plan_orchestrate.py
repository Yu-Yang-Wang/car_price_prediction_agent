# -*- coding: utf-8 -*-
from typing import Literal, Dict, Any, List
from ..state_types import GraphState, TABLES

def planner(state: GraphState) -> GraphState:
    sections = []
    for i, text in enumerate(state["pdf_pages"]):
        low = text.lower()
        if "balance sheet" in low: sections.append({"type":"BS","start":i,"end":i})
        if "cash flow" in low: sections.append({"type":"CF","start":i,"end":i})
        if "income statement" in low or "operations" in low: sections.append({"type":"IS","start":i,"end":i})
        if "md&a" in low or "management" in low: sections.append({"type":"MDNA","start":i,"end":i})
    return {"plan": {"sections": sections}, "dbg_logs": ["Planner done"]}

def router(state: GraphState) -> GraphState:
    types = {s["type"] for s in state["plan"]["sections"]}
    route = "full_pipeline" if TABLES <= types else "summary_only"
    return {"route": route, "dbg_logs": [f"Router -> {route}"]}

def orchestrator(state: GraphState) -> GraphState:
    tasks = []
    for s in state["plan"]["sections"]:
        if s["type"] in TABLES:
            tasks.append({"kind":"TABLE","table":s["type"],"range":(s["start"],s["end"])})
        elif s["type"] == "MDNA":
            tasks.append({"kind":"UNSTRUCTURED","section":"MDNA","range":(s["start"],s["end"])})
    return {"tasks": tasks, "retries": state.get("retries", {}), "results": state.get("results", {}), "dbg_logs": ["Orchestrator ready"]}


def route_decider(state: GraphState) -> Literal["summary_only", "full_pipeline"]:
    """
    条件边判定函数：只返回分支名（给 add_conditional_edges 用）
    """
    return state.get("route", "summary_only")
