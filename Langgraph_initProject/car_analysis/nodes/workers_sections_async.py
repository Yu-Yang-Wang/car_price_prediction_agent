# financial_agents/nodes/workers_sections_async.py
from __future__ import annotations
import asyncio
from typing import Dict
from langgraph.graph import StateGraph
from langgraph.constants import START, END

from ..state_types import GraphState, MAX_RETRIES, TABLES
from .workers_subgraphs import _text_for_table_async, parse_table, normalize_table, check_table

# -------- Worker / Checker（异步） --------
def table_worker_node_async(table: str):
    async def node(state: GraphState) -> GraphState:
        if state.get(f"__final_{table}", False):
            return {}
        tries = (state.get("retries") or {}).get(table, 0)
        text = await _text_for_table_async(state, table)  # <--- 用 MCP 拿文本
        raw = await parse_table(state, table, text)
        return {f"__tmp_{table}": {"raw": raw, "tries": tries},
                "dbg_logs": [f"{table}_Worker(parsed, try={tries})"]}
    return node

def table_checker_node_async(table: str):
    async def node(state: GraphState) -> GraphState:
        retries = dict(state.get("retries", {})); retries.setdefault(table, 0)

        tmp = state.get(f"__tmp_{table}", {})
        raw = tmp.get("raw", {})

        # 归一化 + 规则校验（保持和你 sync 版本一致）
        norm = await normalize_table(table, raw) if asyncio.iscoroutinefunction(normalize_table) \
               else normalize_table(table, raw)
        err  = await check_table(table, norm) if asyncio.iscoroutinefunction(check_table) \
               else check_table(table, norm)

        if err:
            retries[table] += 1
            if retries[table] > MAX_RETRIES:
                return {
                    "retries": {table: retries[table]},
                    f"__ok_{table}": False, f"__final_{table}": True,
                    f"__err_{table}": err,
                    "dbg_logs": [f"{table}_Checker FAILED(final): {err}"]
                }
            # 轻退避（异步）
            await asyncio.sleep(0.05)
            return {
                "retries": {table: retries[table]},
                f"__ok_{table}": False, f"__final_{table}": False,
                f"__err_{table}": err,
                "dbg_logs": [f"{table}_Checker retry {retries[table]}: {err}"]
            }

        return {
            "results": {table: norm},
            "retries": {table: retries[table]},
            f"__ok_{table}": True, f"__final_{table}": True,
            f"__err_{table}": "",
            "dbg_logs": [f"{table}_Checker done"]
        }
    return node

# -------- Join / Router（与 sync 版一致） --------
def join_gate(state: GraphState) -> GraphState:
    ready = all(state.get(f"__final_{t}", False) for t in TABLES)
    return {"_join_ready": ready, "dbg_logs": [f"Join {'go' if ready else 'wait'}"]}

def join_router(state: GraphState) -> str:
    return "go" if state.get("_join_ready") else "wait"

def await_noop(state: GraphState) -> GraphState:
    return {"dbg_logs": ["Await..."]}

# ============================================================
# 子图（异步节点）—— 仍然一次性把三张表并行跑完
# ============================================================
def make_sections_subgraph_async():
    sg = StateGraph(GraphState)

    for t in TABLES:  # 通常为 ["BS","CF","IS"]
        sg.add_node(f"{t}_Worker", table_worker_node_async(t))
        sg.add_node(f"{t}_Checker", table_checker_node_async(t))
        sg.add_edge(START, f"{t}_Worker")
        sg.add_edge(f"{t}_Worker", f"{t}_Checker")

        def _router_factory(table=t):
            def _r(state: GraphState) -> str:
                final = state.get(f"__final_{table}", False)
                ok = state.get(f"__ok_{table}", False)
                in_results = table in (state.get("results") or {})
                return "done" if final and (ok == in_results or (not ok)) else "retry"
            return _r

        sg.add_conditional_edges(
            f"{t}_Checker",
            _router_factory(),
            {"retry": f"{t}_Worker", "done": "Join"},
        )

    sg.add_node("Join", join_gate)
    sg.add_node("Await", await_noop)
    sg.add_conditional_edges("Join", join_router, {"wait": "Await", "go": END})
    sg.add_edge("Await", "Join")

    return sg.compile()

__all__ = [
    "make_sections_subgraph_async",
    "table_worker_node_async",
    "table_checker_node_async",
]
