# -*- coding: utf-8 -*-
from typing import Literal
from ..state_types import GraphState, TABLES

# 允许最多迭代 1 次（你可以改成 2/3）
MAX_REFINE_STEPS = 1

def score_report(state: GraphState) -> GraphState:
    have = set((state.get("results") or {}).keys())
    ok = 1.0 if TABLES <= have else 0.6
    # 不覆盖已有的 refine_iters
    metrics = dict(state.get("metrics", {}))
    metrics["quality"] = ok
    return {"metrics": metrics, "dbg_logs": [f"Score quality={ok}"]}

def refine_once(state: GraphState) -> GraphState:
    # 仅示意：对 agg_report 做一次轻微补全
    agg = dict(state.get("agg_report", {}))
    agg["notes"] = "refined once"

    metrics = dict(state.get("metrics", {}))
    metrics["refine_iters"] = int(metrics.get("refine_iters", 0)) + 1

    return {"agg_report": agg, "metrics": metrics, "dbg_logs": ["Refine once"]}

def refine_router(state: GraphState) -> Literal["refine", "done"]:
    metrics = state.get("metrics", {}) or {}
    quality = float(metrics.get("quality", 0.0))
    iters = int(metrics.get("refine_iters", 0))

    # ✅ 关键：如果有缺表（即使 quality<0.9），也直接结束，避免死循环
    have = set((state.get("results") or {}).keys())
    missing = list(TABLES - have)
    if missing:
        return "done"

    # 没缺表时，最多迭代 MAX_REFINE_STEPS 次
    if quality < 0.9 and iters < MAX_REFINE_STEPS:
        return "refine"
    return "done"
