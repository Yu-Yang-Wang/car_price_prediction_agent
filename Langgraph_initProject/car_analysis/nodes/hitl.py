from langgraph.constants import INTERRUPT
from ..state_types import GraphState

def human_gate(state: GraphState) -> GraphState:
    # 示例：当 BS 缺关键字段时，触发 HITL
    bs = state.get("results", {}).get("BS", {})
    need = any(k not in bs for k in ["total_assets","total_liabilities_equity"])
    if need:
        return INTERRUPT({"reason": "missing_fields", "which": "BS"})
    return {"human_ok": True}
