# financial_agents/nodes/workers_subgraphs.py (补充/参考)
import asyncio
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..state_types import GraphState
from .prompts import PARSER_SYS, PARSER_USER_TEMPLATE

# 你项目里应该有获取 llm/tool 的统一方法
from .tools import fetch_pdf_text_for_table, get_llm

TABLE_SCHEMA_HINTS = {
    "BS": """{
  "table": "BS",
  "currency": "string|null",
  "unit": "string|null",
  "period_end": "string|null",
  "items": {
     "total_assets": "number|null",
     "total_liabilities": "number|null",
     "total_equity": "number|null",
     "notes": "array<string>"
  }
}""",
    "CF": """{
  "table": "CF",
  "currency": "string|null",
  "unit": "string|null",
  "period_end": "string|null",
  "items": {
    "net_cash_from_operating": "number|null",
    "net_cash_from_investing": "number|null",
    "net_cash_from_financing": "number|null",
    "net_change_in_cash": "number|null",
    "notes": "array<string>"
  }
}""",
    "IS": """{
  "table": "IS",
  "currency": "string|null",
  "unit": "string|null",
  "period_end": "string|null",
  "items": {
    "revenue": "number|null",
    "cogs": "number|null",
    "gross_profit": "number|null",
    "operating_income": "number|null",
    "net_income": "number|null",
    "notes": "array<string>"
  }
}""",
}

async def _text_for_table_async(state: GraphState, table: str) -> str:
    pdf_path = state.get("pdf_path")
    page_range = state.get("page_range")  # 可空
    return await fetch_pdf_text_for_table(pdf_path, page_range)

async def parse_table(state: GraphState, table: str, text: str) -> Dict[str, Any]:
    llm = get_llm()  # 例如 ChatOpenAI(...), 已在你项目里封装
    prompt = ChatPromptTemplate.from_messages([
        ("system", PARSER_SYS),
        ("user", PARSER_USER_TEMPLATE),
    ])
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({
        "table": table,
        "schema_hint": TABLE_SCHEMA_HINTS[table],
        "context": text[:6000]  # 安全截断
    })

def normalize_table(table: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    # 你已有实现；这里放个占位
    return raw

def check_table(table: str, norm: Dict[str, Any]) -> str:
    # 你已有实现；这里放个简单校验示例
    if table == "BS":
        items = (norm or {}).get("items", {})
        a = items.get("total_assets"); l = items.get("total_liabilities"); e = items.get("total_equity")
        # 如果三者都有数，做一个松的校验
        if all(isinstance(x, (int, float)) for x in (a, l, e)):
            if abs(a - (l + e)) > max(1.0, 0.02 * abs(a)):
                return "BS not balanced: assets != liabilities + equity"
    return ""
