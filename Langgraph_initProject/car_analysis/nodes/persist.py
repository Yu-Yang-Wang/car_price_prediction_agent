from ..state_types import GraphState
# 关系数据库
from ..db.orm_models import Base, Report, BalanceSheet, CashFlow, IncomeStatement
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 向量库（可选）
from ..vec.vectordb import VectorStore

def store_to_db(state: GraphState) -> GraphState:
    engine = create_engine("sqlite:///financial_reports.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()

    meta = state.get("report_meta", {"company":"DemoCorp","date":"2025-09-01"})
    agg  = state.get("agg_report", {})
    tables = agg.get("tables", {})

    r = Report(company_name=meta.get("company"), report_date=meta.get("date"), pdf_path=state.get("pdf_path","demo.pdf"))
    s.add(r); s.flush()

    bs = tables.get("BS", {})
    s.add(BalanceSheet(
        report_id=r.id,
        total_assets=bs.get("total_assets"),
        total_liabilities_equity=bs.get("total_liilities_equity") or bs.get("total_liabilities_equity"),
        cash=bs.get("cash"),
        inventory=bs.get("inventory"),
    ))

    cf = tables.get("CF", {})
    s.add(CashFlow(
        report_id=r.id,
        net_cash=cf.get("net_cash"),
        cash_change=cf.get("cash_change"),
    ))

    is_ = tables.get("IS", {})
    s.add(IncomeStatement(
        report_id=r.id,
        revenue=is_.get("revenue"),
        profit=is_.get("profit"),
    ))

    s.commit(); s.close()
    return {"dbg_logs": "stored to RDBMS"}

def store_to_vectordb(state: GraphState) -> GraphState:
    vec = VectorStore(persist_path=".chroma")
    chunks = []
    for i, page in enumerate(state.get("pdf_pages", [])):
        chunks.append({"id": f"p{i}", "text": page, "meta": {"page": i}})
    keys = vec.add(chunks)  # 返回 ids
    return {"vectordb_keys": ["mdna:0","mdna:1"], "dbg_logs": ["stored to VectorDB"]}
