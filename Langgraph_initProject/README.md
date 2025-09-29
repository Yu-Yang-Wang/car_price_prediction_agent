# 🚗 Car Analysis Agent Suite

多代理汽车分析系统：结合真实市场检索（Tavily）、两段式 RAG 检索增强、残值模型（joblib/XGBoost）、CarsXE（可选）与一致性校验，自动生成结构化 + Markdown 报告、冲突分析和建议。当前代码支持并行节点调度、可选 LLM 润色，并把所有结果写入 SQLite + Chroma 向量库。

---

## 🏗️ 架构概览

```
Condition
   │
   ├── Market (Tavily→规则→LLM)   ┐
   ├── Residual (ML joblib)       │ 并行
   ├── News (占位)                │
   ├── CarsXE (HTTP，可选)        │
   └── Early RAG (向量检索)       ┘
          ↓
Consistency（交叉校验）
          ↓
Late RAG（向量检索 + LLM 增强）
          ↓
Summary（汇总 + LLM 润色）
          ↓
Save to DB / Generate Report
```

- **早期 RAG**：不调用 LLM，仅检索相似案例/知识；生成 `early_rag.brief` 供中间 Agent 解释使用。
- **晚期 RAG**：带着市场/残值/CarsXE 数据构造检索 + LLM 增强，生成最终解释。
- **Consistency Agent**：检查 Rule vs LLM、Residual vs Market、CarsXE vs Market 等冲突，并可选 LLM 批注。
- **Summary Agent**：汇总所有来源，输出 baseline Markdown，并可选 LLM 润色为专业报告。

---

## 🤖 Agent 角色

| Agent | 文件 | 输出字段 | 说明 |
|-------|------|----------|------|
| Condition | `core/agents/condition.py` | `condition_report` | 解析事故、里程、clean title 等车况基础信息 |
| Early RAG | `core/agents/early_rag.py` | `early_rag` | 向量检索相似案例/知识，生成简短上下文供 LLM 引用 |
| Market | `core/agents/market.py` + `core/workers.py` | `market_analysis` | Tavily 搜索 → 价格过滤 → 规则评分 → LLM 二次意见 |
| Residual | `core/agents/residual.py` | `residual_analysis` | 调用 joblib pipeline 预测残值（`.env` 可配置 `CAR_ML_MODEL_PATH`）|
| News | `core/agents/news.py` | `news_analysis` | 占位，当前返回 `success=False`，可接外部政策/召回数据 |
| CarsXE | `core/agents/carsxe.py` | `rag_insights.carsxe` | 调 CarsXE `/v1/ymm` 或 `/v1/vin`（需 `CARSXE_API_KEY`） |
| Consistency | `core/agents/consistency.py` | `consistency_report` | Rule/Residual/CarsXE/LLM 交叉对比并输出冲突列表 + 可选 LLM 批注 |
| Late RAG | `core/agents/rag.py` | `rag_insights.vector` | 调 `RAGSystem`（向量检索 + LLM）生成增强分析 |
| Summary | `core/agents/summary.py` | `summary_report`, `markdown_refined` | 汇总所有来源，生成 baseline Markdown + 可选 LLM 润色 |

---

## 📦 数据 & 向量库

### SQLite（`database/car_analysis.db`）
- `cars`：车辆基础信息
- `car_analyses`：规则/LLM/残值等分析结果
- `knowledge_base`、`market_data`、`analysis_sessions`、`user_queries` 等辅助表

### Chroma 向量库（`database/chroma_db/`）
- 集合：`cars`、`analyses`、`knowledge`、`market_data`
- 默认使用 OpenAI `text-embedding-ada-002`；缺 key 时 fallback HuggingFace `all-MiniLM-L6-v2`
- 写库时自动同步嵌入（`VectorStoreManager.add_*`）

### CSV 导入

```
python -m car_analysis.utils.ingest_csv <dataset> --csv path.csv --limit N [--offset M]
```
- `car_prices`：含 VIN + MMR → 生成历史成交知识
- `used_cars`：事故/clean title/燃油等
- `used_cars_data`：高维特征（horsepower、torque、seller_rating、daysonmarket ...）
- 避免重复：根据已导入条数计算 offset，例如：
  ```bash
  sqlite3 database/car_analysis.db "SELECT COUNT(*) FROM car_analyses WHERE data_source='used_cars_data_csv';"
  python -m car_analysis.utils.ingest_csv used_cars_data --csv used_cars_data.csv --limit 500 --offset <count>
  ```

---

## 🔢 置信度 / 相似度 / 冲突计算

- **向量相似度**：Chroma 返回 `similarity = 1 - cosine_distance`（0–1）。Early/Late RAG 会展示相似案例及知识条目。
- **市场差价**：`price_delta = price_paid - market_median`；百分比 `price_delta_pct = price_delta / market_median * 100`。
- **规则评分**：`deal_scoring_worker` 综合差价、里程、车龄、样本数量计算 0–100 分。
- **RAG 置信度**：`RAGSystem._calculate_confidence` 基于检索条数/权重归一化（0–1）。
- **冲突检测（Consistency Agent）**：
  - Rule vs LLM 分数差 ≥ 25
  - Residual vs Market 偏离 ≥ 20%
  - CarsXE vs Market 偏离 ≥ 15%
  - delta% > 15 且 LLM 分数 ≥ 75
  - 每项附带 severity / Details / Action，必要时 LLM 批注简评。

---

## 🧠 LLM 调用点

| 场景 | 模块 | 用途 |
|------|------|------|
| 市场分析二次意见 | `core/workers.py` (`llm_opinion_worker`) | 结合 early RAG 简述 + Tavily 数据解释差价 |
| RAG 增强分析 | `core/agents/rag.py` + `rag/rag_system.py` | 检索向量信息后交给 LLM 综合说明 |
| 冲突批注（可选） | `core/agents/consistency.py` | 将 rule-based issues 输入 LLM 给出简短 QA |
| Summary 润色（可选） | `core/agents/summary.py` | 将 baseline Markdown 转为专业结构化报告 |
| 早期 RAG | `core/agents/early_rag.py` | 仅向量检索，不调 LLM |

缺 `OPENAI_API_KEY` 时，上述调用会自动降级为 `success=False`，流程继续执行。

---

## 📥 输入来源

- 单车字典：`{'year', 'make', 'model', 'mileage', 'price_paid', ...}`
  - 来源：CSV 导入后的 DB、PDF 解析(`utils/pdf_extractor.py`)、手动构造
- Demo：`python -m car_analysis.tests.agents_demo --limit 2 [--save-md]`
  - 默认从 DB（`cars` 表）获取最新 N 条车；`--save-md` 会生成 `outputs/agent_report_*.md`
- LangGraph 主流程：`car_analysis/test_quick.py` 或直接调用 `process_single_car`

---

## 📤 输出内容

- `state['summary_report']`：结构化摘要（包含 `analysis_text`）
- `state['markdown_refined']`：如 LLM 成功润色则包含专业 Markdown
- `state['consistency_report']`：冲突列表 + 可选批注
- `outputs/agent_report_*.md`：运行 demo 并 `--save-md` 时生成的 Markdown 报告
- JSON 输出：`car_analysis/outputs/*.json` 保存 LangGraph 执行结果

### 示例报告结构（baseline Markdown）

```
# 2015 Kia Sorento
## Sources
- Paid: $21,000
- Market median: $12,791 (Δ +$8,209, +64.2%), verdict: Overpaid
- Rule score: 40 / LLM score: 38
- Residual model: $15,200

## Early context
<early_rag 简述>

## Retrieved evidence
<向量检索片段>

## Conflicts & cross-check
- [medium] residual_vs_market: ...
- [low] delta_high_but_llm_positive: ...

## Synthesis
Overall the deal looks overpriced by +64.2% vs market median.
```

若 LLM 润色成功，`markdown_refined` 会将以上内容格式化为“Inputs / Sources / Conflicts / Recommendation”等段落。

---

## 🧪 快速体验

```bash
# 激活环境
source .langgraphvenv/bin/activate

# 导入 CSV（可按需分批，每次调节 offset 防重）
python -m car_analysis.utils.ingest_csv car_prices --csv car_prices.csv --limit 500

# 数据验证
sqlite3 database/car_analysis.db "SELECT COUNT(*) FROM cars;"
python car_analysis/db_manager_cli.py --stats

# Agent demo（打印 & 保存 Markdown）
python -m car_analysis.tests.agents_demo --limit 2 --save-md
# 查看 outputs/agent_report_*.md

# 主流程示例
python car_analysis/test_quick.py
```

`.env` 示例：
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
CARSXE_API_KEY=...          # 未激活时 CarsXE 会输出占位
CAR_ML_MODEL_PATH=car_analysis/models/used_car_price_rf.joblib
```

---

## ⚙️ 调参与扩展建议

- **News Agent**：接召回/政策 API（写入 `news_analysis`）。
- **CarsXE**：激活 key 后可改为 `/v1/vin` 查询；暂无 key 时可临时禁用。
- **Early RAG**：提高过滤精度（make/model/year±1、mileage±30%、reliability 权重）。
- **Consistency**：调整冲突阈值（在 `consistency.py` 修改）。
- **Summary 提示词**：修改 `core/agents/summary.py` 里的 prompt 或 fallback 文案。
- **模型更新**：使用新 CSV/Kaggle 数据训练 pipeline，导出 joblib 覆盖 `CAR_ML_MODEL_PATH`。
- **两段式 RAG**：继续优化 early/late 阶段的检索策略，确保输出依据更可靠。

欢迎根据业务需求继续拓展（例如缓存 Embedding、无键 fallback、更多知识源）。如需帮助实现自动 offset 管理、指定 car_id 跑 demo 或更细粒度的冲突解释，请继续提出。祝分析顺利！
