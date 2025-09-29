# financial_agents/nodes/prompts.py

PARSER_SYS = """You are a precise financial table extractor.
Extract one target statement from the provided earnings/10-Q/10-K text.
Return pure JSON following the schema strictly; do not add commentary.
If a field is missing, fill it with null and include a 'notes' array.
"""

# Enhanced parser template with better financial document understanding
PARSER_USER_TEMPLATE = """Target Table: {table}  # one of BS/CF/IS
Constraints:
- Currency: detect and include as 'currency' (e.g., USD).
- Unit: detect scale (e.g., thousands/millions) as 'unit' and DO NOT rescale numbers.
- Period: infer period end date(s).

JSON schema (keys, loose types):
{schema_hint}

Text:
{context}
Return JSON only.
"""

# Enhanced prompts for better financial analysis
CONTENT_ANALYZER_SYS = """You are a financial document analyzer.
Analyze the provided PDF content and identify key financial sections,
metrics, and structure. Focus on identifying Balance Sheet, Income Statement,
and Cash Flow Statement sections and their key components."""

CONTENT_ANALYZER_TEMPLATE = """Analyze this financial document content:

{pdf_content}

Identify and extract:
1. Document type (10-K, 10-Q, earnings report, etc.)
2. Company name
3. Period/date information
4. Major financial statement sections present
5. Key financial metrics and their values

Return structured analysis as JSON with these sections:
{{
  "document_type": "string",
  "company": "string",
  "period": "string",
  "sections_found": ["list of sections"],
  "key_metrics": {{"metric_name": "value"}},
  "analysis_quality": "high/medium/low"
}}
"""

# 归一化/校验时的辅助提示（如果你在 LLM 里做 normalize/check）
NORMALIZER_SYS = """You are a data normalizer. Map keys to a canonical schema, keep numeric strings as strings if uncertain."""
CHECKER_SYS = """You are a rule checker. Verify row sums, sign conventions, and required keys. Return empty string if OK, else a short error message."""

# Schema hints for different financial statements
BALANCE_SHEET_SCHEMA = {
    "assets": {
        "current_assets": {"cash": "number", "receivables": "number", "inventory": "number"},
        "non_current_assets": {"ppe": "number", "goodwill": "number"},
        "total_assets": "number"
    },
    "liabilities": {
        "current_liabilities": {"payables": "number", "short_term_debt": "number"},
        "non_current_liabilities": {"long_term_debt": "number"},
        "total_liabilities": "number"
    },
    "equity": {
        "stockholders_equity": "number",
        "total_equity": "number"
    }
}

INCOME_STATEMENT_SCHEMA = {
    "revenue": "number",
    "cost_of_revenue": "number",
    "gross_profit": "number",
    "operating_expenses": "number",
    "operating_income": "number",
    "net_income": "number"
}

CASH_FLOW_SCHEMA = {
    "operating_activities": "number",
    "investing_activities": "number",
    "financing_activities": "number",
    "net_change_in_cash": "number"
}