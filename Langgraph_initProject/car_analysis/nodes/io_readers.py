from typing import Dict, Any, List
import logging
from ..state_types import GraphState
from ..tools.pdf_processor import pdf_processor

logger = logging.getLogger(__name__)

def read_pdf(state: GraphState) -> GraphState:
    """Read PDF document and extract content using real PDF processing"""

    # Get PDF file path from state
    pdf_path = state.get("pdf_path") or state.get("input_file")

    if not pdf_path:
        # Fallback for testing - use the sample PDF
        pdf_path = "/Users/wangyuyang/Desktop/Langgraph_initProject/car_analysis/mcp-pdf-reader/pdf_resources/sample_financial_report.pdf"
        logger.warning(f"No PDF path in state, using sample: {pdf_path}")

    logger.info(f"Processing PDF: {pdf_path}")

    # Process the PDF document
    result = pdf_processor.process_financial_document(pdf_path)

    if result["success"]:
        # Split content into logical pages/sections for downstream processing
        content = result["content"]

        # Simple section splitting based on common financial document patterns
        sections = []
        if "Balance Sheet" in content or "BALANCE SHEET" in content:
            sections.append("Balance Sheet section found")
        if "Income Statement" in content or "INCOME STATEMENT" in content:
            sections.append("Income Statement section found")
        if "Cash Flow" in content or "CASH FLOW" in content:
            sections.append("Cash Flow section found")
        if "Assets" in content:
            sections.append("Assets information found")
        if "Liabilities" in content:
            sections.append("Liabilities information found")

        # If no specific sections found, split by length
        if not sections:
            # Split content into chunks for processing
            content_chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
            sections = [f"Content chunk {i+1}" for i in range(len(content_chunks))]

        return {
            "pdf_pages": sections,
            "pdf_content": content,
            "pdf_metadata": {
                "file_path": pdf_path,
                "page_count": result.get("page_count", 0),
                "content_length": result.get("content_length", 0),
                "extraction_method": result.get("extraction_method", "text")
            },
            "dbg_logs": [f"ReadPDF completed: {len(content)} chars extracted from {result.get('page_count', 0)} pages"]
        }
    else:
        error_msg = result.get("error", "Unknown error")
        logger.error(f"PDF processing failed: {error_msg}")
        return {
            "pdf_pages": [f"Error: {error_msg}"],
            "pdf_content": "",
            "pdf_metadata": {"error": error_msg},
            "dbg_logs": [f"ReadPDF failed: {error_msg}"]
        }


def load_config() -> Dict[str, Any]:
    # 需要的话可以从 env / yaml 读配置
    return {"company_default": "DemoCorp"}
