# financial_agents/nodes/tools.py
from __future__ import annotations
from typing import Optional, Dict, Any
from tools.pdf_processor import pdf_processor
import logging

logger = logging.getLogger(__name__)

async def fetch_pdf_text_for_table(file_path: str, page_range: Optional[Dict[str, int]] = None) -> str:
    """Fetch PDF text using direct PDF processing instead of MCP client"""
    start = 1
    end = None
    if page_range:
        start = int(page_range.get("start") or 1)
        end = int(page_range["end"]) if page_range.get("end") else None

    logger.info(f"Fetching PDF text for {file_path}, pages {start}-{end}")

    # Use our direct PDF processor
    result = pdf_processor.extract_text(file_path, start, end)

    if result.get("success"):
        combined_text = result.get("combined_text", "")
        logger.info(f"Successfully extracted {len(combined_text)} characters")
        return combined_text
    else:
        # Try OCR as fallback
        logger.warning("Text extraction failed, trying OCR...")
        ocr_result = pdf_processor.extract_text_ocr(file_path, start, end)
        if ocr_result.get("success"):
            ocr_text = ocr_result.get("text", "")
            logger.info(f"OCR extracted {len(ocr_text)} characters")
            return ocr_text
        else:
            logger.error(f"Both text and OCR extraction failed for {file_path}")
            return ""

def get_llm():
    import os
    from langchain_openai import ChatOpenAI

    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, use environment variables directly

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)
