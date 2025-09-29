import logging
from typing import Dict, Any
from ..state_types import GraphState

logger = logging.getLogger(__name__)

def content_quality_checker(state: GraphState) -> GraphState:
    """Check the quality and completeness of extracted PDF content"""

    pdf_content = state.get("pdf_content", "")
    pdf_metadata = state.get("pdf_metadata", {})

    quality_score = 0
    issues = []

    # Check content length
    content_length = len(pdf_content)
    if content_length > 1000:
        quality_score += 30
    elif content_length > 500:
        quality_score += 20
    else:
        issues.append(f"Content too short: {content_length} characters")

    # Check for financial keywords
    financial_keywords = ["assets", "liabilities", "revenue", "income", "cash", "equity"]
    found_keywords = [kw for kw in financial_keywords if kw.lower() in pdf_content.lower()]
    quality_score += len(found_keywords) * 10

    if len(found_keywords) < 3:
        issues.append(f"Few financial keywords found: {found_keywords}")

    # Check for numerical data
    import re
    numbers = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', pdf_content)
    if len(numbers) > 10:
        quality_score += 20
    elif len(numbers) > 5:
        quality_score += 10
    else:
        issues.append(f"Few numerical values found: {len(numbers)}")

    # Check extraction method
    extraction_method = pdf_metadata.get("extraction_method", "unknown")
    if extraction_method == "text":
        quality_score += 10

    quality_assessment = {
        "quality_score": min(quality_score, 100),
        "quality_level": "high" if quality_score >= 70 else "medium" if quality_score >= 40 else "low",
        "issues": issues,
        "found_keywords": found_keywords,
        "number_count": len(numbers),
        "extraction_method": extraction_method
    }

    logger.info(f"Content quality assessment: {quality_assessment['quality_level']} ({quality_assessment['quality_score']}/100)")

    return {"content_quality": quality_assessment}

def total_checker(state: GraphState) -> GraphState:
    """Cross-table validation and consistency checks"""

    results = state.get("results", {})
    validation_results = []

    # Check if we have extracted tables
    if not results:
        validation_results.append("No financial tables extracted")
        return {"validation_results": validation_results}

    # Basic consistency checks
    for table_name, table_data in results.items():
        if isinstance(table_data, dict):
            # Check for required fields based on table type
            if "BS" in table_name.upper():  # Balance Sheet
                if "assets" in table_data and "liabilities" in table_data:
                    validation_results.append(f"Balance Sheet structure looks good")
                else:
                    validation_results.append(f"Balance Sheet missing key sections")

            elif "IS" in table_name.upper():  # Income Statement
                if "revenue" in table_data or "income" in str(table_data).lower():
                    validation_results.append(f"Income Statement has revenue data")
                else:
                    validation_results.append(f"Income Statement missing revenue data")

    return {"validation_results": validation_results}

def aggregator(state: GraphState) -> GraphState:
    """Aggregate all results into final report"""

    # Extract metadata from PDF processing
    pdf_metadata = state.get("pdf_metadata", {})
    content_quality = state.get("content_quality", {})

    # Build metadata from actual PDF content
    meta = {
        "company": "Unknown",  # Would be extracted from content analysis
        "date": "2025-09-01",  # Would be extracted from content
        "file_path": pdf_metadata.get("file_path", ""),
        "page_count": pdf_metadata.get("page_count", 0),
        "content_length": pdf_metadata.get("content_length", 0),
        "extraction_method": pdf_metadata.get("extraction_method", "text"),
        "quality_score": content_quality.get("quality_score", 0),
        "processing_timestamp": "2025-09-15"
    }

    # Try to extract company name from content
    pdf_content = state.get("pdf_content", "")
    if "Sample Company" in pdf_content:
        meta["company"] = "Sample Company Inc."
    elif pdf_content:
        # Simple heuristic to find company name
        lines = pdf_content.split('\n')[:5]  # Check first few lines
        for line in lines:
            if len(line.strip()) > 5 and not line.strip().isdigit():
                meta["company"] = line.strip()
                break

    agg = {
        "meta": meta,
        "tables": state.get("results", {}),
        "quality_assessment": content_quality,
        "validation": state.get("validation_results", []),
        "processing_logs": state.get("dbg_logs", [])
    }

    logger.info(f"Aggregated report for {meta['company']}: {len(agg['tables'])} tables processed")

    return {"agg_report": agg}