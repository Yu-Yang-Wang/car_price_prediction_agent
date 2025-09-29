"""Direct PDF processing tools for financial agents"""
from typing import Optional, List, Dict, Any
import os
import sys
import base64
import logging

# Import the PDF functions from our local copy
from .pdf_functions import read_pdf_text, read_by_ocr, read_pdf_images

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Direct PDF processing without MCP wrapper"""

    def __init__(self):
        self.logger = logger

    def extract_text(self, file_path: str, start_page: int = 1, end_page: Optional[int] = None) -> Dict[str, Any]:
        """Extract text from PDF pages"""
        try:
            result = read_pdf_text(file_path, start_page, end_page)

            # Convert to a consistent format
            pages_text = []
            combined_text = ""

            for page_data in result.get('pages', []):
                page_text = page_data.get('text', '')
                pages_text.append(page_text)
                combined_text += page_text + "\n\n"

            return {
                "success": True,
                "page_count": result.get('page_count', 0),
                "pages_text": pages_text,
                "combined_text": combined_text.strip(),
                "extracted_pages": list(range(start_page, (end_page or result.get('page_count', 0)) + 1))
            }
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return {"success": False, "error": str(e)}

    def extract_text_ocr(self, file_path: str, start_page: int = 1, end_page: Optional[int] = None,
                        language: str = "eng", dpi: int = 300) -> Dict[str, Any]:
        """Extract text using OCR"""
        try:
            result = read_by_ocr(file_path, start_page, end_page, language, dpi)

            return {
                "success": True,
                "text": result.get('text', ''),
                "page_count": result.get('page_count', 0),
                "extracted_pages": result.get('extracted_pages', [])
            }
        except Exception as e:
            self.logger.error(f"Error extracting OCR text from {file_path}: {e}")
            return {"success": False, "error": str(e)}

    def extract_images(self, file_path: str, page_number: int = 1) -> Dict[str, Any]:
        """Extract images from PDF page"""
        try:
            result = read_pdf_images(file_path, page_number)

            return {
                "success": True,
                "images": result.get('images', []),
                "image_count": len(result.get('images', []))
            }
        except Exception as e:
            self.logger.error(f"Error extracting images from {file_path}: {e}")
            return {"success": False, "error": str(e)}

    def process_financial_document(self, file_path: str) -> Dict[str, Any]:
        """Process a financial document and extract all relevant information"""
        try:
            # Extract text from all pages
            text_result = self.extract_text(file_path)

            if not text_result["success"]:
                return text_result

            # Try OCR as backup if text extraction yields poor results
            combined_text = text_result["combined_text"]
            if len(combined_text.strip()) < 100:  # If very little text extracted
                self.logger.info("Text extraction yielded minimal content, trying OCR...")
                ocr_result = self.extract_text_ocr(file_path)
                if ocr_result["success"] and len(ocr_result["text"]) > len(combined_text):
                    combined_text = ocr_result["text"]

            return {
                "success": True,
                "file_path": file_path,
                "page_count": text_result["page_count"],
                "content": combined_text,
                "extraction_method": "text" if len(text_result["combined_text"]) > 100 else "ocr",
                "content_length": len(combined_text)
            }

        except Exception as e:
            self.logger.error(f"Error processing financial document {file_path}: {e}")
            return {"success": False, "error": str(e), "file_path": file_path}

# Create a global instance for easy import
pdf_processor = PDFProcessor()