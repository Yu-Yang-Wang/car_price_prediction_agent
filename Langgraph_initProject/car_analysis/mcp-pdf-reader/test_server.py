#!/usr/bin/env python3
"""
Simple test script to test the PDF server functionality.
This provides an easy way to test the tools without the MCP Inspector.
"""

import os
import sys
import json
from pdf_server import read_pdf_text, read_by_ocr, read_pdf_images

def main():
    print("PDF Server Test Interface")
    print("=" * 40)

    # Check if pdf_resources directory exists and has files
    pdf_dir = "pdf_resources"
    if not os.path.exists(pdf_dir):
        print(f"Creating {pdf_dir} directory...")
        os.makedirs(pdf_dir)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}/")
        print("Please add some PDF files to test with.")
        return

    print(f"Found PDF files: {pdf_files}")

    while True:
        print("\nAvailable commands:")
        print("1. read_pdf_text - Extract text from PDF")
        print("2. read_by_ocr - Extract text using OCR")
        print("3. read_pdf_images - Extract images from PDF")
        print("4. quit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "4":
            break

        if choice not in ["1", "2", "3"]:
            print("Invalid choice!")
            continue

        # Get file
        print(f"\nAvailable PDFs: {pdf_files}")
        file_name = input("Enter PDF filename: ").strip()

        if file_name not in pdf_files:
            print("File not found!")
            continue

        file_path = os.path.join(pdf_dir, file_name)

        try:
            if choice == "1":
                # read_pdf_text
                start_page = input("Start page (default 1): ").strip() or "1"
                end_page = input("End page (default all): ").strip() or None

                result = read_pdf_text(
                    file_path=file_path,
                    start_page=int(start_page),
                    end_page=int(end_page) if end_page else None
                )

                print(f"\nResult: {json.dumps(result, indent=2, ensure_ascii=False)}")

            elif choice == "2":
                # read_by_ocr
                start_page = input("Start page (default 1): ").strip() or "1"
                end_page = input("End page (default all): ").strip() or None
                language = input("Language (default eng): ").strip() or "eng"
                dpi = input("DPI (default 300): ").strip() or "300"

                result = read_by_ocr(
                    file_path=file_path,
                    start_page=int(start_page),
                    end_page=int(end_page) if end_page else None,
                    language=language,
                    dpi=int(dpi)
                )

                print(f"\nResult: {json.dumps(result, indent=2, ensure_ascii=False)}")

            elif choice == "3":
                # read_pdf_images
                page_number = input("Page number (default 1): ").strip() or "1"

                result = read_pdf_images(
                    file_path=file_path,
                    page_number=int(page_number)
                )

                # For images, don't print the base64 data (too long)
                result_summary = result.copy()
                if 'images' in result_summary:
                    for img in result_summary['images']:
                        if 'image_b64' in img:
                            img['image_b64'] = f"<base64 data - {len(img['image_b64'])} chars>"

                print(f"\nResult: {json.dumps(result_summary, indent=2, ensure_ascii=False)}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()