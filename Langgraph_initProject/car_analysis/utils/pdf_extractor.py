"""PDF extraction for car data"""

import json
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.pdf_processor import pdf_processor
from nodes.tools import get_llm
from core.models import CarAnalysisState


async def extract_cars_from_pdf(state: CarAnalysisState) -> CarAnalysisState:
    """Extract car information from PDF"""
    print("📄 Extracting car data from PDF...")

    pdf_path = state.get("pdf_path", "")
    if not pdf_path:
        return {"cars": [], "dbg_logs": ["No PDF path provided"]}

    # Extract PDF content
    result = pdf_processor.process_financial_document(pdf_path)

    if not result["success"]:
        return {"cars": [], "dbg_logs": [f"PDF extraction failed: {result.get('error')}"]}

    content = result["content"]
    print(f"✅ Extracted {len(content)} characters from PDF")

    # Parse car data using LLM
    llm = get_llm()

    car_extraction_prompt = f"""
Extract car information from this text. Find all cars mentioned with their details.

Text:
{content}

Extract each car as JSON with these fields:
- make: string (e.g. "Toyota", "Honda")
- model: string (e.g. "Camry", "Civic")
- year: integer (e.g. 2020)
- mileage: integer (miles, e.g. 65000)
- price_paid: float (dollars, e.g. 21000.0)

Return as JSON array:
[
  {{"make": "Toyota", "model": "Camry", "year": 2020, "mileage": 65000, "price_paid": 21000.0}},
  {{"make": "Honda", "model": "Civic", "year": 2019, "mileage": 45000, "price_paid": 18500.0}}
]
"""

    try:
        response = await llm.ainvoke([
            ("system", "You are a car data extraction expert. Extract car information accurately from text."),
            ("user", car_extraction_prompt)
        ])

        # Parse LLM response
        car_data_text = response.content.strip()
        if car_data_text.startswith("```json"):
            car_data_text = car_data_text.replace("```json", "").replace("```", "").strip()

        cars = json.loads(car_data_text)

        # Add index to each car
        for i, car in enumerate(cars):
            car["index"] = i
            car["raw_text"] = content[:500]  # First 500 chars for context

        print(f"✅ Extracted {len(cars)} cars from PDF")
        for i, car in enumerate(cars):
            print(f"   Car {i+1}: {car['year']} {car['make']} {car['model']} - ${car['price_paid']:,.0f}")

        return {
            "cars": cars,
            "pdf_content": content,
            "dbg_logs": [f"Extracted {len(cars)} cars from PDF"]
        }

    except Exception as e:
        print(f"❌ Error parsing car data: {e}")
        return {
            "cars": [],
            "dbg_logs": [f"Car extraction failed: {e}"]
        }