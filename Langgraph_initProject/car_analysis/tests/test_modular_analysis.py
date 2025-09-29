#!/usr/bin/env python3
"""Test the modular car analysis system with sample data"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import from the current directory (local imports)
from core.orchestrator import process_single_car, generate_final_report


async def test_modular_car_analysis():
    """Test the modular car analysis system"""

    print("🚗 Testing Modular Multi-Car Price Analysis Agent")
    print("=" * 60)

    # Sample car data (same as before)
    sample_cars = [
        {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "mileage": 35000,
            "price_paid": 22500.0,
            "index": 0
        },
        {
            "make": "Honda",
            "model": "Civic",
            "year": 2019,
            "mileage": 42000,
            "price_paid": 18200.0,
            "index": 1
        },
        {
            "make": "Ford",
            "model": "F-150",
            "year": 2021,
            "mileage": 28000,
            "price_paid": 31800.0,
            "index": 2
        }
    ]

    print(f"📋 Testing with {len(sample_cars)} sample cars")
    print("🏗️ Using modular architecture:")
    print("   • car_analysis.workers: Price research, comparison, scoring")
    print("   • car_analysis.checkers: Retry logic and validation")
    print("   • car_analysis.models: Data structures and state")
    print("   • car_analysis.orchestrator: Workflow coordination")
    print()

    # Process each car
    car_reports = []

    for i, car in enumerate(sample_cars):
        try:
            print(f"🚗 Analyzing Car {i+1}/{len(sample_cars)}")
            car_report = await process_single_car(car)
            car_reports.append(car_report)

            # Show progress
            score = car_report.get("deal_score", {}).get("score", 0)
            verdict = car_report.get("deal_score", {}).get("verdict", "Unknown")
            price_paid = car['price_paid']
            market_price = car_report.get("market_analysis", {}).get("median_price", 0)

            print(f"   💰 Paid: ${price_paid:,.0f} | Market: ${market_price:,.0f}")
            print(f"   🎯 Score: {score}/100 | {verdict}")

        except Exception as e:
            print(f"   ❌ Car {i+1}: Analysis failed - {e}")
            import traceback
            traceback.print_exc()

    # Generate final report
    final_report = await generate_final_report(car_reports)

    # Save and display results to outputs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"outputs/modular_car_analysis_test_{timestamp}.json"

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    with open(report_filename, "w") as f:
        json.dump(final_report, f, indent=2)

    # Display summary
    print(f"\n🎯 MODULAR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Report saved: {report_filename}")

    summary = final_report.get("summary", {})
    print(f"🚗 Cars analyzed: {summary.get('total_cars_analyzed', 0)}")
    print(f"✅ Successful: {summary.get('successful_analyses', 0)}")
    print(f"📈 Average score: {summary.get('average_deal_score', 0)}/100")

    print(f"\n📊 Deal Breakdown:")
    for verdict, count in summary.get("deal_categories", {}).items():
        print(f"   • {verdict}: {count} car(s)")

    # Show individual results
    print(f"\n🚗 Individual Car Results:")
    for i, report in enumerate(car_reports):
        car = report.get("car", {})
        score_data = report.get("deal_score", {})
        comparison = report.get("price_comparison", {})

        if score_data.get("success"):
            price_delta = comparison.get("price_delta", 0)
            score = score_data.get("score", 0)
            verdict = score_data.get("verdict", "Unknown")
            data_source = score_data.get("data_source", "unknown")

            print(f"   {i+1}. {car['year']} {car['make']} {car['model']}")
            print(f"      💰 ${car['price_paid']:,.0f} vs market (${price_delta:+,.0f})")
            print(f"      🎯 {score}/100 - {verdict}")
            print(f"      🔍 Data: {data_source}")
        else:
            print(f"   {i+1}. {car['year']} {car['make']} {car['model']} - Analysis failed")

    print(f"\n✨ Modular structure benefits:")
    print(f"   • Clean separation of concerns")
    print(f"   • Easy to test individual components")
    print(f"   • Maintainable and extensible code")
    print(f"   • Proper LangGraph state management")

    return final_report


if __name__ == "__main__":
    result = asyncio.run(test_modular_car_analysis())