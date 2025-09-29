#!/usr/bin/env python3
"""Test failure scenarios in the car analysis system"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Temporarily disable Tavily API to test failure handling
original_key = os.environ.get("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = "INVALID_KEY_FOR_TESTING"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import from the current directory (local imports)
from orchestrator import process_single_car, generate_final_report


async def test_failure_scenarios():
    """Test the car analysis system with failures"""

    print("🚗 Testing Failure Scenarios - Enhanced Error Handling")
    print("=" * 60)
    print("⚠️  TAVILY_API_KEY set to invalid value to simulate failures")
    print()

    # Sample car data
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
        }
    ]

    print(f"📋 Testing with {len(sample_cars)} sample cars (expecting failures)")
    print()

    # Process each car
    car_reports = []

    for i, car in enumerate(sample_cars):
        try:
            print(f"🚗 Analyzing Car {i+1}/{len(sample_cars)} (expecting failure)")
            car_report = await process_single_car(car)
            car_reports.append(car_report)

            # Check failure status
            analysis_status = car_report.get("analysis_status", {})
            if analysis_status.get("failed_permanently"):
                print(f"   ❌ Car {i+1} failed as expected")
                errors = analysis_status.get("errors", [])
                for error in errors:
                    print(f"      💥 {error}")
            else:
                print(f"   ⚠️  Car {i+1} unexpectedly succeeded")

        except Exception as e:
            print(f"   ❌ Car {i+1}: Analysis failed - {e}")
            import traceback
            traceback.print_exc()

    # Generate final report
    final_report = await generate_final_report(car_reports)

    # Save and display results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"failure_test_report_{timestamp}.json"

    with open(report_filename, "w") as f:
        json.dump(final_report, f, indent=2)

    # Display summary
    print(f"\n🎯 FAILURE TEST COMPLETE")
    print("=" * 60)
    print(f"📁 Report saved: {report_filename}")

    summary = final_report.get("summary", {})
    errors = summary.get("error_analysis", {})

    print(f"🚗 Cars analyzed: {summary.get('total_cars_analyzed', 0)}")
    print(f"✅ Successful: {summary.get('successful_analyses', 0)}")
    print(f"❌ Failed: {summary.get('failed_analyses', 0)}")
    print(f"📈 Success rate: {summary.get('success_rate', 0)}%")
    print()

    # Show error analysis
    if errors.get("total_errors", 0) > 0:
        print(f"❌ Error Analysis:")
        print(f"   💥 Total errors: {errors.get('total_errors', 0)}")
        print(f"   📊 Error types:")
        for error_type, count in errors.get("error_types", {}).items():
            print(f"      • {error_type}: {count} occurrence(s)")

        print(f"\n📋 Detailed errors:")
        for i, error in enumerate(errors.get("detailed_errors", []), 1):
            print(f"   {i}. {error}")

    print(f"\n✨ Enhanced error handling benefits:")
    print(f"   • No fallback to fake simulated data")
    print(f"   • Clear error reporting in final results")
    print(f"   • Proper retry limits prevent infinite loops")
    print(f"   • Transparent failure reasons for debugging")

    return final_report


if __name__ == "__main__":
    try:
        result = asyncio.run(test_failure_scenarios())
    finally:
        # Restore original API key
        if original_key:
            os.environ["TAVILY_API_KEY"] = original_key
        else:
            os.environ.pop("TAVILY_API_KEY", None)