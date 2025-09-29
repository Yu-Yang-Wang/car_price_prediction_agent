#!/usr/bin/env python3
"""Quick test for the restructured car analysis system"""

import asyncio
import sys
import os
from datetime import datetime

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Direct imports to avoid circular dependencies
from core.models import CarAnalysisState, Car
from core.workers import price_research_worker, price_comparison_worker, deal_scoring_worker, llm_opinion_worker


async def test_core_functionality():
    """Test core functionality with sample data"""
    print("🚗 Testing Restructured Car Analysis System")
    print("=" * 60)

    # Sample car data
    test_car = Car(
        make="Toyota",
        model="Camry",
        year=2020,
        mileage=35000,
        price_paid=22500,
        index=1
    )

    # Test state - convert Car object to dict for workers
    import dataclasses
    state = CarAnalysisState({
        "current_car": dataclasses.asdict(test_car),
        "research_retries": 0,
        "llm_retries": 0,
        "disagreement_retries": 0
    })

    print("\n🔬 Testing individual workers...")

    # Test 1: Price Research
    print("\n1. Testing price research worker...")
    try:
        research_result = await price_research_worker(state)
        if research_result.get("price_research", {}).get("success"):
            print("   ✅ Price research: SUCCESS")
            prices_found = len(research_result.get("price_research", {}).get("comparable_prices", []))
            print(f"   📊 Found {prices_found} comparable prices")
        else:
            print("   ❌ Price research: FAILED")
        state.update(research_result)
    except Exception as e:
        print(f"   ❌ Price research error: {e}")

    # Test 2: Price Comparison
    print("\n2. Testing price comparison worker...")
    try:
        comparison_result = await price_comparison_worker(state)
        if comparison_result.get("price_comparison", {}).get("success"):
            print("   ✅ Price comparison: SUCCESS")
            delta = comparison_result.get("price_comparison", {}).get("price_delta", 0)
            print(f"   💰 Price delta: ${delta:+,.0f}")
        else:
            print("   ❌ Price comparison: FAILED")
        state.update(comparison_result)
    except Exception as e:
        print(f"   ❌ Price comparison error: {e}")

    # Test 3: Deal Scoring
    print("\n3. Testing deal scoring worker...")
    try:
        scoring_result = await deal_scoring_worker(state)
        if scoring_result.get("deal_score", {}).get("success"):
            print("   ✅ Deal scoring: SUCCESS")
            score = scoring_result.get("deal_score", {}).get("score", 0)
            verdict = scoring_result.get("deal_score", {}).get("verdict", "Unknown")
            print(f"   🎯 Score: {score}/100 - {verdict}")
        else:
            print("   ❌ Deal scoring: FAILED")
        state.update(scoring_result)
    except Exception as e:
        print(f"   ❌ Deal scoring error: {e}")

    # Test 4: LLM Opinion
    print("\n4. Testing LLM opinion worker...")
    try:
        llm_result = await llm_opinion_worker(state)
        if llm_result.get("llm_opinion", {}).get("success"):
            print("   ✅ LLM opinion: SUCCESS")
            llm_score = llm_result.get("llm_opinion", {}).get("score", 0)
            llm_verdict = llm_result.get("llm_opinion", {}).get("verdict", "Unknown")
            print(f"   🧠 LLM Score: {llm_score}/100 - {llm_verdict}")
        else:
            print("   ❌ LLM opinion: FAILED")
        state.update(llm_result)
    except Exception as e:
        print(f"   ❌ LLM opinion error: {e}")

    # Save results to outputs
    print("\n💾 Saving test results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"outputs/quick_test_{timestamp}.json"

    os.makedirs("outputs", exist_ok=True)

    import json
    import dataclasses
    test_report = {
        "test_timestamp": timestamp,
        "test_car": dataclasses.asdict(test_car),
        "final_state": dict(state)
    }

    with open(report_filename, "w") as f:
        json.dump(test_report, f, indent=2)

    print(f"📁 Report saved: {report_filename}")
    print("\n🎉 Core functionality test completed!")


if __name__ == "__main__":
    asyncio.run(test_core_functionality())