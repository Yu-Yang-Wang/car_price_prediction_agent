#!/usr/bin/env python3
"""
🚗 Main Entry Point - Multi-Car Price Analysis with LangGraph and Tavily
"""

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
    print("⚠️ python-dotenv not installed. Make sure OPENAI_API_KEY and TAVILY_API_KEY are set.")

from core.orchestrator import analyze_car_deals


async def main():
    """Main entry point"""

    print("🚗 Multi-Car Price Analysis Agent")
    print("=" * 60)
    print("Powered by LangGraph + Tavily + OpenAI")
    print()

    # Check for PDF argument
    if len(sys.argv) < 2:
        print("📖 Usage:")
        print("  python main.py <car_deals.pdf>")
        print()
        print("📋 Example:")
        print("  python main.py sample_car_deals.pdf")
        print()
        print("🔧 Requirements:")
        print("  • Set OPENAI_API_KEY in .env file")
        print("  • Set TAVILY_API_KEY in .env file")
        print("  • Install: pip install langgraph tavily-python openai")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file '{pdf_path}' not found")
        sys.exit(1)

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Add it to your .env file or set as environment variable")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY not found in environment")
        print("   Add it to your .env file or set as environment variable")
        sys.exit(1)

    print(f"📄 Analyzing PDF: {pdf_path}")
    print()

    try:
        # Run the analysis
        result = await analyze_car_deals(pdf_path)

        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            sys.exit(1)

        # Success message
        summary = result.get("summary", {})

        print()
        print("🎉 Analysis completed successfully!")
        print(f"   📊 Cars analyzed: {summary.get('total_cars_analyzed', 0)}")
        print(f"   📈 Average score: {summary.get('average_deal_score', 0)}/100")
        print(f"   🔍 Real market data via Tavily search")

        # Scoring comparison
        scoring = summary.get("scoring_comparison", {})
        if scoring:
            print()
            print("📊 Scoring Comparison (Successful Analyses Only):")
            print(f"   🤖 Rule-based avg: {scoring.get('average_rule_score', 0)}/100")
            print(f"   🧠 LLM opinion avg: {scoring.get('average_llm_score', 0)}/100")

        # Categories
        print()
        print("📋 Rule-based Categories:")
        for verdict, count in summary.get("rule_based_categories", {}).items():
            print(f"   • {verdict}: {count} cars")

        if summary.get("llm_opinion_categories"):
            print("\n🧠 LLM Opinion Categories:")
            for verdict, count in summary.get("llm_opinion_categories", {}).items():
                print(f"   • {verdict}: {count} cars")

        # Deal breakdown alias (if present)
        if summary.get("deal_categories"):
            print("\n📊 Deal Breakdown:")
            for verdict, count in summary.get("deal_categories", {}).items():
                print(f"   • {verdict}: {count} car(s)")

        # Per-car results
        car_reports = result.get("car_reports", [])
        if car_reports:
            print("\n🚗 Individual Car Results:")
            for i, report in enumerate(car_reports, 1):
                car = report.get("car", {})
                score_data = report.get("deal_score", {})
                comparison = report.get("price_comparison", {})

                year = car.get("year", "?")
                make = car.get("make", "?")
                model = car.get("model", "?")

                if score_data.get("success"):
                    price_delta = comparison.get("price_delta", 0)
                    score = score_data.get("score", 0)
                    verdict = score_data.get("verdict", "Unknown")
                    data_source = score_data.get("data_source", "unknown")
                    price_paid = comparison.get("price_paid", car.get("price_paid", 0))

                    print(f"   {i}. {year} {make} {model}")
                    print(f"      💰 ${price_paid:,.0f} vs market (${price_delta:+,.0f})")
                    print(f"      🎯 {score}/100 - {verdict}")
                    print(f"      🔍 Data: {data_source}")
                else:
                    print(f"   {i}. {year} {make} {model} - Analysis failed")

    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
