"""Orchestration functions for car analysis workflow using LangGraph"""

import json
from typing import Dict, List, Any
from datetime import datetime
from .models import CarAnalysisState


async def aggregate_car_reports(state: CarAnalysisState) -> CarAnalysisState:
    """Aggregate individual car analysis results"""
    print("📊 Generating final report...")

    # Get current car analysis results
    current_car = state.get("current_car", {})
    rule_score = state.get("deal_score", {})
    llm_opinion = state.get("llm_opinion", {})
    comparison = state.get("price_comparison", {})
    research = state.get("price_research", {})
    condition_report = state.get("condition_report", {})
    market_analysis = state.get("market_analysis", {})
    residual_analysis = state.get("residual_analysis", {})
    news_analysis = state.get("news_analysis", {})
    rag_insights = state.get("rag_insights", {})
    summary_section = state.get("summary_report", {})

    # Check for analysis failures
    analysis_errors = state.get("analysis_errors", [])
    failed_permanently = state.get("failed_permanently", False)

    # Create car report with both scoring methods and error information
    def _format_currency(value: Any) -> str:
        if value is None:
            return "N/A"
        try:
            return f"${float(value):,.0f}"
        except Exception:
            return str(value)

    markdown_sections = [
        f"# {current_car.get('year', 'Unknown')} {current_car.get('make', '')} {current_car.get('model', '')}",
        "",
        "## Vehicle Condition",
        f"- Clean title: {condition_report.get('condition_flags', {}).get('clean_title', 'Unknown')}",
        f"- Accident history: {condition_report.get('condition_flags', {}).get('accident_history', 'Unknown')}",
        "",
        "## Market Pricing",
        f"- Market median: {_format_currency(market_analysis.get('market_median'))}",
        f"- Price delta: {_format_currency(market_analysis.get('price_delta'))} ({market_analysis.get('price_delta_pct', 0):+.1f}%)",
        f"- Deal category: {market_analysis.get('deal_category', 'Unknown')}",
        "",
        "## Scoring",
        f"- Rule-based score: {market_analysis.get('rule_score', 'N/A')} ({market_analysis.get('rule_verdict', 'Unknown')})",
        f"- LLM score: {market_analysis.get('llm_score', 'N/A')} ({market_analysis.get('llm_verdict', 'Unknown')})",
        "",
        "## Residual Value",
        f"- Predicted resale: {_format_currency(residual_analysis.get('predicted_price'))}",
        "",
        "## External Insights",
        f"- CarsXE: {rag_insights.get('carsxe', {}).get('success', False)}",
        f"- Vector cases: {len((rag_insights.get('vector', {}) or {}).get('similar_cases', []) or [])}",
        "",
        "## Notes",
        summary_section.get("analysis_text", "See structured data."),
    ]

    car_report = {
        "car": current_car,
        "condition_report": condition_report,
        "market_analysis": market_analysis or research,
        "price_comparison": comparison,
        "deal_score": rule_score,  # Rule-based scoring
        "llm_opinion": llm_opinion,  # LLM-based opinion
        "residual_analysis": residual_analysis,
        "news_analysis": news_analysis,
        "rag_insights": rag_insights,
        "summary": summary_section,
        "markdown_report": "\n".join(markdown_sections),
        "analysis_timestamp": datetime.now().isoformat(),
        "analysis_status": {
            "success": not failed_permanently,
            "failed_permanently": failed_permanently,
            "errors": analysis_errors,
            "error_count": len(analysis_errors)
        }
    }

    # Get existing car reports and add current one
    car_reports = state.get("car_reports", [])
    car_reports.append(car_report)

    # Calculate summary statistics (using rule-based scores as primary)
    total_cars = len(car_reports)
    successful_analyses = sum(1 for r in car_reports if r.get("analysis_status", {}).get("success", False))
    failed_analyses = total_cars - successful_analyses

    # Collect all errors for reporting
    all_errors = []
    for report in car_reports:
        errors = report.get("analysis_status", {}).get("errors", [])
        all_errors.extend(errors)

    # Calculate scores only for successful analyses
    rule_scores = [r.get("deal_score", {}).get("score", 0) for r in car_reports
                   if r.get("analysis_status", {}).get("success", False) and r.get("deal_score", {}).get("success")]
    llm_scores = [r.get("llm_opinion", {}).get("score", 0) for r in car_reports
                  if r.get("analysis_status", {}).get("success", False) and r.get("llm_opinion", {}).get("score")]

    avg_rule_score = sum(rule_scores) / len(rule_scores) if rule_scores else 0
    avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0

    # Categorize deals (using rule-based verdicts) - only successful analyses
    categories = {}
    for report in car_reports:
        if report.get("analysis_status", {}).get("success", False):
            verdict = report.get("deal_score", {}).get("verdict", "Unknown")
            categories[verdict] = categories.get(verdict, 0) + 1

    # Also track LLM agreement - only successful analyses
    llm_categories = {}
    for report in car_reports:
        if report.get("analysis_status", {}).get("success", False):
            llm_verdict = report.get("llm_opinion", {}).get("verdict", "Unknown")
            llm_categories[llm_verdict] = llm_categories.get(llm_verdict, 0) + 1

    # Count error types
    error_summary = {}
    for error in all_errors:
        error_type = error.split(":")[0] if ":" in error else "UNKNOWN_ERROR"
        error_summary[error_type] = error_summary.get(error_type, 0) + 1

    agg_report = {
        "summary": {
            "total_cars_analyzed": total_cars,
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "success_rate": round((successful_analyses / total_cars * 100) if total_cars > 0 else 0, 1),
            "scoring_comparison": {
                "average_rule_score": round(avg_rule_score, 1),
                "average_llm_score": round(avg_llm_score, 1)
            },
            "rule_based_categories": categories,
            "llm_opinion_categories": llm_categories,
            "error_analysis": {
                "total_errors": len(all_errors),
                "error_types": error_summary,
                "detailed_errors": all_errors
            }
        },
        "car_reports": car_reports,
        "generated_at": datetime.now().isoformat()
    }

    return {
        "agg_report": agg_report,
        "car_reports": car_reports,
        "dbg_logs": [f"Generated final report for {total_cars} cars with dual scoring"]
    }


async def process_single_car_langgraph(car_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single car through the LangGraph workflow"""

    print(f"\n🚗 Analyzing: {car_data['year']} {car_data['make']} {car_data['model']}")
    print("=" * 50)

    # Build the workflow graph - import here to avoid circular dependency
    from .graph import build_single_car_graph
    workflow = build_single_car_graph()

    # Initialize state for this car
    initial_state = CarAnalysisState({
        "current_car": car_data,
        "retries": {},
        "dbg_logs": [],
        "car_reports": []  # Initialize empty list for this car
    })

    # Execute the workflow
    try:
        # Run the LangGraph workflow
        final_state = await workflow.ainvoke(initial_state)

        # Get the car reports from final state
        car_reports = final_state.get("car_reports", [])
        if car_reports:
            # Return the last (current) car report
            car_report = car_reports[-1]

            # Show scoring comparison if both exist
            rule_score = car_report.get("deal_score", {}).get("score", 0)
            llm_score = car_report.get("llm_opinion", {}).get("score", 0)
            rule_verdict = car_report.get("deal_score", {}).get("verdict", "Unknown")
            llm_verdict = car_report.get("llm_opinion", {}).get("verdict", "Unknown")

            print(f"   🤖 Rule-based: {rule_score}/100 - {rule_verdict}")
            print(f"   🧠 LLM Opinion: {llm_score}/100 - {llm_verdict}")

            # Check for disagreement
            if abs(rule_score - llm_score) > 20:
                print(f"   ⚠️  Significant disagreement detected!")

            return car_report
        else:
            # Fallback - create basic report structure
            return {
                "car": car_data,
                "error": "No car report generated",
                "analysis_timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        print(f"   ❌ LangGraph workflow failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            "car": car_data,
            "error": str(e),
            "analysis_timestamp": datetime.now().isoformat()
        }


async def process_single_car(car_data: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function - redirects to LangGraph version"""
    return await process_single_car_langgraph(car_data)


async def generate_final_report(car_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary report for all cars with dual scoring analysis"""

    total_cars = len(car_reports)
    successful_analyses = sum(1 for r in car_reports if r.get("deal_score", {}).get("success"))
    failed_analyses = total_cars - successful_analyses

    # Get both rule-based and LLM scores
    rule_scores = [r.get("deal_score", {}).get("score", 0) for r in car_reports
                   if r.get("deal_score", {}).get("success")]
    llm_scores = [r.get("llm_opinion", {}).get("score", 0) for r in car_reports
                  if r.get("llm_opinion", {}).get("score")]

    avg_rule_score = sum(rule_scores) / len(rule_scores) if rule_scores else 0
    avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0
    # Backward-compat: single-number average for display (based on rule score)
    avg_deal_score = avg_rule_score

    # Categorize deals (rule-based)
    rule_categories = {}
    for report in car_reports:
        verdict = report.get("deal_score", {}).get("verdict", "Unknown")
        rule_categories[verdict] = rule_categories.get(verdict, 0) + 1

    # Categorize LLM opinions
    llm_categories = {}
    for report in car_reports:
        llm_verdict = report.get("llm_opinion", {}).get("verdict", "Unknown")
        llm_categories[llm_verdict] = llm_categories.get(llm_verdict, 0) + 1

    # Calculate agreement statistics
    agreements = 0
    disagreements = 0
    for report in car_reports:
        rule_score = report.get("deal_score", {}).get("score", 0)
        llm_score = report.get("llm_opinion", {}).get("score", 0)
        if rule_score and llm_score:
            if abs(rule_score - llm_score) <= 20:  # Within 20 points
                agreements += 1
            else:
                disagreements += 1

    agreement_rate = (agreements / (agreements + disagreements) * 100) if (agreements + disagreements) > 0 else 0

    # Build error analysis from failed entries
    all_errors: List[str] = []
    for r in car_reports:
        status = r.get("analysis_status", {})
        # Errors from status
        if status and not status.get("success", True):
            all_errors.extend(status.get("errors", []))
        # Direct error field from exceptions
        if r.get("error"):
            all_errors.append(str(r.get("error")))

    error_types: Dict[str, int] = {}
    for e in all_errors:
        et = e.split(":")[0] if isinstance(e, str) and ":" in e else (e or "UNKNOWN_ERROR")
        error_types[et] = error_types.get(et, 0) + 1

    final_report = {
        "summary": {
            "total_cars_analyzed": total_cars,
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "success_rate": round((successful_analyses / total_cars * 100) if total_cars > 0 else 0, 1),
            # Backward-compat keys expected by test scripts
            "average_deal_score": round(avg_deal_score, 1),
            "deal_categories": rule_categories,
            "scoring_comparison": {
                "average_rule_score": round(avg_rule_score, 1),
                "average_llm_score": round(avg_llm_score, 1),
                "agreement_rate": round(agreement_rate, 1),
                "agreements": agreements,
                "disagreements": disagreements
            },
            "rule_based_categories": rule_categories,
            "llm_opinion_categories": llm_categories,
            "error_analysis": {
                "total_errors": len(all_errors),
                "error_types": error_types,
                "detailed_errors": all_errors,
            },
        },
        "car_reports": car_reports,
        "generated_at": datetime.now().isoformat()
    }

    return final_report


async def analyze_car_deals(pdf_path: str) -> Dict[str, Any]:
    """Main function to analyze car deals from PDF using LangGraph workflow"""

    print("🚗 Multi-Car Price Analysis Agent (LangGraph + Dual Scoring)")
    print("=" * 60)

    # Step 1: Extract cars from PDF
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.pdf_extractor import extract_cars_from_pdf

    state = CarAnalysisState({"pdf_path": pdf_path})
    extraction_result = await extract_cars_from_pdf(state)

    cars = extraction_result.get("cars", [])

    if not cars:
        print("❌ No cars found in PDF")
        return {"error": "No cars found in PDF"}

    print(f"\n📋 Found {len(cars)} cars to analyze")
    print("🔧 Using enhanced LangGraph workflow with:")
    print("   • Tavily real market data search")
    print("   • Rule-based algorithmic scoring")
    print("   • GPT-4o LLM opinion analysis")
    print("   • Automatic disagreement detection & retry")
    print()

    # Step 2: Process each car using LangGraph workflow
    car_reports = []

    for i, car in enumerate(cars):
        try:
            print(f"🔄 Processing Car {i+1}/{len(cars)} through LangGraph workflow...")
            car_report = await process_single_car_langgraph(car)
            car_reports.append(car_report)

            # Check if analysis was successful
            analysis_status = car_report.get("analysis_status", {})
            is_successful = analysis_status.get("success", False)

            if is_successful:
                # Show progress with both scores for successful analyses
                rule_score = car_report.get("deal_score", {}).get("score", 0)
                llm_score = car_report.get("llm_opinion", {}).get("score", 0)
                rule_verdict = car_report.get("deal_score", {}).get("verdict", "Unknown")
                llm_verdict = car_report.get("llm_opinion", {}).get("verdict", "Unknown")

                print(f"   ✅ Car {i+1} Complete:")
                print(f"      🤖 Rule: {rule_score}/100 - {rule_verdict}")
                print(f"      🧠 LLM: {llm_score}/100 - {llm_verdict}")

                # Check agreement
                if abs(rule_score - llm_score) <= 10:
                    print(f"      ✅ Scores agree (±10 points)")
                elif abs(rule_score - llm_score) <= 20:
                    print(f"      ⚠️  Minor disagreement ({abs(rule_score - llm_score)} points)")
                else:
                    print(f"      ❌ Major disagreement ({abs(rule_score - llm_score)} points)")
            else:
                # Show failure information
                errors = analysis_status.get("errors", [])
                print(f"   ❌ Car {i+1} FAILED:")
                for error in errors:
                    print(f"      💥 {error}")
                print(f"      📋 Analysis marked as permanently failed")

        except Exception as e:
            print(f"   ❌ Car {i+1}: LangGraph analysis failed - {e}")
            car_reports.append({
                "car": car,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            })

    # Step 3: Generate final report with dual scoring analysis
    final_report = await generate_final_report(car_reports)

    # Save report to outputs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"outputs/langgraph_car_analysis_{timestamp}.json"

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    with open(report_filename, "w") as f:
        json.dump(final_report, f, indent=2)

    # Display enhanced summary
    print(f"\n📊 LANGGRAPH ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Report saved: {report_filename}")

    summary = final_report.get("summary", {})
    scoring = summary.get("scoring_comparison", {})
    errors = summary.get("error_analysis", {})

    print(f"🚗 Cars analyzed: {summary.get('total_cars_analyzed', 0)}")
    print(f"✅ Successful: {summary.get('successful_analyses', 0)}")
    print(f"❌ Failed: {summary.get('failed_analyses', 0)}")
    print(f"📈 Success rate: {summary.get('success_rate', 0)}%")
    print()

    # Only show scoring if there were successful analyses
    if summary.get("successful_analyses", 0) > 0:
        print(f"📊 Scoring Comparison (Successful Analyses Only):")
        print(f"   🤖 Rule-based avg: {scoring.get('average_rule_score', 0)}/100")
        print(f"   🧠 LLM opinion avg: {scoring.get('average_llm_score', 0)}/100")
        print()

        print(f"📋 Rule-based Categories:")
        for verdict, count in summary.get("rule_based_categories", {}).items():
            print(f"   • {verdict}: {count} cars")

        print(f"\n🧠 LLM Opinion Categories:")
        for verdict, count in summary.get("llm_opinion_categories", {}).items():
            print(f"   • {verdict}: {count} cars")
        print()

    # Show error analysis if there were failures
    if errors.get("total_errors", 0) > 0:
        print(f"❌ Error Analysis:")
        print(f"   💥 Total errors: {errors.get('total_errors', 0)}")
        print(f"   📊 Error types:")
        for error_type, count in errors.get("error_types", {}).items():
            print(f"      • {error_type}: {count} occurrence(s)")
        print()

    return final_report
