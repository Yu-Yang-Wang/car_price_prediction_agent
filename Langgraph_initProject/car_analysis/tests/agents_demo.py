"""Run each agent step-by-step on a few cars and print results.

Usage:
  source .langgraphvenv/bin/activate
  python -m car_analysis.tests.agents_demo --limit 2

This script does not require any API keys. If OPENAI/TAVILY/CARSXE keys are
missing, the related agents will return structured errors which you can review
to adjust configuration later.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from car_analysis.database.manager import DatabaseManager
from car_analysis.core.models import CarAnalysisState
from car_analysis.core.agents import (
    condition_agent,
    early_rag_agent,
    market_price_agent,
    residual_value_agent,
    news_policy_agent,
    carsxe_agent,
    consistency_agent,
    rag_vector_agent,
    summary_agent,
)


def _pick_sample_cars(limit: int) -> List[Dict[str, Any]]:
    db = DatabaseManager()
    cars = db.search_cars(limit=limit)
    return cars


async def run_for_car(car: Dict[str, Any], save_markdown_dir: Path | None = None) -> Dict[str, Any]:
    state: CarAnalysisState = CarAnalysisState({
        "current_car": car,
        "retries": {},
        "dbg_logs": [],
        "agent_logs": [],
    })

    # Run agents (some require network; failures are captured in their outputs)
    for fn, label in [
        (condition_agent, "condition"),
        (early_rag_agent, "early_rag"),
        (market_price_agent, "market"),
        (residual_value_agent, "residual"),
        (news_policy_agent, "news"),
        (carsxe_agent, "carsxe"),
        (consistency_agent, "consistency"),
        (rag_vector_agent, "rag"),
        (summary_agent, "summary"),
    ]:
        try:
            update = await fn(state)
            state.update(update)
            print(f"✅ Agent {label} finished")
        except Exception as e:
            print(f"❌ Agent {label} failed: {e}")

    # Pretty print highlights
    print("\n=== Agent outputs (keys present) ===")
    for key in [
        "condition_report",
        "early_rag",
        "market_analysis",
        "residual_analysis",
        "news_analysis",
        "rag_insights",
        "consistency_report",
        "summary_report",
        "markdown_refined",
    ]:
        val = state.get(key)
        if val is not None:
            print(f"- {key}: present (success={val.get('success', 'n/a') if isinstance(val, dict) else 'n/a'})")

    summary = state.get("summary_report", {}) or {}
    if summary.get("analysis_text"):
        print("\n--- Summary Analysis Text ---")
        print(summary.get("analysis_text"))
    if state.get("markdown_refined"):
        print("\n--- Markdown Refined ---")
        print(state["markdown_refined"])

    # Save markdown if available
    md = state.get("markdown_refined") or (state.get("summary_report", {}) or {}).get("analysis_text")
    if save_markdown_dir and md:
        save_markdown_dir.mkdir(parents=True, exist_ok=True)
        name = f"agent_report_{car.get('year')}_{car.get('make')}_{car.get('model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path = save_markdown_dir / name
        path.write_text(str(md))
        print(f"📄 Saved markdown to {path}")

    return state


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run agent demo on a few cars")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--save-md", action="store_true", help="Save markdown to outputs/")
    args = parser.parse_args()

    cars = _pick_sample_cars(limit=args.limit)
    if not cars:
        print("No cars found in DB. Please ingest CSV first using ingest_csv.py.")
        return

    output_dir = Path("outputs") if args.save_md else None

    for i, car in enumerate(cars, 1):
        print("\n" + "=" * 80)
        print(f"Car {i}/{len(cars)}: {car.get('year')} {car.get('make')} {car.get('model')} - ${car.get('price_paid', 0):,.0f}")
        print("=" * 80)
        await run_for_car(car, save_markdown_dir=output_dir)


if __name__ == "__main__":
    asyncio.run(main())
