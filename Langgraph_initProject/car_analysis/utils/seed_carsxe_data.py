"""Seed the database/vector store using CarsXE market valuations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from car_analysis.database.manager import DatabaseManager
from car_analysis.rag.embeddings import EmbeddingManager
from car_analysis.rag.vector_store import VectorStoreManager
from car_analysis.tools.carsxe_api import CarsXEClient


logger = logging.getLogger("carsxe_seed")


def _parse_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = value.replace("$", "").replace(",", "").replace(" mi.", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else None


def _parse_hp(engine_str: str) -> Optional[float]:
    import re

    match = re.search(r"([\d\.]+)\s*HP", engine_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_displacement(engine_str: str) -> Optional[float]:
    import re

    match = re.search(r"([\d\.]+)\s*L", engine_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _derive_deal_category(price_paid: Optional[float], market_price: Optional[float]) -> Optional[str]:
    if price_paid is None or market_price is None or market_price <= 0:
        return None

    delta_pct = (price_paid - market_price) / market_price * 100
    if delta_pct <= -10:
        return "Exceptional Deal"
    if delta_pct <= -3:
        return "Good Deal"
    if abs(delta_pct) <= 5:
        return "Fair Deal"
    if delta_pct <= 10:
        return "Slightly Overpriced"
    return "Overpriced"


def _extract_average_price(payload: Any) -> Optional[float]:
    keys = {
        "average_market_price",
        "averageMarketPrice",
        "average_price",
        "average",
        "market_average_price",
        "marketAverage",
        "mean_price",
        "retail_average",
        "retailAverage",
        "wholesale_average",
    }

    def _coerce(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace("$", "").replace(",", ""))
            except ValueError:
                return None
        return None

    def _search(obj: Any) -> Optional[float]:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in keys:
                    maybe = _coerce(val)
                    if maybe is not None:
                        return maybe
                nested = _search(val)
                if nested is not None:
                    return nested
        elif isinstance(obj, list):
            for item in obj:
                nested = _search(item)
                if nested is not None:
                    return nested
        return None

    return _search(payload)


def seed_from_csv(csv_path: Path, *, limit: int, offset: int, sleep: float) -> None:
    carsxe_client = CarsXEClient()
    if not carsxe_client.available:
        raise RuntimeError("CarsXE client unavailable. Install carsxe-api and set CARSXE_API_KEY.")

    db_manager = DatabaseManager()
    embedding_manager = EmbeddingManager()
    vector_manager = VectorStoreManager(embedding_manager=embedding_manager)

    processed = 0
    successes = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx < offset:
                continue
            if processed >= limit:
                break

            processed += 1

            make = (row.get("brand") or "").strip()
            model = (row.get("model") or "").strip()
            try:
                year = int(row.get("model_year", "0") or 0)
            except ValueError:
                logger.warning("Skipping row %s due to invalid year: %s", idx, row.get("model_year"))
                continue

            mileage = _parse_int(row.get("milage")) or 0
            price_paid = _parse_float(row.get("price"))
            engine_info = row.get("engine", "") or ""

            hp = _parse_hp(engine_info) or 0.0
            displacement = _parse_displacement(engine_info) or 0.0
            fuel_type = (row.get("fuel_type") or "UNKNOWN").strip()
            transmission = (row.get("transmission") or "UNKNOWN").strip()
            clean_title = 1 if (row.get("clean_title") or "").strip().lower() == "yes" else 0

            try:
                carsxe_payload = carsxe_client.fetch_market_value_by_trim(
                    make=make,
                    model=model,
                    year=year,
                    mileage=mileage or None,
                )
            except Exception as exc:
                logger.error("CarsXE lookup failed for %s %s %s: %s", year, make, model, exc)
                continue

            market_price = _extract_average_price(carsxe_payload)
            price_delta = None
            price_delta_pct = None
            deal_category = None
            if price_paid is not None and market_price is not None:
                price_delta = price_paid - market_price
                price_delta_pct = price_delta / market_price * 100
                deal_category = _derive_deal_category(price_paid, market_price)

            car_payload = {
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "price_paid": price_paid or 0.0,
                "trim": model,
                "engine": engine_info,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "hp": hp,
                "engine_displacement": displacement,
                "condition": row.get("accident"),
                "clean_title": clean_title,
                "pdf_source": str(csv_path),
                "raw_text": json.dumps(row, ensure_ascii=False),
            }

            try:
                car_id = db_manager.save_car(car_payload)
            except Exception as exc:
                logger.error("Failed to save car %s %s %s: %s", year, make, model, exc)
                continue

            analysis_data = {
                "rule_based_score": None,
                "rule_based_verdict": deal_category,
                "llm_score": None,
                "llm_verdict": deal_category,
                "llm_reasoning": json.dumps(carsxe_payload, ensure_ascii=False)[:2000],
                "market_median_price": market_price,
                "price_delta": price_delta,
                "price_delta_percent": price_delta_pct,
                "deal_category": deal_category,
                "data_source": "carsxe",
                "comparable_count": None,
                "research_quality": "carsxe_api",
                "success": market_price is not None,
                "analysis_version": "carsxe_seed_v1",
            }

            try:
                analysis_id = db_manager.save_analysis(car_id, analysis_data)
            except Exception as exc:
                logger.error("Failed to save analysis for car ID %s: %s", car_id, exc)
                continue

            try:
                car_dict = db_manager.get_car(car_id)
                if car_dict:
                    vector_manager.add_car(car_id, car_dict)

                car_with_analysis = db_manager.get_car_with_analysis(car_id)
                if car_with_analysis and car_with_analysis.get("analysis"):
                    vector_manager.add_analysis(analysis_id, car_id, car_with_analysis["analysis"])

                knowledge_content = (
                    f"CarsXE valuation for {year} {make} {model} with {mileage:,} miles. "
                )
                if market_price is not None:
                    knowledge_content += f"Average market price ${market_price:,.0f}."
                else:
                    knowledge_content += "Average market price unavailable."

                knowledge_content += "\n\nDataset price: "
                if price_paid is not None:
                    knowledge_content += f"${price_paid:,.0f}"
                    if price_delta is not None and price_delta_pct is not None:
                        knowledge_content += f" (delta ${price_delta:+,.0f}, {price_delta_pct:+.1f}%)."
                else:
                    knowledge_content += "unknown."

                knowledge_content += "\n\nCarsXE raw payload:\n" + json.dumps(
                    carsxe_payload, ensure_ascii=False
                )[:3000]

                knowledge_data = {
                    "title": f"{year} {make} {model} CarsXE valuation",
                    "content": knowledge_content,
                    "content_type": "carsxe_market",
                    "category": make,
                    "tags": [model, str(year), "carsxe"],
                    "source": "carsxe_api",
                    "reliability_score": 0.8,
                }

                knowledge_id = db_manager.add_knowledge(**knowledge_data)
                vector_manager.add_knowledge(knowledge_id, knowledge_data)
            except Exception as exc:
                logger.warning("Vector/knowledge sync failed for car ID %s: %s", car_id, exc)

            successes += 1
            logger.info(
                "Seeded %s %s %s (car_id=%s, analysis_id=%s, market_price=%s)",
                year,
                make,
                model,
                car_id,
                analysis_id,
                f"${market_price:,.0f}" if market_price is not None else "N/A",
            )

            if sleep > 0:
                time.sleep(sleep)

    logger.info("Completed CarsXE seeding: processed=%s successes=%s", processed, successes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed database with CarsXE valuations")
    parser.add_argument("--csv", type=Path, default=Path("used_cars.csv"), help="CSV dataset path")
    parser.add_argument("--limit", type=int, default=25, help="Number of rows to process")
    parser.add_argument("--offset", type=int, default=0, help="Number of initial rows to skip")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between CarsXE API calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    csv_path: Path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")

    seed_from_csv(csv_path, limit=args.limit, offset=args.offset, sleep=args.sleep)


if __name__ == "__main__":
    main()

