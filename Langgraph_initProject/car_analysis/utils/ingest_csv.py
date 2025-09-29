"""Ingest various CSV datasets into the car analysis database for RAG use."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from car_analysis.database.manager import DatabaseManager
from car_analysis.rag.embeddings import EmbeddingManager
from car_analysis.rag.vector_store import VectorStoreManager


logger = logging.getLogger("csv_ingest")


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    cleaned = (
        value.replace("$", "")
        .replace(",", "")
        .replace(" mi.", "")
        .replace(" miles", "")
        .strip()
    )
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else None


def _sync_vectors(db_manager: DatabaseManager, vector_manager: VectorStoreManager, car_id: int) -> None:
    car_dict = db_manager.get_car(car_id)
    if car_dict:
        vector_manager.add_car(car_id, car_dict)

    car_with_analysis = db_manager.get_car_with_analysis(car_id)
    if car_with_analysis and car_with_analysis.get("analysis"):
        analysis = car_with_analysis["analysis"]
        analysis_id = analysis.get("id") or car_with_analysis.get("analysis", {}).get("id")
        if analysis_id is None:
            analysis_id = car_with_analysis["analysis"].get("analysis_id", car_id)
        sanitized = {
            key: ("" if value is None else value)
            for key, value in analysis.items()
        }
        vector_manager.add_analysis(analysis_id, car_id, sanitized)


def _insert_knowledge(
    db_manager: DatabaseManager,
    vector_manager: VectorStoreManager,
    title: str,
    content: str,
    *,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source: str = "csv_ingest",
    reliability: float = 0.6,
) -> None:
    knowledge_data = {
        "title": title,
        "content": content,
        "content_type": "csv_dataset",
        "category": category,
        "tags": tags or [],
        "source": source,
        "reliability_score": reliability,
    }
    knowledge_id = db_manager.add_knowledge(
        title=title,
        content=content,
        content_type=knowledge_data["content_type"],
        category=category,
        tags=tags,
        source=source,
    )
    vector_manager.add_knowledge(knowledge_id, knowledge_data)


def ingest_car_prices(
    path: Path,
    *,
    limit: int,
    offset: int,
    db_manager: DatabaseManager,
    vector_manager: VectorStoreManager,
) -> None:
    logger.info("Ingesting car_prices dataset from %s", path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx < offset:
                continue
            if (idx - offset) >= limit:
                break

            try:
                year = int(row.get("year", "0") or 0)
            except ValueError:
                logger.warning("Skipping row %s due to invalid year: %s", idx, row.get("year"))
                continue

            make = (row.get("make") or "").strip()
            model = (row.get("model") or "").strip()
            trim = (row.get("trim") or "").strip()
            mileage = _parse_int(row.get("odometer")) or 0
            price_paid = _parse_float(row.get("sellingprice")) or 0.0
            mmr_price = _parse_float(row.get("mmr"))
            vin = (row.get("vin") or "").strip().upper()

            car_payload = {
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "price_paid": price_paid,
                "trim": trim,
                "color": row.get("color"),
                "engine": row.get("body"),
                "transmission": row.get("transmission"),
                "condition": row.get("condition"),
                "location": row.get("state"),
                "pdf_source": str(path),
                "raw_text": json.dumps(row, ensure_ascii=False),
            }

            try:
                car_id = db_manager.save_car(car_payload)
            except Exception as exc:
                logger.error("Failed to save car %s %s %s: %s", year, make, model, exc)
                continue

            price_delta = None
            price_delta_pct = None
            deal_category = None
            if mmr_price is not None and mmr_price > 0:
                price_delta = price_paid - mmr_price
                price_delta_pct = price_delta / mmr_price * 100
                if price_delta_pct <= -10:
                    deal_category = "Exceptional Deal"
                elif price_delta_pct <= -3:
                    deal_category = "Good Deal"
                elif abs(price_delta_pct) <= 5:
                    deal_category = "Fair Deal"
                elif price_delta_pct <= 10:
                    deal_category = "Slightly Overpriced"
                else:
                    deal_category = "Overpriced"

            analysis_data = {
                "rule_based_score": None,
                "rule_based_verdict": deal_category,
                "llm_score": None,
                "llm_verdict": deal_category,
                "llm_reasoning": "MMR-based dataset entry",
                "market_median_price": mmr_price,
                "price_delta": price_delta,
                "price_delta_percent": price_delta_pct,
                "deal_category": deal_category,
                "data_source": "car_prices_csv",
                "comparable_count": None,
                "research_quality": "mmr_history",
                "success": mmr_price is not None,
                "analysis_version": "car_prices_v1",
            }

            try:
                analysis_id = db_manager.save_analysis(car_id, analysis_data)
            except Exception as exc:
                logger.error("Failed to save analysis for car ID %s: %s", car_id, exc)
                continue

            summary = (
                f"Historical sale on {row.get('saledate')} for {year} {make} {model} {trim}. "
                f"Selling price ${price_paid:,.0f}, MMR ${mmr_price:,.0f}"
                if mmr_price is not None
                else f"Historical sale on {row.get('saledate')} for {year} {make} {model} {trim}."
            )

            _insert_knowledge(
                db_manager,
                vector_manager,
                f"Historical sale: {year} {make} {model} ({vin or 'no VIN'})",
                summary,
                category=make,
                tags=[model, str(year), "car_prices_csv"],
                reliability=0.7,
            )

            _sync_vectors(db_manager, vector_manager, car_id)


def ingest_used_cars(
    path: Path,
    *,
    limit: int,
    offset: int,
    db_manager: DatabaseManager,
    vector_manager: VectorStoreManager,
) -> None:
    logger.info("Ingesting used_cars dataset from %s", path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx < offset:
                continue
            if (idx - offset) >= limit:
                break

            try:
                year = int(row.get("model_year", "0") or 0)
            except ValueError:
                logger.warning("Skipping row %s due to invalid year: %s", idx, row.get("model_year"))
                continue

            make = (row.get("brand") or "").strip()
            model = (row.get("model") or "").strip()
            mileage = _parse_int(row.get("milage")) or 0
            price_paid = _parse_float(row.get("price")) or 0.0

            car_payload = {
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "price_paid": price_paid,
                "color": row.get("ext_col"),
                "engine": row.get("engine"),
                "fuel_type": row.get("fuel_type"),
                "transmission": row.get("transmission"),
                "condition": row.get("accident"),
                "clean_title": 1 if (row.get("clean_title") or "").strip().lower() == "yes" else 0,
                "pdf_source": str(path),
                "raw_text": json.dumps(row, ensure_ascii=False),
            }

            try:
                car_id = db_manager.save_car(car_payload)
            except Exception as exc:
                logger.error("Failed to save car %s %s %s: %s", year, make, model, exc)
                continue

            analysis_data = {
                "rule_based_score": None,
                "rule_based_verdict": None,
                "llm_score": None,
                "llm_verdict": None,
                "llm_reasoning": "Raw used_cars CSV entry",
                "market_median_price": None,
                "price_delta": None,
                "price_delta_percent": None,
                "deal_category": None,
                "data_source": "used_cars_csv",
                "comparable_count": None,
                "research_quality": "raw_listing",
                "success": False,
                "analysis_version": "used_cars_v1",
            }

            try:
                db_manager.save_analysis(car_id, analysis_data)
            except Exception as exc:
                logger.warning("Analysis save failed for car ID %s: %s", car_id, exc)

            summary = (
                f"Listing: {year} {make} {model}, {mileage:,} miles, price ${price_paid:,.0f}. "
                f"Accident info: {row.get('accident')}. Clean title: {row.get('clean_title')}"
            )

            _insert_knowledge(
                db_manager,
                vector_manager,
                f"Used car listing {year} {make} {model}",
                summary,
                category=make,
                tags=[model, str(year), "used_cars_csv"],
                reliability=0.5,
            )

            _sync_vectors(db_manager, vector_manager, car_id)


def ingest_used_cars_data(
    path: Path,
    *,
    limit: int,
    offset: int,
    db_manager: DatabaseManager,
    vector_manager: VectorStoreManager,
) -> None:
    logger.info("Ingesting used_cars_data dataset from %s", path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx < offset:
                continue
            if (idx - offset) >= limit:
                break

            try:
                year = int(row.get("year", "0") or 0)
            except ValueError:
                logger.warning("Skipping row %s due to invalid year: %s", idx, row.get("year"))
                continue

            make = (row.get("make_name") or "").strip()
            model = (row.get("model_name") or "").strip()
            mileage = _parse_int(row.get("mileage")) or 0
            price_paid = _parse_float(row.get("price")) or 0.0

            car_payload = {
                "make": make,
                "model": model,
                "year": year,
                "mileage": mileage,
                "price_paid": price_paid,
                "trim": row.get("trim_name"),
                "color": row.get("exterior_color"),
                "engine": row.get("engine_type"),
                "fuel_type": row.get("fuel_type"),
                "transmission": row.get("transmission_display"),
                "condition": row.get("vehicle_damage_category"),
                "location": row.get("city"),
                "clean_title": 0 if (row.get("salvage") or "").strip().lower() == "true" else 1,
                "hp": _parse_float(row.get("horsepower")) or _parse_float(row.get("power")),
                "engine_displacement": _parse_float(row.get("engine_displacement")),
                "pdf_source": str(path),
                "raw_text": json.dumps(row, ensure_ascii=False),
            }

            try:
                car_id = db_manager.save_car(car_payload)
            except Exception as exc:
                logger.error("Failed to save car %s %s %s: %s", year, make, model, exc)
                continue

            days_on_market = _parse_int(row.get("daysonmarket"))
            seller_rating = _parse_float(row.get("seller_rating"))

            analysis_data = {
                "rule_based_score": None,
                "rule_based_verdict": None,
                "llm_score": None,
                "llm_verdict": None,
                "llm_reasoning": "used_cars_data CSV entry",
                "market_median_price": None,
                "price_delta": None,
                "price_delta_percent": None,
                "deal_category": None,
                "data_source": "used_cars_data_csv",
                "comparable_count": None,
                "research_quality": "rich_listing",
                "success": False,
                "analysis_version": "used_cars_data_v1",
            }

            try:
                db_manager.save_analysis(car_id, analysis_data)
            except Exception as exc:
                logger.warning("Analysis save failed for car ID %s: %s", car_id, exc)

            summary_parts = [
                f"Listing: {year} {make} {model}",
                f"Price ${price_paid:,.0f}",
                f"Mileage {mileage:,} mi.",
            ]
            if days_on_market is not None:
                summary_parts.append(f"Days on market: {days_on_market}")
            if seller_rating is not None:
                summary_parts.append(f"Seller rating: {seller_rating}")

            summary = "; ".join(summary_parts)

            _insert_knowledge(
                db_manager,
                vector_manager,
                f"Rich listing {year} {make} {model}",
                summary,
                category=make,
                tags=[model, str(year), "used_cars_data_csv"],
                reliability=0.6,
            )

            _sync_vectors(db_manager, vector_manager, car_id)


DATASET_HANDLERS = {
    "car_prices": ingest_car_prices,
    "used_cars": ingest_used_cars,
    "used_cars_data": ingest_used_cars_data,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CSV datasets into car analysis DB")
    parser.add_argument("dataset", choices=DATASET_HANDLERS.keys(), help="Dataset key")
    parser.add_argument("--csv", type=Path, required=True, help="Path to CSV file")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to process")
    parser.add_argument("--offset", type=int, default=0, help="Rows to skip before ingestion")
    parser.add_argument("--log", default="INFO", help="Logging level (INFO/DEBUG)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    db_manager = DatabaseManager()
    embedding_manager = EmbeddingManager()
    vector_manager = VectorStoreManager(embedding_manager=embedding_manager)

    handler = DATASET_HANDLERS[args.dataset]
    handler(
        args.csv,
        limit=args.limit,
        offset=args.offset,
        db_manager=db_manager,
        vector_manager=vector_manager,
    )


if __name__ == "__main__":
    main()
