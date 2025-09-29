"""Lightweight GraphRAG service using Neo4j + Cypher.

This integrates optionally: if the Neo4j driver or env vars are missing,
the service becomes a no-op and callers should proceed without graph context.

Env vars:
  NEO4J_URI (default: bolt://localhost:7687)
  NEO4J_USER (default: neo4j)
  NEO4J_PASSWORD (required for auth)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore


class GraphService:
    """Minimal graph wrapper for upserts and a few retrieval queries."""

    def __init__(self):
        self._driver = None
        self._available = False

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if GraphDatabase is None:
            # Driver not installed; run as disabled
            return

        if not password:
            # Missing credentials; run as disabled
            return

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Probe connection
            with self._driver.session() as session:
                session.run("RETURN 1 AS ok").single()
            self._available = True
            print("🕸️ GraphService connected to Neo4j")
        except Exception as e:  # pragma: no cover
            print(f"⚠️ GraphService disabled: {e}")
            self._driver = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._driver is not None

    def close(self):  # pragma: no cover
        try:
            if self._driver:
                self._driver.close()
        finally:
            self._driver = None
            self._available = False

    # ---------- Upserts ----------
    def upsert_car(self, car_id: int, car: Dict[str, Any]) -> None:
        if not self.available:
            return
        q = (
            "MERGE (c:Car {id: $id}) "
            "SET c.make=$make, c.model=$model, c.year=$year, c.mileage=$mileage, "
            "    c.price_paid=$price_paid"
        )
        params = {
            "id": int(car_id),
            "make": car.get("make"),
            "model": car.get("model"),
            "year": int(car.get("year") or 0),
            "mileage": int(car.get("mileage") or 0),
            "price_paid": float(car.get("price_paid") or 0.0),
        }
        with self._driver.session() as s:
            s.run(q, **params)

    def upsert_analysis(self, analysis_id: int, analysis: Dict[str, Any]) -> None:
        if not self.available:
            return
        q = (
            "MERGE (a:Analysis {id: $id}) "
            "SET a.rule_based_score=$rule_based_score, a.rule_based_verdict=$rule_based_verdict, "
            "    a.llm_score=$llm_score, a.llm_verdict=$llm_verdict, a.market_median_price=$market_median_price, "
            "    a.price_delta=$price_delta, a.price_delta_percent=$price_delta_percent, a.deal_category=$deal_category, "
            "    a.research_quality=$research_quality, a.success=$success"
        )
        params = {
            "id": int(analysis_id),
            "rule_based_score": analysis.get("rule_based_score"),
            "rule_based_verdict": analysis.get("rule_based_verdict"),
            "llm_score": analysis.get("llm_score"),
            "llm_verdict": analysis.get("llm_verdict"),
            "market_median_price": analysis.get("market_median_price"),
            "price_delta": analysis.get("price_delta"),
            "price_delta_percent": analysis.get("price_delta_percent"),
            "deal_category": analysis.get("deal_category"),
            "research_quality": analysis.get("research_quality"),
            "success": bool(analysis.get("success", False)),
        }
        with self._driver.session() as s:
            s.run(q, **params)

    def link_car_analysis(self, car_id: int, analysis_id: int) -> None:
        if not self.available:
            return
        q = (
            "MATCH (c:Car {id:$car_id}), (a:Analysis {id:$analysis_id}) "
            "MERGE (c)-[:HAS_ANALYSIS]->(a)"
        )
        with self._driver.session() as s:
            s.run(q, car_id=int(car_id), analysis_id=int(analysis_id))

    # ---------- Retrieval ----------
    def context_for_car(self, car: Dict[str, Any], limit: int = 8) -> str:
        """Return a small, human-readable context from graph for a given car.

        Strategy: same make/model within ±2 years, highest rule-based scores.
        """
        if not self.available:
            return ""

        make = car.get("make")
        model = car.get("model")
        year = int(car.get("year") or 0)
        if not make or not model or not year:
            return ""

        q = (
            "MATCH (c:Car {make:$make, model:$model})-[:HAS_ANALYSIS]->(a:Analysis) "
            "WHERE abs(coalesce(c.year,0) - $year) <= 2 AND a.success = true "
            "RETURN c.year AS year, c.mileage AS mileage, c.price_paid AS price_paid, "
            "       a.rule_based_score AS score, a.deal_category AS category, a.market_median_price AS median "
            "ORDER BY score DESC NULLS LAST LIMIT $limit"
        )
        rows: List[Dict[str, Any]] = []
        with self._driver.session() as s:
            for rec in s.run(q, make=make, model=model, year=year, limit=limit):
                rows.append({k: rec.get(k) for k in ["year", "mileage", "price_paid", "score", "category", "median"]})

        if not rows:
            return ""

        lines = [f"Similar {make} {model} within ±2 years (top {len(rows)} by score):"]
        for r in rows:
            y = r.get("year")
            lines.append(
                f"- {y}: score {r.get('score')}, category {r.get('category')}, "
                f"paid ${r.get('price_paid'):,} vs median ${r.get('median'):,} (mileage {r.get('mileage'):,})"
            )
        return "\n".join(lines)

