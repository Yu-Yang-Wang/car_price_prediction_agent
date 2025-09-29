"""Worker functions for car analysis"""

import os
import re
import asyncio
from datetime import datetime
from typing import Dict, Any
from .models import CarAnalysisState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ..nodes.tools import get_llm

# Preferred listing/valuation domains to reduce noise/SEO pages
PREFERRED_DOMAINS = {
    "autotrader.com",
    "cars.com",
    "cargurus.com",
    "edmunds.com",
    "kbb.com",
}

# Relaxed filtering knobs so the price range isn't overly tight
MILEAGE_WINDOW_PCT = 0.40  # allow ±40% mileage window
IQR_MULTIPLIER = 2.0       # trim only extreme outliers
IQR_MIN_SAMPLES = 12       # apply IQR only when enough samples

llm_opinion_prompt = ChatPromptTemplate.from_template("""
You are a professional car market analyst. Evaluate the fairness of this used car deal:

Car Info:
- Year: {year}
- Make: {make}
- Model: {model}
- Mileage: {mileage}
- Paid Price: ${paid_price}
- Market Median Price: ${market_median}

Context (optional):
{context}

Respond in JSON with the fields:
- score (0-100)
- verdict (Exceptional/Good/Fair/Poor/Bad)
- reasoning (short explanation)
""")

_LLM_CHAIN = None


def _get_llm_chain():
    global _LLM_CHAIN
    if _LLM_CHAIN is None:
        llm = get_llm()
        _LLM_CHAIN = llm_opinion_prompt | llm | StrOutputParser()
    return _LLM_CHAIN

async def price_research_worker(state: CarAnalysisState) -> CarAnalysisState:
    """Worker 1: Search online prices via Tavily web search"""
    print("🔍 Researching market prices with Tavily...")

    car = state.get("current_car", {})
    if not car:
        return {"price_research": {"success": False, "error": "No current car"}}

    make = car.get("make", "")
    model = car.get("model", "")
    year = car.get("year", 0)
    mileage = car.get("mileage", 0)

    try:
        # Initialize Tavily client
        from tavily import TavilyClient

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        client = TavilyClient(api_key=tavily_api_key)

        # Optimized search queries (remove trade-in related terms)
        search_queries = [
            f"used {year} {make} {model} for sale price",
            f"{year} {make} {model} used car market value",
            f"{year} {make} {model} {mileage} miles used car",
            f"buy used {year} {make} {model}",
        ]

        all_extracted_prices = []

        from urllib.parse import urlparse

        def domain_allowed(url: str) -> bool:
            try:
                netloc = urlparse(url).netloc.lower()
                # Strip subdomain
                parts = netloc.split(":")[0].split(".")
                base = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
                return base in PREFERRED_DOMAINS
            except Exception:
                return False

        def extract_mileages(text: str):
            # Find mileage-like numbers (e.g., 35,000 miles, 35000 mi)
            patterns = [
                r"([0-9]{1,3}(?:,[0-9]{3})+)\s*(?:miles|mi)\b",
                r"\b([0-9]{4,6})\s*(?:miles|mi)\b",
            ]
            vals = []
            import re as _re
            for pat in patterns:
                for m in _re.findall(pat, text, _re.IGNORECASE):
                    try:
                        s = m if isinstance(m, str) else m[0]
                        vals.append(int(str(s).replace(",", "")))
                    except Exception:
                        continue
            return vals

        def within_mileage_window(text: str, target: int, pct: float = MILEAGE_WINDOW_PCT) -> bool:
            if not target:
                return True
            vals = extract_mileages(text)
            if not vals:
                # No mileage in text: allow but we already restrict by domain+title
                return True
            low, high = int(target * (1 - pct)), int(target * (1 + pct))
            return any(low <= v <= high for v in vals)

        for query in search_queries:
            print(f"   🔍 Searching: {query}")

            try:
                # Search with Tavily
                search_result = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=12,
                    include_domains=list(PREFERRED_DOMAINS),
                )

                # Extract prices from search results
                for result in search_result.get("results", []):
                    title = result.get("title", "")
                    url = result.get("url", "")
                    content = result.get("content", "") + " " + title

                    # Filter by domain
                    if not domain_allowed(url):
                        continue

                    # Filter by title relevance: require make+model and preferably year
                    title_lc = title.lower()
                    if not (make.lower() in title_lc and model.lower() in title_lc):
                        continue
                    # If a year appears in title, prefer matching year; if not present, still allow
                    if str(year) not in title_lc:
                        # allow, but later mileage filter will further constrain
                        pass

                    # Filter by mileage proximity when detectable
                    if not within_mileage_window(content, mileage, pct=MILEAGE_WINDOW_PCT):
                        continue

                    # Enhanced price extraction patterns with more comprehensive coverage
                    price_patterns = [
                        r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)',  # $25,999 or $25,999.00
                        r'from\s+\$([0-9]{1,3}(?:,[0-9]{3})*)',          # from $25,999
                        r'to\s+\$([0-9]{1,3}(?:,[0-9]{3})*)',            # to $25,999
                        r'range\s+from\s+\$([0-9]{1,3}(?:,[0-9]{3})*)',  # range from $25,999
                        r'sale\s+from\s+\$([0-9]{1,3}(?:,[0-9]{3})*)',   # sale from $25,999
                        r'starting\s+at\s+\$([0-9]{1,3}(?:,[0-9]{3})*)', # starting at $25,999
                        r'prices?\s+range\s+from\s+\$([0-9]{1,3}(?:,[0-9]{3})*)', # prices range from $25,999
                        r'([0-9]{1,3}(?:,[0-9]{3})*)\s*dollars?',         # 25,999 dollars
                        r'Price:?\s*\$?([0-9]{1,3}(?:,[0-9]{3})*)',       # Price: $25999
                        r'Asking:?\s*\$?([0-9]{1,3}(?:,[0-9]{3})*)',      # Asking: $25999
                        r'MSRP:?\s*\$([0-9]{1,3}(?:,[0-9]{3})*)',         # MSRP: $25999
                        r'trade-in\s+prices?\s+range\s+from\s+\$([0-9]{1,3}(?:,[0-9]{3})*)', # trade-in prices range from $25,999
                        r'\$([0-9]{5,7})',                                 # $25999 (5-7 digits)
                        r'([0-9]{2,3})[kK]',                              # 25k format
                    ]

                    for pattern in price_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            try:
                                # Handle tuple matches from groups
                                if isinstance(match, tuple):
                                    match = match[0] if match else ""

                                # Clean price string
                                price_str = str(match).replace(',', '').replace('$', '').strip()

                                # Handle 'k' notation
                                if price_str.endswith('k') or price_str.endswith('K'):
                                    price = float(price_str[:-1]) * 1000
                                else:
                                    price = float(price_str)

                                # Filter reasonable car prices (5k-120k)
                                if 5000 <= price <= 120000:
                                    all_extracted_prices.append(price)
                            except (ValueError, AttributeError):
                                continue

                await asyncio.sleep(0.2)  # Rate limiting

            except Exception as search_error:
                print(f"   ⚠️ Search query failed: {search_error}")
                continue

        # Process extracted prices
        if all_extracted_prices:
            # Remove duplicates and sort
            unique_prices = sorted(list(set(all_extracted_prices)))

            # IQR trimming to remove outliers
            def iqr_trim(values):
                if len(values) < IQR_MIN_SAMPLES:
                    return values
                import math
                qs = values
                q1_idx = int(0.25 * (len(qs) - 1))
                q3_idx = int(0.75 * (len(qs) - 1))
                Q1, Q3 = qs[q1_idx], qs[q3_idx]
                IQR = Q3 - Q1
                low, high = Q1 - IQR_MULTIPLIER * IQR, Q3 + IQR_MULTIPLIER * IQR
                trimmed = [v for v in qs if low <= v <= high]
                if len(trimmed) < max(5, int(0.7 * len(qs))):
                    return values
                return trimmed

            trimmed = iqr_trim(unique_prices)
            if len(trimmed) >= 5:
                filtered_prices = trimmed
            else:
                filtered_prices = unique_prices

            # Recompute median on filtered
            n = len(filtered_prices)
            if n == 0:
                raise ValueError("No valid prices after filtering")
            if n % 2 == 0:
                median_price = (filtered_prices[n//2 - 1] + filtered_prices[n//2]) / 2
            else:
                median_price = filtered_prices[n//2]

            research_result = {
                "success": True,
                "search_queries": search_queries,
                "raw_results_count": len(search_result.get("results", [])) * len(search_queries),
                "extracted_prices": filtered_prices,
                "median_price": median_price,
                "price_range": {
                    "min": min(filtered_prices),
                    "max": max(filtered_prices)
                },
                "sample_count": len(filtered_prices),
                "search_method": "tavily_real_web_search",
                "timestamp": datetime.now().isoformat()
            }

            print(f"   ✅ Found {len(filtered_prices)} comparable listings (filtered)")
            print(f"   🧹 Filters: domains={','.join(sorted(PREFERRED_DOMAINS))}, mileage±{int(MILEAGE_WINDOW_PCT*100)}%, IQR×{IQR_MULTIPLIER}")
            print(f"   📊 Market median: ${median_price:,.0f}")
            print(f"   📈 Price range: ${min(filtered_prices):,.0f} - ${max(filtered_prices):,.0f}")

        else:
            # No fallback data - report failure
            error_msg = f"Failed to find any valid car prices from {len(search_queries)} search queries"
            print(f"   ❌ {error_msg}")

            research_result = {
                "success": False,
                "error": error_msg,
                "search_queries": search_queries,
                "extracted_prices": [],
                "sample_count": 0,
                "search_method": "tavily_real_web_search_failed",
                "timestamp": datetime.now().isoformat()
            }

        return {
            "price_research": research_result,
            "dbg_logs": [f"Price research completed for {year} {make} {model} via {research_result['search_method']}"]
        }

    except Exception as e:
        print(f"   ❌ Price research failed: {e}")
        return {
            "price_research": {"success": False, "error": str(e)},
            "dbg_logs": [f"Price research failed: {e}"]
        }


async def price_comparison_worker(state: CarAnalysisState) -> CarAnalysisState:
    """Worker 2: Compare price paid vs market prices"""
    print("📊 Comparing prices...")

    car = state.get("current_car", {})
    research = state.get("price_research", {})

    if not research.get("success"):
        return {"price_comparison": {"success": False, "error": "No price research data"}}

    try:
        price_paid = car.get("price_paid", 0)
        median_price = research.get("median_price", 0)
        price_range = research.get("price_range", {})

        # Calculate comparison metrics
        price_delta = price_paid - median_price
        price_delta_pct = (price_delta / median_price * 100) if median_price > 0 else 0

        # Determine if overpaid/underpaid
        if price_paid < price_range.get("min", 0):
            verdict_category = "Exceptional Deal"
        elif price_paid < median_price * 0.95:
            verdict_category = "Good Deal"
        elif price_paid <= median_price * 1.05:
            verdict_category = "Fair Price"
        elif price_paid <= median_price * 1.15:
            verdict_category = "Slightly Overpaid"
        else:
            verdict_category = "Overpaid"

        # Avoid divide-by-zero and clamp percentile to [0,100]
        denom = max(1e-9, (price_range.get("max", 1) - price_range.get("min", 0)))
        percentile = ((price_paid - price_range.get("min", 0)) / denom) * 100
        percentile = max(0.0, min(100.0, percentile))

        comparison_result = {
            "success": True,
            "price_paid": price_paid,
            "market_median": median_price,
            "price_delta": price_delta,
            "price_delta_pct": price_delta_pct,
            "verdict_category": verdict_category,
            "market_position": {
                "percentile": percentile
            }
        }

        print(f"   💰 Paid: ${price_paid:,.0f}")
        print(f"   📈 Market: ${median_price:,.0f}")
        print(f"   📊 Delta: ${price_delta:+,.0f} ({price_delta_pct:+.1f}%)")
        print(f"   🎯 Verdict: {verdict_category}")

        return {
            "price_comparison": comparison_result,
            "dbg_logs": [f"Price comparison: {verdict_category} (${price_delta:+,.0f})"]
        }

    except Exception as e:
        print(f"   ❌ Price comparison failed: {e}")
        return {
            "price_comparison": {"success": False, "error": str(e)},
            "dbg_logs": [f"Price comparison failed: {e}"]
        }


async def deal_scoring_worker(state: CarAnalysisState) -> CarAnalysisState:
    """Worker 3: Score the deal (0-100)"""
    print("🎯 Scoring the deal...")

    car = state.get("current_car", {})
    comparison = state.get("price_comparison", {})
    research = state.get("price_research", {})

    if not comparison.get("success"):
        return {"deal_score": {"success": False, "error": "No price comparison data"}}

    try:
        price_delta_pct = comparison.get("price_delta_pct", 0)
        mileage = car.get("mileage", 100000)
        year = car.get("year", 2010)
        current_year = datetime.now().year
        car_age = current_year - year
        sample_count = research.get("sample_count", 0)
        search_method = research.get("search_method", "unknown")

        # Scoring algorithm (0-100)
        score = 50  # Base score

        # Price delta impact (most important factor)
        if price_delta_pct <= -20:  # 20%+ under market
            score += 40
        elif price_delta_pct <= -10:  # 10-20% under market
            score += 30
        elif price_delta_pct <= -5:   # 5-10% under market
            score += 20
        elif price_delta_pct <= 5:    # Within 5% of market
            score += 10
        elif price_delta_pct <= 15:   # 5-15% over market
            score -= 15
        else:  # 15%+ over market
            score -= 30

        # Mileage factor
        expected_mileage = car_age * 12000  # 12k miles/year
        mileage_delta = mileage - expected_mileage

        if mileage_delta <= -20000:  # Low mileage
            score += 15
        elif mileage_delta <= -10000:
            score += 10
        elif mileage_delta <= 10000:
            score += 5
        elif mileage_delta <= 30000:
            score -= 5
        else:  # High mileage
            score -= 15

        # Age factor
        if car_age <= 3:
            score += 5
        elif car_age >= 10:
            score -= 5

        # Data quality bonus for real Tavily data
        data_quality_bonus = 0
        if search_method == "tavily_real_web_search" and sample_count >= 10:
            data_quality_bonus = 10
            score += data_quality_bonus

        # Ensure score is 0-100
        score = max(0, min(100, score))

        # Final verdict
        if score >= 90:
            verdict = "Exceptional Deal ⭐⭐⭐"
        elif score >= 75:
            verdict = "Good Deal ⭐⭐"
        elif score >= 60:
            verdict = "Fair Deal ⭐"
        elif score >= 40:
            verdict = "Poor Deal ⚠️"
        else:
            verdict = "Bad Deal ❌"

        scoring_result = {
            "success": True,
            "score": score,
            "verdict": verdict,
            "data_source": search_method,
            "confidence": "high" if search_method == "tavily_real_web_search" else "medium",
            "scoring_breakdown": {
                "price_impact": price_delta_pct,
                "mileage_vs_expected": mileage_delta,
                "car_age": car_age,
                "sample_size": sample_count,
                "data_quality_bonus": data_quality_bonus
            }
        }

        print(f"   🎯 Deal Score: {score}/100")
        print(f"   📈 Final Verdict: {verdict}")
        print(f"   🔍 Data Source: {search_method}")

        return {
            "deal_score": scoring_result,
            "dbg_logs": [f"Deal scoring: {score}/100 - {verdict} (data: {search_method})"]
        }

    except Exception as e:
        print(f"   ❌ Deal scoring failed: {e}")
        return {
            "deal_score": {"success": False, "error": str(e)},
            "dbg_logs": [f"Deal scoring failed: {e}"]
        }
    




async def llm_opinion_worker(state: CarAnalysisState) -> CarAnalysisState:
    """Worker: Get LLM opinion on the car deal"""
    print("🧠 Getting LLM opinion...")

    car = state.get("current_car", {})
    comparison = state.get("price_comparison", {})

    if not comparison.get("success") or not comparison.get("verdict_category"):
        print("   ⚠️  Price comparison not ready, skipping LLM opinion")
        return {
            **state,
            "llm_opinion": {
                "score": 0,
                "verdict": "Unknown",
                "reasoning": "Price comparison not available"
            }
        }

    try:
        llm_chain = _get_llm_chain()
        early = state.get("early_rag", {}) or {}
        context = early.get("brief") or ""
        output = await llm_chain.ainvoke({
            "year": car.get("year", "Unknown"),
            "make": car.get("make", "Unknown"),
            "model": car.get("model", "Unknown"),
            "mileage": car.get("mileage", 0),
            "paid_price": comparison.get("price_paid", 0),
            "market_median": comparison.get("market_median", 0),
            "context": context,
        })

        print(f"   🧠 LLM raw response: {output[:200]}...")

        import json
        try:
            # Clean the output - remove markdown code block if present
            clean_output = output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output.replace("```json", "").replace("```", "").strip()
            elif clean_output.startswith("```"):
                clean_output = clean_output.replace("```", "").strip()

            llm_json = json.loads(clean_output)

            # Ensure required fields exist
            if "score" not in llm_json:
                llm_json["score"] = 50
            if "verdict" not in llm_json:
                llm_json["verdict"] = "Fair"
            if "reasoning" not in llm_json:
                llm_json["reasoning"] = "LLM evaluation"

            print(f"   🧠 LLM Score: {llm_json.get('score', 0)}/100")
            print(f"   🧠 LLM Verdict: {llm_json.get('verdict', 'Unknown')}")

        except Exception as parse_error:
            print(f"   ❌ LLM JSON parsing failed: {parse_error}")
            llm_json = {
                "score": 50,
                "verdict": "Unknown",
                "reasoning": f"LLM parsing failed: {str(parse_error)}"
            }

        return {
            **state,
            "llm_opinion": llm_json,
            "dbg_logs": [f"LLM opinion: {llm_json.get('score', 0)}/100 - {llm_json.get('verdict', 'Unknown')}"]
        }

    except Exception as e:
        print(f"   ❌ LLM opinion failed: {e}")
        return {
            **state,
            "llm_opinion": {
                "score": 0,
                "verdict": "Error",
                "reasoning": f"LLM error: {str(e)}"
            },
            "dbg_logs": [f"LLM opinion failed: {e}"]
        }
