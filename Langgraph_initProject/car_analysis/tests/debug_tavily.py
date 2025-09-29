#!/usr/bin/env python3
"""Debug Tavily search functionality"""

import os
import re
import asyncio
import json
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

async def test_tavily_search():
    """Test Tavily search with detailed debugging"""

    print("🔍 Debugging Tavily Search")
    print("=" * 50)

    # Check API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("❌ TAVILY_API_KEY not found in environment")
        return

    print(f"✅ TAVILY_API_KEY found: {tavily_api_key[:10]}...")

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_api_key)
        print("✅ TavilyClient initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TavilyClient: {e}")
        return

    # Test different search queries
    test_queries = [
        "2020 Toyota Camry used car price for sale",
        "used Toyota Camry 2020 price",
        "Toyota Camry 2020 for sale",
        "2020 Toyota Camry market value",
        "buy used 2020 Toyota Camry"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)

        try:
            # Search with Tavily
            search_result = client.search(
                query=query,
                search_depth="basic",
                max_results=5
            )

            print(f"✅ Search completed")
            print(f"📊 Results returned: {len(search_result.get('results', []))}")

            # Show first few results
            for j, result in enumerate(search_result.get("results", [])[:3]):
                print(f"\n   Result {j+1}:")
                print(f"   📰 Title: {result.get('title', 'No title')[:100]}...")
                print(f"   🔗 URL: {result.get('url', 'No URL')}")
                print(f"   📝 Content: {result.get('content', 'No content')[:200]}...")

                # Test price extraction on this result
                content = result.get("content", "") + " " + result.get("title", "")
                extracted_prices = extract_prices_from_text(content)
                if extracted_prices:
                    print(f"   💰 Extracted prices: {extracted_prices}")
                else:
                    print(f"   ⚠️  No prices found in this result")

            # Overall price extraction for this query
            all_prices = []
            for result in search_result.get("results", []):
                content = result.get("content", "") + " " + result.get("title", "")
                prices = extract_prices_from_text(content)
                all_prices.extend(prices)

            unique_prices = sorted(list(set(all_prices)))
            print(f"\n   📊 Query Summary:")
            print(f"   💰 Total unique prices found: {len(unique_prices)}")
            if unique_prices:
                print(f"   💵 Price range: ${min(unique_prices):,.0f} - ${max(unique_prices):,.0f}")
                median = unique_prices[len(unique_prices)//2] if unique_prices else 0
                print(f"   📈 Median price: ${median:,.0f}")

        except Exception as e:
            print(f"❌ Search failed: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.5)  # Rate limiting

def extract_prices_from_text(text):
    """Extract prices from text using multiple patterns"""

    prices = []

    # Enhanced price extraction patterns
    price_patterns = [
        r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)',  # $25,999 or $25,999.00
        r'([0-9]{1,3}(?:,[0-9]{3})*)\s*dollars?',         # 25,999 dollars
        r'Price:?\s*\$?([0-9]{1,3}(?:,[0-9]{3})*)',       # Price: $25999
        r'Asking:?\s*\$?([0-9]{1,3}(?:,[0-9]{3})*)',      # Asking: $25999
        r'\$([0-9]{5,7})',                                 # $25999 (5-7 digits)
        r'([0-9]{2,3})[kK]',                              # 25k format
        r'starting\s+at\s+\$([0-9,]+)',                   # starting at $25,999
        r'MSRP:?\s*\$([0-9,]+)',                          # MSRP: $25999
    ]

    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Clean price string
                if isinstance(match, tuple):
                    match = match[0] if match else ""

                price_str = str(match).replace(',', '').replace('$', '').strip()

                # Handle 'k' notation
                if price_str.endswith('k') or price_str.endswith('K'):
                    price = float(price_str[:-1]) * 1000
                else:
                    price = float(price_str)

                # Filter reasonable car prices (5k-100k)
                if 5000 <= price <= 100000:
                    prices.append(price)

            except (ValueError, AttributeError):
                continue

    return prices

if __name__ == "__main__":
    asyncio.run(test_tavily_search())