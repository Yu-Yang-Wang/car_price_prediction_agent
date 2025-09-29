# 🚗 Car Analysis Agent - LangGraph Multi-Car Price Analysis

A sophisticated multi-car price analysis system built with LangGraph that evaluates used car deals using real web data via Tavily search and dual scoring (rule-based + LLM).

## ✨ Features

- **Real Web Search**: Uses Tavily API to fetch actual market prices from car listing websites
- **Dual Scoring System**: Combines rule-based algorithms with GPT-4o LLM analysis
- **Multi-Agent Architecture**: Dedicated agents for condition checks, market pricing, residual prediction, external (CarsXE/RAG) insights, and summary reporting
- **Enhanced Error Handling**: Proper retry limits, transparent failure reporting, no fallback to fake data
- **Comprehensive Price Extraction**: Advanced regex patterns to extract prices from various formats
- **LangGraph Workflow**: Sophisticated state management with conditional edges and retry logic

## 🏗️ Architecture

```
car_analysis/
├── __init__.py          # Main exports and module setup
├── models.py            # TypedDict state definitions and data structures
├── workers.py           # Core analysis workers (research, comparison, scoring, LLM)
├── checkers.py          # Retry logic and validation functions
├── orchestrator.py      # Report generation and workflow coordination
└── graph.py            # LangGraph state graph definition and routing
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install langgraph langchain-core langchain-openai tavily-python python-dotenv

# Optional tooling for external market data + ML predictor
pip install carsxe-api scikit-learn joblib
```

### 2. Set Up Environment Variables

The `.env` file is already configured with API keys:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=<your-openai-api-key>

# Tavily Search API Configuration
TAVILY_API_KEY=<your-tavily-api-key>

# Optional: CarsXE API for additional market data
CARSXE_API_KEY=your-carsxe-key

# Optional: override path to ML predictor artefact
# CAR_ML_MODEL_PATH=/absolute/path/to/used_car_price_rf.joblib
```

### 3. Seed optional knowledge sources

```bash
python -m car_analysis.utils.seed_carsxe_data --csv used_cars.csv --limit 25
```

This fetches CarsXE valuations for rows in `used_cars.csv`, stores them in the
database, and syncs them to the vector store to warm up RAG queries.

### 4. Run the Analysis

#### Option A: Test with Sample Cars
```bash
python test_modular_analysis.py
```

This will analyze 3 sample cars and generate a detailed report.

#### Option B: Debug Tavily Search
```bash
python debug_tavily.py
```

This tests Tavily search functionality directly with detailed debugging output.

#### Option C: Test Failure Scenarios
```bash
python test_failure_scenarios.py
```

This tests the enhanced error handling by simulating API failures.

### Optional: CarsXE market lookup helper

```python
from car_analysis.tools.carsxe_api import carsxe_client

valuation = carsxe_client.fetch_market_value_by_trim(
    make="Toyota",
    model="Camry",
    year=2020,
    mileage=45000,
)
```

> The helper stays disabled unless the `carsxe-api` package is installed and
> `CARSXE_API_KEY` is present.  Wrap calls in `try/except CarsXENotConfigured` if
> you need to fall back gracefully.

### Optional: ML predictor tool scaffold

```python
from car_analysis.tools.ml_predictor import ml_predictor

forecast = ml_predictor.predict_price({
    "year": 2019,
    "mileage": 42000,
    "make": "Toyota",
    "model": "Camry",
    "state": "CA",
})
```

Train a scikit-learn pipeline (e.g. RandomForest) on your Kaggle dataset, store
it with `joblib.dump`, and either place it at
`car_analysis/models/used_car_price_rf.joblib` or set `CAR_ML_MODEL_PATH`.

## 📊 Expected Output

When you run `test_modular_analysis.py`, you'll see:

```
🚗 Testing Modular Multi-Car Price Analysis Agent
============================================================
📋 Testing with 3 sample cars

🚗 Analyzing Car 1/3

🚗 Analyzing: 2020 Toyota Camry
==================================================
🔍 Researching market prices with Tavily...
   🔍 Searching: used 2020 Toyota Camry for sale price
   🔍 Searching: 2020 Toyota Camry used car market value
   🔍 Searching: 2020 Toyota Camry 35000 miles used car
   🔍 Searching: buy used 2020 Toyota Camry
   🔍 Searching: 2020 Toyota Camry trade-in value
   ✅ Found 20 comparable listings
   📊 Market median: $17,368
   📈 Price range: $8,427 - $34,991
📊 Comparing prices...
   💰 Paid: $22,500
   📈 Market: $17,368
   📊 Delta: $+5,132 (+29.5%)
   🎯 Verdict: Overpaid
🎯 Scoring the deal...
   🎯 Deal Score: 45/100
   📈 Final Verdict: Poor Deal ⚠️
🧠 Getting LLM opinion...
   🧠 LLM Score: 30/100
   🧠 LLM Verdict: Poor
📊 Generating final report...
   🤖 Rule-based: 45/100 - Poor Deal ⚠️
   🧠 LLM Opinion: 30/100 - Poor
   💰 Paid: $22,500 | Market: $17,368
   🎯 Score: 45/100 | Poor Deal ⚠️
```

## 🔧 How It Works

### 0. Agent framework
- **Condition Agent** – inspects vehicle basics (title, accident history, mileage)
- **Market Agent** – runs Tavily market research, price comparison, dual scoring
- **Residual Agent** – calls the ML predictor (joblib pipeline)
- **CarsXE Agent** – fetches external market valuations via CarsXE API
- **RAG Vector Agent** – retrieves similar cases & contextual analysis from the knowledge base
- **Summary Agent** – aggregates all findings into structured/Markdown output

### 1. Price Research Worker (`workers.py`)
- Uses Tavily API to search 5 different queries per car
- Extracts prices using comprehensive regex patterns
- Handles various price formats: "$16,699 to $25,995", "range from $14,062 - $27,917", etc.
- Calculates median from unique price listings

### 2. Price Comparison Worker (`workers.py`)
- Compares paid price vs market median
- Categorizes deals: Exceptional/Good/Fair/Slightly Overpaid/Overpaid
- Calculates price delta and percentage difference

### 3. Deal Scoring Worker (`workers.py`)
- Rule-based scoring algorithm (0-100)
- Factors: price delta, mileage vs expected, car age, data quality
- Assigns verdict levels with star ratings

### 4. LLM Opinion Worker (`workers.py`)
- GPT-4o analysis for second opinion
- JSON-structured response with score, verdict, reasoning
- Validates against rule-based scoring

### 5. Retry & Validation Logic (`checkers.py`)
- Max retry limits prevent infinite loops
- Disagreement detection between rule-based and LLM scores
- Transparent error reporting in final results

## 📁 Generated Reports

The system generates detailed JSON reports with:

- **Individual car analysis** with market data, scoring breakdown, and LLM opinions
- **Summary statistics** including success rates and score distributions
- **Error analysis** with categorized failure reasons
- **Comparative analysis** between rule-based and LLM scoring

Example report file: `modular_car_analysis_test_20250916_145106.json`

## 🐛 Troubleshooting

### Missing Dependencies
```bash
pip install langgraph langchain-core langchain-openai tavily-python python-dotenv
```

### API Key Issues
- Check that `.env` file exists and contains valid API keys
- Verify Tavily API key is active: `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`
- Verify OpenAI API key is active

### No Price Results Found
- The enhanced price extraction patterns should find results from most car searches
- If still having issues, run `debug_tavily.py` to test Tavily connectivity directly

## 🎯 Key Improvements

1. **Enhanced Price Extraction**: Comprehensive regex patterns extract prices from various web formats
2. **No Fallback Data**: System reports actual failures instead of using simulated data
3. **Proper Retry Limits**: Prevents infinite loops with max retry counts
4. **Real Web Search**: Tavily integration provides actual market data from car listing sites
5. **Dual Validation**: Rule-based + LLM scoring with disagreement detection

## 🔍 Example Analysis Results

- **2020 Toyota Camry**: $22,500 paid vs $17,368 market → 45/100 (Poor Deal)
- **2019 Honda Civic**: $18,200 paid vs $18,980 market → 85/100 (Good Deal)
- **2021 Ford F-150**: $31,800 paid vs $29,608 market → 60/100 (Fair Deal)

Each analysis includes detailed market data, price breakdowns, and both algorithmic and AI-powered evaluations.
