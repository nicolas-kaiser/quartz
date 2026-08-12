# Quartz Demo — Streamlit App

Interactive testing and demo UI for Quartz. Real market data comes from Yahoo
Finance; ESG and climate scores are **randomly generated placeholders** (seeded,
so reproducible), stored locally as CSV.

## How it works

```
Yahoo Finance ──▶ fetch_data.py ──▶ demo/data/*.csv
                                         │
Streamlit (app.py) ── builds JSON spec ──▶ quartz-demo (Rust binary)
                  ◀── JSON solution ─────  PortfolioModel::solve() / Clarabel
```

The Streamlit app estimates annualized expected returns and the covariance
matrix from daily prices, attaches tags (currency, sector) and scores to each
asset, and sends the whole problem as JSON to the `quartz-demo` binary over
stdin. The binary builds the `Universe`/`Strategy`/`Restrictions` with the
regular Quartz API, solves, and returns the solution as JSON on stdout.

## Setup

From the repo root:

```sh
# 1. Build the Rust bridge binary
cargo build --release -p quartz-demo

# 2. Install Python dependencies
pip install -r demo/requirements.txt

# 3. Fetch market data (writes demo/data/prices.csv and demo/data/assets.csv)
python demo/fetch_data.py

# 4. Launch the app
streamlit run demo/app.py
```

If Yahoo Finance is unreachable, `fetch_data.py` falls back to seeded synthetic
price paths and says so.

## What you can play with

- **Risk model** — full sample covariance, or a k-factor PCA model
  (Σ = BFBᵀ + D estimated by eigendecomposition) that exercises Quartz's
  O(nk²) factor covariance path
- **Objective weights** — variance, expected return, environmental / social /
  governance scores, transition and physical risk (Quartz normalizes them)
- **Group constraints** — min/max allocation per currency or sector
- **Portfolio score bounds** — e.g. average environmental score ≥ 7
- **Restrictions** — long-only, max weight per asset, exclusions by asset or sector
- **Pareto Frontier tab** — sweep any two objectives against each other
  (e.g. volatility vs expected return); dominated points shown grey, the
  efficient frontier connected, with your current strategy marked on the chart

The "Universe & Data" tab shows the asset table, normalized prices, and the
return correlation matrix. The JSON spec sent to Rust is visible in an expander
on the Optimize tab.

## Files

| File | Purpose |
|------|---------|
| `fetch_data.py` | Downloads prices, generates random ESG scores, writes CSVs |
| `app.py` | Streamlit UI, spec builder, solver bridge |
| `data/prices.csv` | Daily adjusted close prices (dates × tickers) |
| `data/assets.csv` | Asset tags + random ESG/climate scores |
| `../crates/quartz-demo` | Rust binary: JSON spec in → JSON solution out |
