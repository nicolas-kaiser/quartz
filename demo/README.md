# Quartz Demo — Streamlit App

Interactive testing and demo UI for Quartz. Real market data comes from Yahoo
Finance; ESG and climate scores are **randomly generated placeholders** (seeded,
so reproducible), stored locally as CSV.

## How it works

```
Yahoo Finance ──▶ fetch_data.py ──▶ demo/data/*.csv
                                         │
Streamlit (app.py) ──▶ import quartz (PyO3 bindings, in-process)
                        PortfolioModel::solve() / solve_batch / frontier / Clarabel
```

The Streamlit app estimates annualized expected returns and the covariance
matrix from daily prices, attaches tags (currency, sector) and scores to each
asset, and solves in-process through the `quartz` Python bindings. Batch
solves (the Backtest tab) release the GIL and run rayon-parallel in Rust.

## Setup

From the repo root:

```sh
# 1. Build and install the quartz Python bindings
pip install maturin
maturin build --release -m crates/quartz-python/Cargo.toml
pip install target/wheels/quartz-*.whl

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
- **Risk constraints** — max tracking error vs an equal-weight benchmark
  (annualized, compiled as a second-order cone) and max CVaR over the last
  500 daily return scenarios (daily-loss units, Rockafellar–Uryasev); both
  apply to the Optimize and Pareto Frontier tabs
- **Pareto Frontier tab** — sweep any two objectives against each other
  (e.g. volatility vs expected return); dominated points shown grey, the
  efficient frontier connected, with your current strategy marked on the chart
- **Backtest tab** — rolling re-optimization (weekly/monthly rebalances,
  trailing estimation window) solved as one parallel batch in Rust; shows the
  equity curve, weight evolution, realized stats, and the parallel speedup

The "Universe & Data" tab shows the asset table, normalized prices, and the
return correlation matrix. The JSON spec sent to Rust is visible in an expander
on the Optimize tab.

## Files

| File | Purpose |
|------|---------|
| `fetch_data.py` | Downloads prices, generates random ESG scores, writes CSVs |
| `app.py` | Streamlit UI, spec builder, quartz-bindings backend |
| `data/prices.csv` | Daily adjusted close prices (dates × tickers) |
| `data/assets.csv` | Asset tags + random ESG/climate scores |
| `../crates/quartz-python` | PyO3 bindings the app imports (`import quartz`) |
| `../crates/quartz-demo` | Standalone CLI: JSON spec in → JSON solution out (no longer used by the app) |
