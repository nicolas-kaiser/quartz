"""Fetch market data from Yahoo Finance and build the demo universe.

Writes two CSV files into demo/data/:
  - prices.csv  : daily adjusted close prices (dates x tickers)
  - assets.csv  : one row per asset with tags (currency, sector, asset_class)
                  and randomly generated ESG / climate scores

The ESG and climate scores are RANDOM (seeded for reproducibility) — they are
demo placeholders, not real sustainability data.

Usage:
    python fetch_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
PERIOD = "2y"
SEED = 42

# Ticker -> (currency, sector). Kept static to avoid slow/flaky yf .info calls.
UNIVERSE = {
    "AAPL": ("USD", "Technology"),
    "MSFT": ("USD", "Technology"),
    "JPM": ("USD", "Financials"),
    "XOM": ("USD", "Energy"),
    "JNJ": ("USD", "Healthcare"),
    "NESN.SW": ("CHF", "Consumer Staples"),
    "NOVN.SW": ("CHF", "Healthcare"),
    "MC.PA": ("EUR", "Consumer Discretionary"),
    "SAP.DE": ("EUR", "Technology"),
    "AIR.PA": ("EUR", "Industrials"),
}


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(tickers, period=PERIOD, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices[tickers].dropna(how="all").ffill().dropna()
    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no usable price data")
    return prices


def synthetic_prices(tickers: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Fallback when Yahoo Finance is unreachable: seeded GBM paths."""
    n_days = 504
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    mu = rng.uniform(0.0, 0.12, len(tickers)) / 252
    sigma = rng.uniform(0.10, 0.35, len(tickers)) / np.sqrt(252)
    shocks = rng.standard_normal((n_days, len(tickers)))
    log_paths = np.cumsum(mu - 0.5 * sigma**2 + sigma * shocks, axis=0)
    return pd.DataFrame(100.0 * np.exp(log_paths), index=dates, columns=tickers)


def build_assets(tickers: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """One row per asset: tags + random ESG/climate scores (demo placeholders)."""
    rows = []
    for t in tickers:
        currency, sector = UNIVERSE[t]
        rows.append(
            {
                "ticker": t,
                "currency": currency,
                "sector": sector,
                "asset_class": "Equity",
                # Random scores on a 0-10 scale (higher = better for ESG,
                # higher = worse for risk scores).
                "environmental_impact": round(rng.uniform(2.0, 9.5), 2),
                "social_score": round(rng.uniform(2.0, 9.5), 2),
                "governance_score": round(rng.uniform(3.0, 9.5), 2),
                "physical_risk": round(rng.uniform(0.5, 5.0), 2),
                "transition_risk": round(rng.uniform(0.5, 5.0), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tickers = list(UNIVERSE)
    rng = np.random.default_rng(SEED)

    try:
        prices = fetch_prices(tickers)
        source = "Yahoo Finance"
    except Exception as e:  # network down, rate limit, delisted ticker...
        print(f"WARNING: Yahoo Finance fetch failed ({e}); generating synthetic prices")
        prices = synthetic_prices(tickers, rng)
        source = "synthetic (Yahoo Finance unavailable)"

    assets = build_assets(tickers, rng)

    prices.to_csv(DATA_DIR / "prices.csv", index_label="date")
    assets.to_csv(DATA_DIR / "assets.csv", index=False)

    print(f"Price source: {source}")
    print(f"Wrote {DATA_DIR / 'prices.csv'} ({len(prices)} rows, {len(prices.columns)} tickers)")
    print(f"Wrote {DATA_DIR / 'assets.csv'} ({len(assets)} assets)")


if __name__ == "__main__":
    main()
