"""Streamlit demo for Quartz — multi-dimensional portfolio optimizer.

Loads market data prepared by fetch_data.py (Yahoo Finance prices + random
ESG/climate scores), lets you configure a Strategy and Restrictions
interactively, and solves via the `quartz-demo` Rust binary (JSON over
stdin/stdout).

Run from the repo root:
    cargo build --release -p quartz-demo
    python demo/fetch_data.py
    streamlit run demo/app.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
TRADING_DAYS = 252

SCORE_COLUMNS = [
    "environmental_impact",
    "social_score",
    "governance_score",
    "physical_risk",
    "transition_risk",
]

st.set_page_config(page_title="Quartz Demo", page_icon="🔷", layout="wide")


# ---------------------------------------------------------------- data layer
def find_binary() -> Path | None:
    for profile in ("release", "debug"):
        p = REPO_ROOT / "target" / profile / "quartz-demo"
        if p.exists():
            return p
    return None


@st.cache_data
def load_data():
    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col="date", parse_dates=True)
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    returns = prices.pct_change().dropna()
    cov = returns.cov() * TRADING_DAYS  # annualized covariance
    exp_ret = returns.mean() * TRADING_DAYS  # annualized expected return
    return prices, assets, cov, exp_ret


def solve(spec: dict, binary: Path) -> dict:
    proc = subprocess.run(
        [str(binary)],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": proc.stderr or proc.stdout or "no output from solver"}


# ---------------------------------------------------------------- page setup
binary = find_binary()
if binary is None:
    st.error(
        "The `quartz-demo` binary was not found. Build it first:\n\n"
        "```\ncargo build --release -p quartz-demo\n```"
    )
    st.stop()

if not (DATA_DIR / "prices.csv").exists():
    st.error(
        "No market data found. Fetch it first:\n\n"
        "```\npython demo/fetch_data.py\n```"
    )
    st.stop()

prices, assets, cov, exp_ret = load_data()
tickers = list(prices.columns)

st.title("🔷 Quartz — Multi-Dimensional Portfolio Optimizer")
st.caption(
    f"{len(tickers)} assets · Yahoo Finance prices ({prices.index[0]:%Y-%m-%d} → "
    f"{prices.index[-1]:%Y-%m-%d}) · ESG/climate scores are random demo data · "
    "solved in Rust via Clarabel"
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Strategy")

st.sidebar.subheader("Objective weights")
st.sidebar.caption("Quartz normalizes the weights to sum to 1.")
w_risk = st.sidebar.slider("Minimize financial risk (variance)", 0.0, 1.0, 0.40, 0.05)
w_ret = st.sidebar.slider("Maximize expected return", 0.0, 1.0, 0.30, 0.05)
w_env = st.sidebar.slider("Maximize environmental impact", 0.0, 1.0, 0.15, 0.05)
w_soc = st.sidebar.slider("Maximize social score", 0.0, 1.0, 0.0, 0.05)
w_gov = st.sidebar.slider("Maximize governance score", 0.0, 1.0, 0.0, 0.05)
w_trans = st.sidebar.slider("Minimize transition risk", 0.0, 1.0, 0.15, 0.05)
w_phys = st.sidebar.slider("Minimize physical risk", 0.0, 1.0, 0.0, 0.05)

st.sidebar.subheader("Group constraints")
group_constraints = []
for tag_key in ("currency", "sector"):
    values = sorted(assets[tag_key].unique())
    with st.sidebar.expander(f"By {tag_key}"):
        for val in values:
            lo, hi = st.slider(val, 0.0, 1.0, (0.0, 1.0), 0.05, key=f"grp_{tag_key}_{val}")
            if (lo, hi) != (0.0, 1.0):
                group_constraints.append(
                    {"tag_key": tag_key, "tag_value": val, "lower": lo, "upper": hi}
                )

st.sidebar.subheader("Portfolio score bounds")
score_bounds = []
min_env = st.sidebar.slider("Min environmental impact (0 = off)", 0.0, 10.0, 0.0, 0.5)
if min_env > 0:
    score_bounds.append({"score_key": "environmental_impact", "bound": "min", "threshold": min_env})
max_trans = st.sidebar.slider("Max transition risk (10 = off)", 0.0, 10.0, 10.0, 0.5)
if max_trans < 10:
    score_bounds.append({"score_key": "transition_risk", "bound": "max", "threshold": max_trans})

st.sidebar.header("Restrictions")
long_only = st.sidebar.checkbox("Long only", value=True)
max_weight = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 0.30, 0.05)
excluded_assets = st.sidebar.multiselect("Exclude assets", tickers)
excluded_sectors = st.sidebar.multiselect("Exclude sectors", sorted(assets["sector"].unique()))

# ---------------------------------------------------------------- build spec
asset_specs = []
for _, row in assets.iterrows():
    scores = {col: float(row[col]) for col in SCORE_COLUMNS}
    scores["expected_return"] = float(exp_ret[row["ticker"]])
    asset_specs.append(
        {
            "id": row["ticker"],
            "tags": {
                "currency": row["currency"],
                "sector": row["sector"],
                "asset_class": row["asset_class"],
            },
            "scores": scores,
        }
    )

dimensions = []
if w_risk > 0:
    dimensions.append({"kind": "risk", "sense": "minimize", "weight": w_risk})
for key, weight, sense in [
    ("expected_return", w_ret, "maximize"),
    ("environmental_impact", w_env, "maximize"),
    ("social_score", w_soc, "maximize"),
    ("governance_score", w_gov, "maximize"),
    ("transition_risk", w_trans, "minimize"),
    ("physical_risk", w_phys, "minimize"),
]:
    if weight > 0:
        dimensions.append({"kind": "linear", "score_key": key, "sense": sense, "weight": weight})

spec = {
    "assets": asset_specs,
    "covariance": cov.loc[tickers, tickers].values.tolist(),
    "strategy": {
        "name": "Streamlit Demo",
        "dimensions": dimensions,
        "groups": group_constraints,
        "score_bounds": score_bounds,
    },
    "restrictions": {
        "long_only": long_only,
        "fully_invested": True,
        "max_single_weight": max_weight,
        "exclude_assets": excluded_assets,
        "exclude_tags": [["sector", s] for s in excluded_sectors],
    },
}

# ---------------------------------------------------------------- tabs
tab_opt, tab_data = st.tabs(["Optimize", "Universe & Data"])

with tab_opt:
    if not dimensions:
        st.warning("Set at least one objective weight above zero.")
        st.stop()

    result = solve(spec, binary)

    if "error" in result:
        st.error(f"Solver error: {result['error']}")
        st.stop()

    status = result["status"]
    if status == "Optimal":
        st.success(f"Status: **{status}**")
    elif status == "Infeasible":
        st.error(
            "Status: **Infeasible** — the constraints contradict each other. "
            "Loosen group bounds, score bounds, or the max weight."
        )
        st.stop()
    else:
        st.warning(f"Status: **{status}**")

    weights = pd.DataFrame(result["weights"]).set_index("id")["weight"]
    weights = weights[weights.abs() > 1e-6]
    port = pd.DataFrame({"weight": weights}).join(assets.set_index("ticker"))

    c1, c2, c3, c4 = st.columns(4)
    scores = result["portfolio_scores"]
    variance = scores.get("financial_risk", 0.0)
    c1.metric("Expected return", f"{scores.get('expected_return', 0) * 100:.2f}%")
    c2.metric("Volatility (ann.)", f"{np.sqrt(max(variance, 0)) * 100:.2f}%")
    c3.metric("Solve time", f"{result['solve_time_s'] * 1000:.2f} ms")
    c4.metric("Iterations", result["iterations"])

    col_w, col_pie = st.columns([3, 2])
    with col_w:
        fig = px.bar(
            port.reset_index().sort_values("weight", ascending=False),
            x="id",
            y="weight",
            color="sector",
            labels={"id": "", "weight": "Weight"},
            title="Optimal weights",
        )
        fig.update_layout(yaxis_tickformat=".0%", legend_title="")
        st.plotly_chart(fig, width="stretch")
    with col_pie:
        by_ccy = port.groupby("currency")["weight"].sum().reset_index()
        fig = px.pie(by_ccy, values="weight", names="currency", hole=0.45, title="Currency allocation")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Portfolio scores")
    score_rows = [
        {"dimension": k, "portfolio value": v}
        for k, v in sorted(scores.items())
        if k != "financial_risk"
    ]
    score_rows.append({"dimension": "variance (wᵀΣw)", "portfolio value": variance})
    st.dataframe(pd.DataFrame(score_rows), hide_index=True, width="stretch")

    with st.expander("Holdings detail"):
        detail = port.copy()
        detail["weight"] = (detail["weight"] * 100).round(2)
        st.dataframe(detail, width="stretch")

    with st.expander("Problem spec sent to Rust (JSON)"):
        st.json(spec)

with tab_data:
    st.subheader("Universe")
    universe_view = assets.set_index("ticker").join(
        (exp_ret * 100).round(2).rename("expected_return_%")
    )
    st.dataframe(universe_view, width="stretch")
    st.caption(
        "Expected returns and covariance are estimated from Yahoo Finance daily "
        "prices (annualized). ESG and climate scores are randomly generated demo values."
    )

    st.subheader("Normalized prices (base 100)")
    norm = prices / prices.iloc[0] * 100
    fig = px.line(norm, labels={"value": "Price (base 100)", "date": "", "variable": ""})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Return correlation")
    corr = prices.pct_change().dropna().corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            texttemplate="%{z:.2f}",
        )
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")
