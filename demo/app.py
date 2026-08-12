"""Streamlit demo for Quartz — multi-dimensional portfolio optimizer.

Loads market data prepared by fetch_data.py (Yahoo Finance prices + random
ESG/climate scores), lets you configure a Strategy and Restrictions
interactively, and solves in-process through the `quartz` Python bindings
(PyO3; batch solves run rayon-parallel with the GIL released).

Run from the repo root:
    pip install maturin
    maturin build --release -m crates/quartz-python/Cargo.toml
    pip install target/wheels/quartz-*.whl
    python demo/fetch_data.py
    streamlit run demo/app.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import quartz
except ImportError:
    quartz = None

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
@st.cache_data
def load_data():
    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col="date", parse_dates=True)
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    returns = prices.pct_change().dropna()
    cov = returns.cov() * TRADING_DAYS  # annualized covariance
    exp_ret = returns.mean() * TRADING_DAYS  # annualized expected return
    return prices, assets, cov, exp_ret, returns


def pca_factor_model(cov: pd.DataFrame, k: int):
    """Top-k PCA factor model estimated from the sample covariance.

    Eigendecompose Σ, keep the k largest components: B = Q_k, F = diag(λ_k),
    D = diag(Σ − BFBᵀ) clipped to ≥ 0 so the residual variance stays valid.
    """
    eigvals, eigvecs = np.linalg.eigh(cov.values)  # ascending order
    top = np.argsort(eigvals)[::-1][:k]
    loadings = eigvecs[:, top]
    factor_var = eigvals[top]
    common = (loadings * factor_var) @ loadings.T
    specific = np.clip(np.diag(cov.values) - np.diag(common), 1e-10, None)
    return loadings, np.diag(factor_var), specific


# ------------------------------------------------ quartz bindings interpreter
# The app keeps building a JSON-like spec dict (it drives the UI and the
# "spec JSON" expander); these helpers interpret it with the quartz bindings
# and return the same dict shapes the old subprocess bridge produced.
def _build_universe(asset_specs, covariance=None, factor_model=None, score_overrides=None,
                    scenarios=None):
    assets = []
    for a in asset_specs:
        scores = dict(a["scores"])
        if score_overrides and a["id"] in score_overrides:
            scores.update(score_overrides[a["id"]])
        assets.append(quartz.Asset(a["id"], tags=a.get("tags"), scores=scores))
    if factor_model is not None:
        fm = (
            factor_model["loadings"],
            factor_model["factor_cov"],
            factor_model["specific_variance"],
        )
        return quartz.Universe(assets=assets, factor_model=fm, scenarios=scenarios)
    return quartz.Universe(assets=assets, covariance=covariance, scenarios=scenarios)


def _build_strategy(sspec: dict):
    s = quartz.Strategy(sspec["name"])
    for d in sspec["dimensions"]:
        if d["kind"] == "risk":
            s = s.minimize_risk(d["weight"])
        elif d["sense"] == "maximize":
            s = s.maximize(d["score_key"], d["weight"])
        else:
            s = s.minimize(d["score_key"], d["weight"])
    for g in sspec.get("groups", []):
        s = s.group(g["tag_key"], g["tag_value"], g["lower"], g["upper"])
    for b in sspec.get("score_bounds", []):
        if b["bound"] == "min":
            s = s.score_min(b["score_key"], b["threshold"])
        else:
            s = s.score_max(b["score_key"], b["threshold"])
    if "tracking_error" in sspec:
        te = sspec["tracking_error"]
        s = s.max_tracking_error(te["benchmark"], te["max_te"])
    if "cvar" in sspec:
        s = s.max_cvar(sspec["cvar"]["alpha"], sspec["cvar"]["max_cvar"])
    # Fully-invested is controlled via restrictions in the demo spec.
    return s.fully_invested(False)


def _build_restrictions(rspec: dict):
    return quartz.Restrictions(
        long_only=rspec.get("long_only", False),
        fully_invested=rspec.get("fully_invested", False),
        max_single_weight=rspec.get("max_single_weight"),
        exclude_assets=rspec.get("exclude_assets", []),
        exclude_tags=[tuple(t) for t in rspec.get("exclude_tags", [])],
    )


def _status_name(status) -> str:
    return str(status).split(".")[-1]


def _solution_dict(sol) -> dict:
    return {
        "status": _status_name(sol.status),
        "weights": [{"id": k, "weight": v} for k, v in sol.weights.items()],
        "portfolio_scores": dict(sol.portfolio_scores),
        "objective_value": sol.objective_value,
        "solve_time_s": sol.solve_time_s,
        "iterations": sol.iterations,
    }


def solve(spec: dict) -> dict:
    try:
        strategy = _build_strategy(spec["strategy"])
        restrictions = _build_restrictions(spec["restrictions"])

        if "batch" in spec:
            problems, build_errors = [], {}
            for i, item in enumerate(spec["batch"]["items"]):
                cov = item.get("covariance")
                fm = item.get("factor_model")
                if cov is None and fm is None:
                    cov, fm = spec.get("covariance"), spec.get("factor_model")
                scen = item.get("scenarios", spec.get("scenarios"))
                try:
                    u = _build_universe(spec["assets"], cov, fm, item.get("scores"), scen)
                    problems.append(quartz.Problem(u, strategy, restrictions=restrictions))
                except quartz.QuartzError as e:
                    build_errors[i] = str(e)
                    problems.append(None)
            live = [p for p in problems if p is not None]
            t0 = time.perf_counter()
            results = iter(quartz.solve_batch(live))
            wall_time_s = time.perf_counter() - t0
            solutions, sum_solve_time_s = [], 0.0
            for i, p in enumerate(problems):
                if p is None:
                    solutions.append({"error": build_errors[i]})
                    continue
                r = next(results)
                if isinstance(r, quartz.QuartzError):
                    solutions.append({"error": str(r)})
                else:
                    sum_solve_time_s += r.solve_time_s
                    solutions.append(_solution_dict(r))
            return {
                "solutions": solutions,
                "n_items": len(solutions),
                "wall_time_s": wall_time_s,
                "sum_solve_time_s": sum_solve_time_s,
            }

        universe = _build_universe(
            spec["assets"], spec.get("covariance"), spec.get("factor_model"),
            scenarios=spec.get("scenarios"),
        )

        if "frontier" in spec:
            fs = spec["frontier"]
            if fs["mode"] == "sweep":
                fr = quartz.sweep(
                    universe, strategy, fs["dim_a"], fs["dim_b"],
                    n_points=fs.get("n_points", 25), restrictions=restrictions,
                )
            else:
                fr = quartz.simplex_grid(
                    universe, strategy,
                    resolution=fs.get("resolution", 5), restrictions=restrictions,
                )
            return {
                "points": [
                    {
                        "dimension_weights": list(p.dimension_weights.items()),
                        "weights": list(p.weights.items()),
                        "weights_vec": p.weights_vec,
                        "portfolio_scores": dict(p.portfolio_scores),
                        "objective_value": p.objective_value,
                        "is_efficient": p.is_efficient,
                    }
                    for p in fr.points
                ],
                "objective_dims": fr.objective_dims,
                "n_skipped": fr.n_skipped,
            }

        sol = quartz.solve(universe, strategy, restrictions=restrictions)
        return _solution_dict(sol)
    except (quartz.QuartzError, ValueError, KeyError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- page setup
if quartz is None:
    st.error(
        "The `quartz` Python module is not installed. Build it first:\n\n"
        "```\npip install maturin\n"
        "maturin build --release -m crates/quartz-python/Cargo.toml\n"
        "pip install target/wheels/quartz-*.whl\n```"
    )
    st.stop()

if not (DATA_DIR / "prices.csv").exists():
    st.error(
        "No market data found. Fetch it first:\n\n"
        "```\npython demo/fetch_data.py\n```"
    )
    st.stop()

prices, assets, cov, exp_ret, returns = load_data()
tickers = list(prices.columns)

st.title("🔷 Quartz — Multi-Dimensional Portfolio Optimizer")
st.caption(
    f"{len(tickers)} assets · Yahoo Finance prices ({prices.index[0]:%Y-%m-%d} → "
    f"{prices.index[-1]:%Y-%m-%d}) · ESG/climate scores are random demo data · "
    "solved in Rust via Clarabel"
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Risk model")
cov_choice = st.sidebar.radio(
    "Covariance model",
    ["Sample (full n×n)", "k-factor (PCA)"],
    help="The factor model compiles to a sparse QP with k auxiliary variables "
    "(y = Bᵀw), scaling O(nk²) instead of O(n²).",
)
use_factor = cov_choice.startswith("k-factor")
n_factors = 0
if use_factor:
    n_factors = st.sidebar.slider("Number of factors k", 1, min(5, len(tickers) - 1), 3)

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

st.sidebar.header("Risk constraints")
use_te = st.sidebar.checkbox("Max tracking error vs equal-weight benchmark")
max_te = 0.05
if use_te:
    max_te = st.sidebar.slider("Max TE (annualized)", 0.01, 0.25, 0.05, 0.01)
use_cvar = st.sidebar.checkbox("Max CVaR (scenario-based)")
cvar_alpha, max_cvar = 0.95, 0.02
if use_cvar:
    cvar_alpha = st.sidebar.selectbox("CVaR confidence α", [0.90, 0.95, 0.99], index=1)
    max_cvar = st.sidebar.slider("Max CVaR (daily loss)", 0.005, 0.05, 0.02, 0.005)

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

if use_te:
    equal_weight = [1.0 / len(tickers)] * len(tickers)
    spec["strategy"]["tracking_error"] = {"benchmark": equal_weight, "max_te": max_te}
if use_cvar:
    spec["strategy"]["cvar"] = {"alpha": cvar_alpha, "max_cvar": max_cvar}
    spec["scenarios"] = returns.tail(500)[tickers].values.tolist()

cov_ordered = cov.loc[tickers, tickers]
if use_factor:
    loadings, factor_cov, specific = pca_factor_model(cov_ordered, n_factors)
    spec["factor_model"] = {
        "loadings": loadings.tolist(),
        "factor_cov": factor_cov.tolist(),
        "specific_variance": specific.tolist(),
    }
    risk_model_label = f"{n_factors}-factor PCA (Σ = BFBᵀ + D)"
else:
    spec["covariance"] = cov_ordered.values.tolist()
    risk_model_label = "sample covariance (full n×n)"

# ---------------------------------------------------------------- tabs
tab_opt, tab_frontier, tab_backtest, tab_data = st.tabs(
    ["Optimize", "Pareto Frontier", "Backtest", "Universe & Data"]
)

with tab_opt:
    if not dimensions:
        st.warning("Set at least one objective weight above zero.")
        st.stop()

    result = solve(spec)

    if "error" in result:
        st.error(f"Solver error: {result['error']}")
        st.stop()

    status = result["status"]
    if status == "Optimal":
        st.success(f"Status: **{status}** · risk model: {risk_model_label}")
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

    if "tracking_error" in scores or "cvar" in scores:
        r1, r2, _, _ = st.columns(4)
        if "tracking_error" in scores:
            r1.metric(
                "Tracking error (ann.)",
                f"{scores['tracking_error'] * 100:.2f}%",
                help=f"vs equal-weight benchmark, limit {max_te * 100:.0f}%",
            )
        if "cvar" in scores:
            r2.metric(
                f"CVaR {cvar_alpha:.0%} (daily)",
                f"{scores['cvar'] * 100:.2f}%",
                help=f"expected loss over the worst {1 - cvar_alpha:.0%} of the last "
                f"{len(spec.get('scenarios', []))} days, limit {max_cvar * 100:.1f}%",
            )

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

with tab_frontier:
    st.caption(
        "Sweep two objectives against each other while all other dimension "
        "weights stay fixed; grey points are dominated, colored points form "
        "the Pareto frontier."
    )
    active_dims = [
        "financial_risk" if d["kind"] == "risk" else d["score_key"] for d in dimensions
    ]
    if len(active_dims) < 2:
        st.warning("Give at least two objectives a nonzero weight to explore a frontier.")
    else:
        c1, c2, c3 = st.columns(3)
        default_x = active_dims.index("financial_risk") if "financial_risk" in active_dims else 0
        default_y = (
            active_dims.index("expected_return")
            if "expected_return" in active_dims
            else (1 if default_x != 1 else 0)
        )
        dim_x = c1.selectbox("X axis (trade off)", active_dims, index=default_x)
        dim_y = c2.selectbox("Y axis (trade off)", active_dims, index=default_y)
        n_points = c3.slider("Points", 5, 100, 25)

        if dim_x == dim_y:
            st.warning("Pick two different dimensions.")
        else:
            frontier_spec = dict(spec)
            frontier_spec["frontier"] = {
                "mode": "sweep",
                "dim_a": dim_y,
                "dim_b": dim_x,
                "n_points": n_points,
            }
            fresult = solve(frontier_spec)

            if "error" in fresult:
                st.error(f"Frontier error: {fresult['error']}")
            else:
                def axis_value(scores: dict, dim: str) -> float:
                    # Volatility is more readable than variance; sqrt is
                    # monotone so efficiency flags are unaffected.
                    if dim == "financial_risk":
                        return float(np.sqrt(max(scores.get(dim, 0.0), 0.0)))
                    return float(scores.get(dim, 0.0))

                def axis_label(dim: str) -> str:
                    return "Volatility (ann.)" if dim == "financial_risk" else dim

                rows = []
                for pt in fresult["points"]:
                    top = sorted(pt["weights"], key=lambda t: -t[1])[:3]
                    rows.append(
                        {
                            "x": axis_value(pt["portfolio_scores"], dim_x),
                            "y": axis_value(pt["portfolio_scores"], dim_y),
                            "efficient": pt["is_efficient"],
                            "top holdings": ", ".join(
                                f"{tid} {tw * 100:.0f}%" for tid, tw in top if tw > 0.005
                            ),
                        }
                    )
                fdf = pd.DataFrame(rows)

                fig = go.Figure()
                dominated = fdf[~fdf["efficient"]]
                efficient = fdf[fdf["efficient"]].sort_values("x")
                if not dominated.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=dominated["x"], y=dominated["y"], mode="markers",
                            marker=dict(color="grey", size=7, opacity=0.5),
                            name="dominated", text=dominated["top holdings"],
                            hovertemplate="%{x:.4f}, %{y:.4f}<br>%{text}<extra>dominated</extra>",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=efficient["x"], y=efficient["y"], mode="lines+markers",
                        marker=dict(size=9), name="Pareto frontier",
                        text=efficient["top holdings"],
                        hovertemplate="%{x:.4f}, %{y:.4f}<br>%{text}<extra>efficient</extra>",
                    )
                )
                # Current sidebar strategy as a reference marker
                current = solve(spec)
                if "error" not in current and current["status"] == "Optimal":
                    fig.add_trace(
                        go.Scatter(
                            x=[axis_value(current["portfolio_scores"], dim_x)],
                            y=[axis_value(current["portfolio_scores"], dim_y)],
                            mode="markers",
                            marker=dict(symbol="star", size=16, color="orange"),
                            name="current strategy",
                        )
                    )
                fig.update_layout(
                    xaxis_title=axis_label(dim_x),
                    yaxis_title=axis_label(dim_y),
                    title=f"{axis_label(dim_y)} vs {axis_label(dim_x)} "
                    f"({len(fdf)} points, {fresult['n_skipped']} infeasible skipped)",
                    height=550,
                )
                st.plotly_chart(fig, width="stretch")

                with st.expander("Frontier points detail"):
                    detail_rows = []
                    for pt in fresult["points"]:
                        row = {name: w for name, w in pt["dimension_weights"]}
                        row.update(
                            {f"→ {k}": v for k, v in sorted(pt["portfolio_scores"].items())}
                        )
                        row["efficient"] = pt["is_efficient"]
                        detail_rows.append(row)
                    st.dataframe(pd.DataFrame(detail_rows), width="stretch")

with tab_backtest:
    st.caption(
        "Rolling backtest: at each rebalance date, covariance and expected "
        "returns are re-estimated from a trailing window and the strategy is "
        "re-solved — all dates in one parallel batch (Rust + rayon). "
        "Simplifications: dates are solved independently (no turnover "
        "chaining) and weights are held constant between rebalances."
    )
    c1, c2 = st.columns(2)
    rebal_freq = c1.radio("Rebalance frequency", ["Weekly", "Monthly"], horizontal=True)
    window = c2.slider("Trailing estimation window (days)", 60, 252, 126)

    returns_bt = prices.pct_change().dropna()
    period = "W" if rebal_freq == "Weekly" else "M"
    all_rebal = [g.index[-1] for _, g in returns_bt.groupby(returns_bt.index.to_period(period))]
    positions = {d: i for i, d in enumerate(returns_bt.index)}
    rebal_dates = [d for d in all_rebal if positions[d] >= window]

    if len(rebal_dates) < 2:
        st.warning(
            f"Not enough history: need at least 2 rebalance dates with "
            f"{window} days of trailing data. Shorten the window."
        )
    else:
        items = []
        for t in rebal_dates:
            win = returns_bt.loc[:t].tail(window)
            win_cov = win.cov() * TRADING_DAYS
            item = {
                "scores": {
                    tkr: {"expected_return": float(r)}
                    for tkr, r in (win.mean() * TRADING_DAYS).items()
                }
            }
            if use_factor:
                loadings, factor_cov_m, specific = pca_factor_model(win_cov, n_factors)
                item["factor_model"] = {
                    "loadings": loadings.tolist(),
                    "factor_cov": factor_cov_m.tolist(),
                    "specific_variance": specific.tolist(),
                }
            else:
                item["covariance"] = win_cov.values.tolist()
            items.append(item)

        # v1: risk constraints (TE/CVaR) apply to the Optimize and Frontier
        # tabs; the backtest solves each date without them.
        batch_spec = {k: v for k, v in spec.items() if k not in ("frontier", "scenarios")}
        batch_spec["strategy"] = {
            k: v for k, v in spec["strategy"].items() if k not in ("tracking_error", "cvar")
        }
        batch_spec["batch"] = {"items": items}
        if use_te or use_cvar:
            st.caption("Note: the TE/CVaR risk constraints apply to the Optimize "
                       "and Pareto Frontier tabs, not to the backtest.")
        bresult = solve(batch_spec)

        if "error" in bresult:
            st.error(f"Backtest error: {bresult['error']}")
        else:
            weights_rows, used_dates, n_bad = [], [], 0
            for t, sol in zip(rebal_dates, bresult["solutions"]):
                if "error" in sol or sol.get("status") != "Optimal":
                    n_bad += 1
                    continue
                weights_rows.append({w["id"]: w["weight"] for w in sol["weights"]})
                used_dates.append(t)

            if not weights_rows:
                st.error("No rebalance date solved to optimality — loosen the constraints.")
            else:
                wall_ms = bresult["wall_time_s"] * 1000
                sum_ms = bresult["sum_solve_time_s"] * 1000
                st.success(
                    f"**{bresult['n_items']} portfolio solves in {wall_ms:.1f} ms wall time** "
                    f"({sum_ms:.1f} ms of solver time ≈ {sum_ms / max(wall_ms, 1e-9):.1f}× parallel)"
                    + (f" · {n_bad} dates skipped (non-optimal)" if n_bad else "")
                )

                W = pd.DataFrame(weights_rows, index=pd.DatetimeIndex(used_dates))[tickers].fillna(0.0)

                # Equity curve: hold weights from each rebalance close to the next
                segments = []
                for k, t in enumerate(W.index):
                    end = W.index[k + 1] if k + 1 < len(W.index) else returns_bt.index[-1]
                    seg = returns_bt.loc[t:end].iloc[1:]
                    if not seg.empty:
                        segments.append(seg @ W.iloc[k])
                port_ret = pd.concat(segments)
                equity = (1 + port_ret).cumprod()

                ann_ret = port_ret.mean() * TRADING_DAYS
                ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
                turnover_per_rebal = W.diff().abs().sum(axis=1).iloc[1:].mean()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ann. return (realized)", f"{ann_ret * 100:.2f}%")
                m2.metric("Ann. volatility", f"{ann_vol * 100:.2f}%")
                m3.metric("Sharpe (rf=0)", f"{ann_ret / ann_vol:.2f}" if ann_vol > 0 else "—")
                m4.metric("Avg turnover / rebalance", f"{turnover_per_rebal * 100:.1f}%")

                fig = px.line(
                    equity, labels={"value": "Growth of 1", "index": "", "variable": ""},
                    title=f"Equity curve — {len(W)} rebalances, {window}d window, "
                    f"risk model: {risk_model_label}",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch")

                fig = px.area(
                    W, labels={"value": "Weight", "index": "", "variable": ""},
                    title="Weight evolution at rebalance dates",
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")

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
