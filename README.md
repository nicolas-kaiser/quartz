<p align="center">
  <img src="https://img.shields.io/badge/rust-stable-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License">
  <img src="https://img.shields.io/badge/solvers-Clarabel%20%7C%20OSQP-green" alt="Solvers">
</p>

# 🔷 Quartz

**Multi-dimensional portfolio optimizer for Rust — with Python bindings and an interactive demo.**

Quartz optimizes portfolios across **any number of dimensions** — not just risk and return. Physical risk, transition risk, environmental impact, social score, governance: define your own dimensions, constrain by currency, sector, tracking error, or CVaR, and let the solver do the rest.

---

## Why Quartz?

Traditional portfolio optimizers (Markowitz) work in 2D: risk vs. return. Real-world asset management balances **many more dimensions** simultaneously:

| Dimension | Example |
|-----------|---------|
| Financial risk | Portfolio variance (σ²) |
| Expected return | μᵀw |
| Physical risk | Exposure to natural catastrophes |
| Transition risk | Climate transition exposure |
| Environmental / social / governance | ESG scores |
| _...your own_ | Any numerical score per asset |

Quartz is a **modeler + compiler**: it translates a high-level portfolio strategy into a conic program (QP/SOCP) and delegates solving to [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) — a state-of-the-art interior-point solver in pure Rust — or optionally to [OSQP](https://osqp.org) for warm-started sequential re-solving.

## Features

**Modeling**
- **N-dimensional objectives** — minimize variance, maximize return, minimize climate exposure, maximize ESG, all at once with configurable weights
- **Group constraints** — currency buckets (USD 10–20%, EUR 40–60%), sector limits, asset-class allocation
- **Score constraints** — portfolio-average environmental score ≥ 6.0, transition risk ≤ 3.0
- **Tracking error limit** — `‖w − w_b‖_Σ ≤ TE`, compiled to a second-order cone (works with both covariance models)
- **CVaR limit** — scenario-based expected tail loss via the Rockafellar–Uryasev linearization
- **Turnover control** — cap `Σ|wᵢ − wᵢ_prev|` against previous weights
- **Exclusion lists** — by tag (sector = Tobacco) or by specific asset
- **Strategy / Tactic / Restriction** — three-layer separation of long-term policy, short-term tilts, and hard compliance
- **Strategy files** — load/save strategies, tactics, and restrictions as hand-writable YAML or JSON

**Risk models**
- **Full covariance** — dense n×n Σ
- **Factor models** — `Σ = BFBᵀ + D` compiles to a sparse QP with k auxiliary variables: O(nk²) instead of O(n²)

**Analysis & scale**
- **Pareto frontier exploration** — sweep two objectives against each other, or enumerate a full simplex grid, with non-dominated filtering
- **Parallel batch solving** — `solve_batch` runs independent problems on all cores via `rayon` (1000-date backtest in ~13 ms)
- **OSQP warm-start backend** (optional) — sequential re-solves (turnover-chained backtests, live re-optimization) seed each solve with the previous solution

**Interfaces**
- **Rust API** — builder-pattern, ~0.1–1 ms per solve at typical sizes
- **Python bindings** — `import quartz` (PyO3 + maturin, abi3 wheel); batch solves release the GIL and run rayon-parallel
- **Streamlit demo** — interactive app with real market data: optimize, explore frontiers, backtest
- **Pure Rust by default** — no C/C++ toolchain needed unless you opt into the OSQP feature

## Quick start (Rust)

```toml
[dependencies]
quartz-portfolio = { path = "crates/quartz-portfolio" }
quartz-core = { path = "crates/quartz-core" }
clarabel = "0.11"
```

```rust
use clarabel::algebra::CscMatrix;
use quartz_core::Asset;
use quartz_portfolio::{PortfolioModel, Restrictions, Strategy};

// Define your investment universe
let universe = quartz_core::Universe::builder()
    .add_asset(Asset::new("AAPL")
        .tag("currency", "USD").tag("sector", "Technology")
        .score("expected_return", 0.10)
        .score("environmental_impact", 6.5)
        .score("transition_risk", 3.0))
    .add_asset(Asset::new("NESN")
        .tag("currency", "CHF").tag("sector", "Consumer")
        .score("expected_return", 0.04)
        .score("environmental_impact", 8.0)
        .score("transition_risk", 1.5))
    .add_asset(Asset::new("GREEN_BOND")
        .tag("currency", "EUR").tag("asset_class", "Bond")
        .score("expected_return", 0.025)
        .score("environmental_impact", 9.5)
        .score("transition_risk", 0.5))
    .covariance_full(CscMatrix::from(&[
        [0.04,  0.002, 0.001],
        [0.002, 0.01,  0.001],
        [0.001, 0.001, 0.003],
    ]))
    .build()?;

// A multi-dimensional strategy with risk limits
let strategy = Strategy::builder("ESG Balanced")
    .minimize_risk(0.40)                          // 40% weight on variance
    .maximize("expected_return", 0.25)            // 25% on return
    .minimize("transition_risk", 0.15)            // 15% on climate risk
    .maximize("environmental_impact", 0.20)       // 20% on ESG
    .group("currency", "EUR", 0.30, 0.60)         // 30-60% in EUR
    .score_min("environmental_impact", 7.0)       // portfolio avg ESG ≥ 7.0
    .max_tracking_error(vec![0.34, 0.33, 0.33], 0.05)  // ≤5% TE vs benchmark
    .build();

// Hard compliance restrictions
let restrictions = Restrictions::builder()
    .long_only()
    .fully_invested()
    .max_single_weight(0.50)
    .build();

let solution = PortfolioModel::new(&universe)
    .strategy(&strategy)
    .restrictions(restrictions)
    .solve()?;

println!("Status: {:?}", solution.status);
for (id, w) in &solution.weights {
    println!("  {id}: {:.2}%", w * 100.0);
}
// portfolio_scores includes every score dimension plus financial_risk,
// tracking_error, and cvar when the corresponding constraints are active
```

### Going further

```rust
// Factor covariance: Σ = BFBᵀ + D — O(nk²) compilation
let universe = Universe::builder()
    .assets(assets)
    .covariance_factor(loadings, factor_cov, specific_variance)
    .build()?;

// CVaR: needs return scenarios on the universe
let universe = Universe::builder()
    .assets(assets)
    .covariance_full(sigma)
    .scenarios(daily_returns)               // S rows × n assets
    .build()?;
let strategy = Strategy::builder("Tail-aware")
    .maximize("expected_return", 1.0)
    .max_cvar(0.95, 0.02)                   // worst-5% expected loss ≤ 2%
    .build();

// Pareto frontier: trade return against risk over 25 points
let frontier = FrontierExplorer::new(&universe, &strategy)
    .restrictions(restrictions)
    .sweep("expected_return", "financial_risk", 25)?;
for p in frontier.points.iter().filter(|p| p.is_efficient) { /* plot */ }

// Parallel backtest: one problem per date, solved on all cores
let problems: Vec<BatchProblem> = universes.iter()
    .map(|u| BatchProblem::new(u, &strategy).restrictions(restrictions.clone()))
    .collect();
let results = solve_batch(&problems, &SolverSettings::default());

// Sequential warm-started re-solving (OSQP backend, `osqp` feature)
let sol_t = PortfolioModel::new(&universe_t)
    .strategy(&strategy)
    .turnover(previous_weights, 0.10)
    .backend(Backend::Osqp)
    .solver_settings(SolverSettings::default_for(Backend::Osqp))
    .warm_start(&sol_t_minus_1)
    .solve()?;
```

## Strategy files

Strategies are loadable from hand-written YAML or JSON — see
[`crates/quartz-portfolio/examples/strategies/esg_balanced.yaml`](crates/quartz-portfolio/examples/strategies/esg_balanced.yaml):

```yaml
name: ESG Balanced
dimensions:
- name: financial_risk
  type: quadratic
  sense: minimize
  weight: 40            # relative weights fine — normalized on load
- name: expected_return
  type: linear
  score_key: expected_return
  sense: maximize
  weight: 60
score_constraints:
- score_key: environmental_impact
  bound: !min 7.0       # or !max 3.0 / !range [4.0, 7.0]
tracking_error: {benchmark_weights: [0.5, 0.5, 0.0], max_te: 0.05}
cvar: {alpha: 0.95, max_cvar: 0.02}
```

```rust
let strategy = Strategy::load("esg_balanced.yaml")?;   // .json / .yaml / .yml
strategy.save("backup.json")?;
```

`Tactic` and `Restrictions` have the same `load`/`save`/`from_yaml_str`/... API.

## Python

```sh
pip install maturin
maturin build --release -m crates/quartz-python/Cargo.toml
pip install target/wheels/quartz-*.whl
```

```python
import quartz

u = quartz.Universe(
    assets=[
        quartz.Asset("AAPL", tags={"currency": "USD"},
                     scores={"expected_return": 0.10, "environmental_impact": 6.5}),
        quartz.Asset("NESN", tags={"currency": "CHF"},
                     scores={"expected_return": 0.04, "environmental_impact": 8.0}),
    ],
    covariance=[[0.04, 0.002], [0.002, 0.01]],   # numpy arrays work too
    scenarios=daily_returns,                      # optional, for CVaR
)
s = (quartz.Strategy("ESG")
     .minimize_risk(0.5)
     .maximize("expected_return", 0.3)
     .maximize("environmental_impact", 0.2)
     .max_cvar(0.95, 0.02))
r = quartz.Restrictions(long_only=True, fully_invested=True, max_single_weight=0.6)

sol = quartz.solve(u, s, restrictions=r)
print(sol.status, sol.weights, sol.portfolio_scores)

# Batch (GIL released → rayon-parallel), Pareto frontier, strategy files:
sols = quartz.solve_batch([(u, s)] * 100, restrictions=r)
frontier = quartz.sweep(u, s, "expected_return", "financial_risk", n_points=25, restrictions=r)
s2 = quartz.Strategy.from_file("esg_balanced.yaml")
```

Errors raise `quartz.QuartzError` (a `ValueError` subclass); infeasibility is a
status on the solution, not an exception.

## Interactive demo

A Streamlit app in [`demo/`](demo/) drives the whole feature set on real Yahoo
Finance data (ESG scores are random placeholders):

- **Optimize** — sidebar strategy/restriction controls, sample or k-factor PCA
  risk model, tracking-error and CVaR limits, strategy-YAML export
- **Pareto Frontier** — sweep any two objectives; dominated points grey,
  efficient frontier connected, current strategy marked
- **Backtest** — rolling re-optimization solved as one parallel batch, with
  equity curve, weight evolution, and the measured parallel speedup

```sh
pip install maturin && maturin build --release -m crates/quartz-python/Cargo.toml
pip install target/wheels/quartz-*.whl -r demo/requirements.txt
python demo/fetch_data.py          # Yahoo Finance → demo/data/*.csv
streamlit run demo/app.py
```

See [`demo/README.md`](demo/README.md) for details.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Strategy + Tactic + Restrictions                    │  You define this
│  "Minimize risk, maximize ESG, EUR 30-60%, TE ≤ 5%"  │  (code or YAML)
├──────────────────────────────────────────────────────┤
│  Compiler                          (quartz-portfolio)│  Quartz translates
│  Builds P, q, A, b + cone list from your constraints │
├──────────────────────────────────────────────────────┤
│  Solver                              (quartz-solver) │  Clarabel (or OSQP)
│  min ½xᵀPx + qᵀx  s.t. Ax + s = b, s ∈ K            │
└──────────────────────────────────────────────────────┘
```

### Crate structure

| Crate | Purpose |
|-------|---------|
| **quartz-core** | Data types: `Asset`, `Universe`, `Dimension`, covariance models, scenarios |
| **quartz-solver** | Solver backends: Clarabel (default) and OSQP (`osqp` feature, warm starts) |
| **quartz-portfolio** | Everything else: constraints, compiler, `PortfolioModel`, frontier, batch, strategy files |
| **quartz-python** | PyO3 bindings (`import quartz`), built with maturin |
| **quartz-demo** | Standalone JSON stdin/stdout CLI for scripting without Rust or Python bindings |

### Three-layer constraint model

| Layer | Purpose | Example |
|-------|---------|---------|
| **Strategy** | Long-term investment policy: dimension weights, allocation targets, risk limits | "40% risk, 25% return, EUR 40-60%, TE ≤ 5%" |
| **Tactic** | Short-term overlay that tightens strategy bounds | "Q2 2026: EUR 50-65%, risk weight to 55%" |
| **Restriction** | Hard compliance constraints, non-negotiable | "No tobacco, no shorting, max 5% per name" |

Tactics merge with strategies by **interval intersection**: bounds are tightened, never loosened. An empty intersection errors before solving.

### Solver backends

| | Clarabel (default) | OSQP (`osqp` feature) |
|---|---|---|
| Algorithm | Interior-point | ADMM (first-order) |
| Accuracy | ~1e-8 | ~1e-6 (with polishing) |
| Cones | QP + SOC (tracking error) | QP only |
| Warm starts | No (algorithmic limitation) | Yes — big win for sequential re-solves |
| Dependencies | Pure Rust | Vendored C (needs cmake + a C compiler) |

## Examples

```sh
cargo run --example markowitz -p quartz-portfolio            # min-variance, 3 assets
cargo run --example multi_dimension -p quartz-portfolio      # 5 assets × 5 dimensions
cargo run --example frontier -p quartz-portfolio             # Pareto sweep table
cargo run --example strategy_file -p quartz-portfolio        # load & solve a YAML strategy
cargo run --release --example backtest -p quartz-portfolio   # 1000 dates in parallel
cargo run --release --example backtest_warmstart -p quartz-portfolio --features osqp
                                                             # chained backtest: Clarabel vs OSQP warm
```

## Building & testing

```sh
cargo build                                            # all crates, pure Rust
cargo test                                             # full test suite
cargo test -p quartz-portfolio --no-default-features   # serial fallback (no rayon)
cargo test -p quartz-solver --features osqp            # OSQP backend (needs cmake + cc)

# Python bindings
python3 -m venv .venv && .venv/bin/pip install maturin pytest numpy
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m crates/quartz-python/Cargo.toml
.venv/bin/pytest crates/quartz-python/tests -q
```

Requires **Rust stable** (edition 2021). No C/C++ toolchain needed for the default build; only the optional `osqp` feature compiles vendored C.

## Roadmap

All original roadmap items are shipped:

- [x] Factor covariance model support (`Σ = BFBᵀ + D`) for O(nk²) scaling
- [x] Pareto frontier exploration (multi-objective trade-off visualization)
- [x] Parallel batch solving with `rayon` (backtest 1000 dates in parallel)
- [x] Python bindings via PyO3 + maturin
- [x] OSQP backend for warm-start support
- [x] SOCP support for CVaR and tracking error constraints
- [x] JSON/YAML strategy file loading

Possible future directions: OSQP workspace reuse (`update_*` APIs) for another
sequential-solve multiplier, tactic overrides for risk limits (min-tightening),
per-date scenarios in the demo backtest, cardinality constraints (MIP).

## License

Quartz is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

This means you are free to use, modify, and distribute Quartz, but any modified version — including use over a network (SaaS) — **must also be released under AGPL-3.0** with source code available.

See [LICENSE](LICENSE) for the full text.
