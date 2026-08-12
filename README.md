<p align="center">
  <img src="https://img.shields.io/badge/rust-stable-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License">
  <img src="https://img.shields.io/badge/solver-Clarabel-green" alt="Solver">
</p>

# 🔷 Quartz

**Multi-dimensional portfolio optimizer for Rust.**

Quartz lets you optimize portfolios across **any number of dimensions** — not just risk and return. Physical risk, transition risk, environmental impact, social score, governance — define your own dimensions, set constraints by currency, sector, asset class, and let the solver do the rest.

---

## Why Quartz?

Traditional portfolio optimizers (Markowitz) work in 2D: risk vs. return. Real-world asset management requires balancing **many more dimensions** simultaneously:

| Dimension | Example |
|-----------|---------|
| Financial risk | Portfolio variance (σ²) |
| Expected return | μᵀw |
| Physical risk | Exposure to natural catastrophes |
| Transition risk | Climate transition exposure |
| Environmental impact | ESG environmental score |
| Social impact | ESG social score |
| Governance | ESG governance score |
| _...your own_ | Any numerical score per asset |

Quartz is a **modeler + compiler**: it translates your high-level portfolio strategy into a quadratic program (QP) and delegates solving to [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs), a state-of-the-art conic solver written in pure Rust.

## Features

- **N-dimensional optimization** — minimize risk, maximize return, minimize climate exposure, maximize ESG scores, all at once with configurable weights
- **Group constraints** — currency buckets (USD 10–20%, EUR 40–60%), sector limits, asset class allocation
- **Score constraints** — enforce minimum environmental score ≥ 6.0, maximum transition risk ≤ 3.0
- **Strategy / Tactic / Restriction** — three-layer architecture separating long-term vision, short-term adjustments, and compliance rules
- **Exclusion lists** — exclude by tag (sector = Tobacco) or by specific asset
- **Turnover control** — limit total portfolio rebalancing with warm-start from previous weights
- **Tracking error & CVaR limits** — `‖w − w_b‖_Σ ≤ TE` compiles to a second-order cone; scenario-based CVaR uses the Rockafellar–Uryasev linearization (`.max_tracking_error(...)`, `.max_cvar(...)`)
- **Factor covariance models** — `Σ = BFBᵀ + D` compiles to a sparse QP with k auxiliary variables (O(nk²) instead of O(n²))
- **Pareto frontier exploration** — sweep two dimensions against each other or enumerate a full simplex grid, with non-dominated filtering (`FrontierExplorer`)
- **Parallel batch solving** — `solve_batch` runs independent problems on all cores via `rayon` (1000-date backtest in ~13 ms); opt out with `default-features = false`
- **Python bindings** — `import quartz` (PyO3 + maturin); batch solves release the GIL and run rayon-parallel
- **OSQP warm-start backend** (optional `osqp` feature) — sequential re-solves (turnover-chained backtests, live re-optimization) seed each solve with the previous solution: `.backend(Backend::Osqp).warm_start(&prev)`
- **Pure Rust** — zero C/C++ dependencies, compiles anywhere Rust does
- **~1ms solve time** — for typical 5-asset problems with full constraint sets

## Quick Start

Add to your `Cargo.toml`:

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
    .build()
    .unwrap();

// Define a multi-dimensional strategy
let strategy = Strategy::builder("ESG Balanced")
    .minimize_risk(0.40)                          // 40% weight on variance
    .maximize("expected_return", 0.25)            // 25% weight on return
    .minimize("transition_risk", 0.15)            // 15% weight on climate risk
    .maximize("environmental_impact", 0.20)       // 20% weight on ESG
    .group("currency", "EUR", 0.30, 0.60)         // 30-60% in EUR
    .score_min("environmental_impact", 7.0)       // Portfolio avg ESG ≥ 7.0
    .score_max("transition_risk", 2.5)            // Portfolio avg climate risk ≤ 2.5
    .build();

// Compliance restrictions
let restrictions = Restrictions::builder()
    .long_only()
    .fully_invested()
    .max_single_weight(0.50)
    .build();

// Solve
let solution = PortfolioModel::new(&universe)
    .strategy(&strategy)
    .restrictions(restrictions)
    .solve()
    .unwrap();

println!("Status: {:?}", solution.status);
for (id, w) in &solution.weights {
    println!("  {}: {:.2}%", id, w * 100.0);
}
for (key, val) in &solution.portfolio_scores {
    println!("  {}: {:.4}", key, val);
}
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Strategy + Tactic + Restrictions                   │  You define this
│  "Minimize risk, maximize ESG, USD 10-20%..."       │
├─────────────────────────────────────────────────────┤
│  Compiler                          (quartz-portfolio)│  Quartz translates
│  Builds P, q, A, b matrices from your constraints   │
├─────────────────────────────────────────────────────┤
│  Solver                              (quartz-solver) │  Clarabel solves
│  min ½xᵀPx + qᵀx  s.t. Ax + s = b, s ∈ K         │
└─────────────────────────────────────────────────────┘
```

### Crate structure

| Crate | Purpose | Dependencies |
|-------|---------|-------------|
| **quartz-core** | `Asset`, `Universe`, `Dimension` — data types | `clarabel` (CscMatrix), `serde` |
| **quartz-solver** | Thin Clarabel wrapper — `CompiledProblem → RawSolution` | `clarabel`, `quartz-core` |
| **quartz-portfolio** | Modeler, compiler, facade — all business logic | `quartz-core`, `quartz-solver` |

### Three-layer constraint model

| Layer | Purpose | Example |
|-------|---------|---------|
| **Strategy** | Long-term investment approach with dimension weights and allocation targets | "40% risk, 25% return, 15% climate, EUR 40-60%" |
| **Tactic** | Short-term overlay that tightens strategy bounds | "Q2 2026: EUR 50-65%, increase risk weight to 55%" |
| **Restriction** | Hard compliance constraints, non-negotiable | "No tobacco, no weapons, max 5% per name" |

Tactics merge with strategies by **interval intersection**: bounds are tightened, never loosened. If the intersection is empty, Quartz returns an error before solving.

## Examples

```sh
# Minimum variance portfolio (3 assets)
cargo run --example markowitz -p quartz-portfolio

# Multi-dimensional ESG optimization (5 assets, 5 dimensions, group constraints)
cargo run --example multi_dimension -p quartz-portfolio

# Pareto frontier sweep (risk vs return trade-off table)
cargo run --example frontier -p quartz-portfolio

# Parallel backtest: 1000 dates, serial vs parallel timing
cargo run --release --example backtest -p quartz-portfolio

# Sequential turnover-chained backtest: Clarabel vs OSQP cold vs OSQP warm-started
cargo run --release --example backtest_warmstart -p quartz-portfolio --features osqp
```

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
)
s = (quartz.Strategy("ESG")
     .minimize_risk(0.5)
     .maximize("expected_return", 0.3)
     .maximize("environmental_impact", 0.2))
r = quartz.Restrictions(long_only=True, fully_invested=True, max_single_weight=0.6)

sol = quartz.solve(u, s, restrictions=r)
print(sol.status, sol.weights, sol.portfolio_scores)

# Batch (GIL released, rayon-parallel) and Pareto frontier:
sols = quartz.solve_batch([(u, s)] * 100, restrictions=r)
frontier = quartz.sweep(u, s, "expected_return", "financial_risk", n_points=25, restrictions=r)
```

## Building

```sh
cargo build          # Build all crates
cargo test           # Run all tests (29 unit tests)
```

Requires **Rust stable** (edition 2021). No C/C++ toolchain needed for the default build; the optional `osqp` feature vendors the OSQP C solver and needs cmake + a C compiler (`cargo test -p quartz-solver --features osqp`).

## Roadmap

- [x] Factor covariance model support (`Σ = BFBᵀ + D`) for O(nk²) scaling
- [x] Pareto frontier exploration (multi-objective trade-off visualization)
- [x] Parallel batch solving with `rayon` (backtest 1000 dates in parallel)
- [x] Python bindings via PyO3 + maturin
- [x] OSQP backend for warm-start support
- [x] SOCP support for CVaR and tracking error constraints
- [ ] JSON/YAML strategy file loading

## License

Quartz is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

This means you are free to use, modify, and distribute Quartz, but any modified version — including use over a network (SaaS) — **must also be released under AGPL-3.0** with source code available.

See [LICENSE](LICENSE) for the full text.
