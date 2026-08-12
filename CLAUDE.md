# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

All code lives in the `quartz/` subdirectory — that is the git repository and Cargo workspace root. Run all cargo commands from `quartz/`.

## What Quartz is

A multi-dimensional portfolio optimizer in Rust. It is a **modeler + compiler**, not a solver: it translates high-level portfolio strategies (risk, return, ESG scores, climate risk, custom dimensions) into a quadratic program `min ½xᵀPx + qᵀx s.t. Ax + s = b, s ∈ K` and delegates solving to Clarabel.rs. Licensed AGPL-3.0.

## Commands

```sh
cargo build                                              # Build all crates
cargo test                                               # Run all unit tests
cargo test -p quartz-portfolio                           # Test a single crate
cargo test -p quartz-portfolio compiler::                # Run tests in one module
cargo run --example markowitz -p quartz-portfolio        # Min-variance, 3 assets
cargo run --example multi_dimension -p quartz-portfolio  # Multi-dim ESG, 5 assets
```

Rust stable, edition 2021. Pure Rust — no C/C++ toolchain needed.

## Architecture

Three workspace crates with a strict layering (data → solver wrapper → business logic):

| Crate | Role |
|-------|------|
| `quartz-core` | Data types only: `Asset`, `Universe`, `Dimension`, `CovarianceModel`. Depends only on `clarabel` (for `CscMatrix`) and `serde`. No solver logic. |
| `quartz-solver` | Thin Clarabel wrapper: `CompiledProblem → RawSolution` via `solve_qp()`. All Clarabel solver API usage is isolated here. |
| `quartz-portfolio` | Everything else: constraints, `Strategy`/`Tactic`/`Restrictions`, the compiler, and the `PortfolioModel` facade. |

A `quartz-python` crate (PyO3 bindings) is planned but not yet implemented.

A fourth crate, `quartz-demo`, is a JSON stdin/stdout CLI bridge used by the Streamlit demo in `demo/` (see `demo/README.md`): `cargo build --release -p quartz-demo`, `python demo/fetch_data.py` (Yahoo Finance → CSVs in `demo/data/`), then `streamlit run demo/app.py`. ESG/climate scores in the demo data are random placeholders.

### Compilation pipeline

`PortfolioModel::solve()` (`model.rs`) → `compile()` (`compiler.rs`) → `solve_qp()` (quartz-solver) → `PortfolioSolution` (`solution.rs`).

The compiler is the heart of the system. Each constraint type in `quartz-portfolio/src/constraints/` implements `compile() -> ConstraintContribution` — a set of `(row, col, val)` triplets, `b` entries, and equality/inequality row counts. The compiler offsets local row indices, assembles everything into Clarabel's `CscMatrix` (no `sprs` dependency), and orders rows **equalities first (ZeroConeT), then inequalities (NonnegativeConeT)**.

### Three-layer constraint model

- **Strategy** — long-term dimension weights and allocation bounds ("40% risk, 25% return, EUR 40–60%")
- **Tactic** — short-term overlay merged into the strategy by **interval intersection** (`[max(l), min(u)]`); bounds only tighten, never loosen; empty intersection is a `MergeError` before solving
- **Restrictions** — hard compliance constraints (long-only, fully invested, max weight, exclusions)

### Key design decisions

- Auxiliary variables extend the decision vector in a fixed order: `[w (n), t (n, if turnover), y (k, if factor covariance)]`. Turnover hardcodes t at columns `n..2n`, so y always comes after t (`n_aux` in `CompiledProblem` counts both).
- Tags are `HashMap<String, String>` and scores `HashMap<String, f64>` — extensible, not enums.
- `CovarianceModel::Factor` (Σ = BFBᵀ + D) compiles via k auxiliary variables y = Bᵀw so the objective is `yᵀFy + wᵀDw` (block-diagonal sparse P, O(nk²)) plus k equality link rows (`constraints/factor.rs`). y vars are only allocated when a quadratic dimension is present — dead free variables can make Clarabel's KKT system singular. Covariance matrices (Σ and F) must be stored full-symmetric; the compiler extracts the upper triangle (Clarabel silently drops strict-lower-triangle P entries).
- No MIP support in v1: no cardinality or semi-continuous constraints.

## Conventions

- Builder patterns for all public-facing types (`Asset::new("X").tag(...).score(...)`, `Strategy::builder(...)`, `Restrictions::builder()`).
- Error types are per-crate plain enums implementing `Display` + `Error` — no `anyhow`/`thiserror`.
- `serde::Serialize`/`Deserialize` on all public data types.
- Tests live in `#[cfg(test)] mod tests` at the bottom of each file; examples in `crates/quartz-portfolio/examples/`.
- **Adding a new constraint type**: create a file in `quartz-portfolio/src/constraints/`, implement `compile() -> ConstraintContribution`, re-export from `constraints/mod.rs`, and wire it into `compiler.rs`.
