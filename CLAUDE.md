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
cargo test -p quartz-portfolio --no-default-features     # Serial fallback (no rayon) must also pass
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

Two more crates sit on top:

- `quartz-python` — PyO3 bindings (`import quartz`), built with maturin. The `pyo3/extension-module` feature is enabled **only** in its pyproject.toml (`[tool.maturin] features`), never in Cargo.toml — that keeps workspace `cargo test` linking libpython normally; the crate also sets `[lib] test = false, doctest = false` (tests are pytest). Binding patterns: kwargs constructors for data types, `PyRefMut` chaining for Strategy/Tactic (relies on `#[derive(Clone)]` on the Rust builders), one `QuartzError(ValueError)` exception, and `Python::detach` around every solve (clone Rust values out of pyclasses first — PyRef is not Send) so `solve_batch` runs rayon-parallel with the GIL released. Dev loop: `python3 -m venv .venv && .venv/bin/pip install maturin pytest numpy`, then `VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m crates/quartz-python/Cargo.toml` and `.venv/bin/pytest crates/quartz-python/tests -q`.
- `quartz-demo` — JSON stdin/stdout CLI bridge (single solve, `frontier`, and `batch` modes). The Streamlit demo in `demo/` now imports the `quartz` bindings directly instead of shelling out to this binary; the app needs the wheel installed (`maturin build --release -m crates/quartz-python/Cargo.toml && pip install target/wheels/quartz-*.whl`). Demo data: `python demo/fetch_data.py` (Yahoo Finance → CSVs; ESG/climate scores are random placeholders), then `streamlit run demo/app.py`.

### Compilation pipeline

`PortfolioModel::solve()` (`model.rs`) → `compile()` (`compiler.rs`) → `solve_qp()` (quartz-solver) → `PortfolioSolution` (`solution.rs`).

The compiler is the heart of the system. Each constraint type in `quartz-portfolio/src/constraints/` implements `compile() -> ConstraintContribution` — a set of `(row, col, val)` triplets, `b` entries, and equality/inequality row counts. The compiler offsets local row indices, assembles everything into Clarabel's `CscMatrix` (no `sprs` dependency), and orders rows **equalities first (ZeroConeT), then inequalities (NonnegativeConeT)**.

### Parallel batch solving

`batch.rs` exposes `solve_batch(&[BatchProblem], &SolverSettings)` — independent problems (e.g. one per backtest date) solved on the rayon pool, results in input order, errors isolated per item (infeasible is an `Ok` status, not `Err`). Parallelism is the `parallel` cargo feature (default on; `rayon` optional dep) routed through the crate-private `par::par_map` shim so the serial fallback compiles identically — the frontier module uses the same shim. Clarabel is deterministic and has no shared mutable state, so results are bit-identical across thread counts. Turnover *chaining* across dates is inherently sequential and out of scope; don't use `verbose: true` in parallel runs (interleaved stdout).

### Pareto frontier

`frontier.rs` re-solves the QP across dimension-weight combinations: `FrontierExplorer::sweep(dim_a, dim_b, n)` trades two dimensions (others fixed; the base strategy lies on the sweep) and `simplex_grid(resolution)` enumerates all compositions over every dimension (capped at 10 000 solves). Non-dominated points are flagged via `pareto_flags` over sense-canonicalized metrics. Two invariants: the tactic is merged **once up front** (per-point solves pass tactic `None` — `tactic::merge` re-normalizes weights and would distort swept values), and the quadratic dimension is kept at weight 0 rather than dropped (so `financial_risk` stays reported and factor-model y-vars stay constrained).

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
