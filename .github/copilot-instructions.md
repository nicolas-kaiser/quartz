# Quartz — Project Guidelines

## Overview

Quartz is a multi-dimensional portfolio optimizer in Rust. It's a **modeler + compiler** that translates portfolio strategies into QP problems solved by Clarabel.rs. It is not a solver itself.

## Architecture

```
quartz/
├── crates/
│   ├── quartz-core/        # Types: Asset, Universe, Dimension (no solver dependency)
│   ├── quartz-solver/      # Thin wrapper around Clarabel (CompiledProblem → RawSolution)
│   ├── quartz-portfolio/   # Modeler + Compiler (Strategy, Tactic, Restriction → QP)
│   └── quartz-python/      # PyO3 bindings (not yet implemented)
```

- **quartz-core** depends only on `clarabel` (for `CscMatrix`) and `serde`. No solver logic.
- **quartz-solver** wraps `clarabel::solver`. All Clarabel-specific API usage is isolated here.
- **quartz-portfolio** contains all business logic: constraints, strategy/tactic/restriction, the compiler, and the `PortfolioModel` facade.

## Code Style

- Rust edition 2021, stable toolchain.
- Use builder patterns for public-facing types (`Asset::new("X").tag(...).score(...)`).
- Constraints implement a `compile()` method returning `ConstraintContribution` (triplets + cone info).
- The compiler assembles contributions into Clarabel's `CscMatrix` via triplet format, equalities first then inequalities.
- `serde::Serialize`/`Deserialize` on all public data types for JSON/YAML config support.
- Tags are `HashMap<String, String>` (not enums) for extensibility.
- Scores are `HashMap<String, f64>`.

## Build and Test

```sh
cargo build                                              # Build all crates
cargo test                                               # Run all 29 unit tests
cargo run --example markowitz -p quartz-portfolio        # Min-variance 3 assets
cargo run --example multi_dimension -p quartz-portfolio  # Multi-dim ESG 5 assets
```

## Key Design Decisions

- **No custom solver**: We delegate to Clarabel.rs. The value is in the modeler/compiler, not the solver.
- **Clarabel's CscMatrix directly**: No `sprs` dependency. Avoids conversion overhead.
- **Compiler builds A via triplets**: Each constraint produces `(row, col, val)` triplets. The compiler offsets rows and assembles into CSC at the end.
- **Row ordering in A**: Equalities first (ZeroConeT), then inequalities (NonnegativeConeT).
- **Tactic merges by interval intersection**: `[max(l_strat, l_tact), min(u_strat, u_tact)]`. Error if empty.
- **Turnover adds n auxiliary variables**: Decision vector goes from `[w]` (n) to `[w, t]` (2n).
- **No MIP support in v1**: No cardinality or semi-continuous constraints.
- **Factor model support**: `CovarianceModel::Factor` is defined but not yet implemented in the compiler.

## Conventions

- Error types are per-crate enums (not `anyhow`). Use `thiserror` only if needed later.
- Tests go in `#[cfg(test)] mod tests` at the bottom of each file.
- Examples go in `crates/quartz-portfolio/examples/`.
- When adding a new constraint type: create a file in `constraints/`, implement `compile() -> ConstraintContribution`, re-export from `constraints/mod.rs`, and wire it into `compiler.rs`.

## License

AGPL-3.0 — all modifications must be published.
