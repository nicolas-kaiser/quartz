# Dimension normalisation: making the weights mean what they say

A strategy gives each dimension a weight: "40% risk, 25% return, 15% transition
risk, 20% environmental impact". By default Quartz divides every dimension by
its **feasible range** before the weight applies, so a weight is a share of a
0–1 trade-off and does not depend on the units of the scores.

Code: `crates/quartz-portfolio/src/normalization.rs`, applied in
`PortfolioModel::solve` (`model.rs`).

## The problem with raw weights

The dimensions come in unrelated units: a variance around 0.01, a return
around 0.1, an ESG score between 0 and 10. If the weights multiply the raw
values, the dimension with the largest numbers decides, whatever its weight.

Example: the three-asset universe of the README, long only and fully invested,
weights 40/25/15/20. Going from the worst to the best portfolio on one
dimension changes the raw objective by:

| Dimension | Weight | Worst → best | Raw effect on the objective |
|---|---|---|---|
| `financial_risk` (variance) | 0.40 | 0.0400 → 0.0026 | 0.015 |
| `expected_return` | 0.25 | 0.025 → 0.10 | 0.019 |
| `transition_risk` | 0.15 | 3.0 → 0.5 | 0.375 |
| `environmental_impact` | 0.20 | 6.5 → 9.5 | 0.600 |

The two score dimensions hold 35% of the weight and about 97% of the
objective. Making the weights sum to 1 does not help: multiplying all weights
by the same number never changes the optimum.

## The method

For a strategy with dimensions `k = 1..m`, value `f_k(w)` (variance `wᵀΣw` or
score `sᵀw`), sign `σ_k` (`+1` minimise, `−1` maximise) and weight `λ_k`:

1. **Anchor solves.** For each dimension with a non-zero weight, solve the
   problem with that dimension alone in the objective. Everything else is
   unchanged: group and score bounds, tracking error, CVaR, restrictions,
   turnover, transaction costs, the merged tactic. This gives `m` portfolios
   `w¹ … wᵐ`.
2. **Ranges.** `best_k = f_k(wᵏ)` and `worst_k` is the worst `f_k(wʲ)` over
   all anchors `j` (the highest for a minimised dimension, the lowest for a
   maximised one). This is the payoff table of multi-objective optimisation.
3. **Objective.** The solved problem is

   ```
   minimise   Σ_k  λ_k · σ_k · f_k(w) / |worst_k − best_k|
   ```

So `P = 2 · (λ_risk / range_risk) · Σ` and a linear dimension adds
`λ_k · σ_k · s / range_k` to `q`. Moving a dimension across its whole range now
changes the objective by exactly its weight.

`worst` comes from the other dimensions' optima and not from a separate
"worst case" solve, because the worst variance is a non-convex problem and
because the worst value that is ever relevant is the one reached when another
dimension gets everything it wants.

In the example the ranges are 0.0374 (variance), 0.075, 2.5 and 3.0, and the
four dimensions carry 0.40, 0.25, 0.15 and 0.20 of the objective.

### What changes in the result

With the same universe and restrictions:

| Weights risk/return/transition/environment | `raw` | `range` |
|---|---|---|
| 40 / 25 / 15 / 20 | 100% GREEN_BOND | 100% GREEN_BOND |
| 40 / 45 / 5 / 10 | 100% GREEN_BOND | 38% AAPL, 16% NESN, 46% GREEN_BOND |
| 30 / 60 / 5 / 5 | 100% GREEN_BOND | 75% AAPL, 25% NESN |

Under `raw` the portfolio stays in the bond until return has about 94% of the
weight (the other three kept in proportion), and then jumps to 100% AAPL. The first row stays the same under `range`, and rightly so: the
bond is the best asset on three of the four dimensions, which together hold
75% of the weight.

A change of units no longer matters: an ESG score on a 0–100 scale, or a
covariance in percent², gives the same portfolio.

## Using it

Range normalisation is the default everywhere. `Strategy.normalization`
selects it:

```rust
use quartz_portfolio::Normalization;

let strategy = Strategy::builder("ESG")
    .minimize_risk(0.4)
    .maximize("expected_return", 0.6)
    .normalization(Normalization::Raw)     // default: Normalization::Range
    .build();
```

```yaml
name: ESG
normalization: raw        # or: range (default when the key is absent)
dimensions: [...]
```

```python
s = quartz.Strategy("ESG").minimize_risk(0.4).maximize("expected_return", 0.6)
s = s.normalization("raw")            # default "range"
sol = quartz.solve(universe, s, restrictions=r)
sol.dimension_ranges                  # {"financial_risk": (best, worst), ...}
```

The JSON bridge (`quartz-demo`) takes `"normalization": "raw"` in `strategy`
and returns `dimension_ranges`; the Streamlit demo has a checkbox.

`PortfolioSolution.dimension_ranges` lists the ranges that were used, one
`DimensionRange { name, best, worst }` per dimension in strategy order. It is
empty when the weights were applied raw.

`objective_value` is the normalised objective above, without a constant: it is
not comparable between `raw` and `range`, nor between two problems with
different ranges.

### Cost and reuse

One extra solve per dimension with a non-zero weight, run in parallel with the
`parallel` feature. A strategy with one weighted dimension needs none.

The ranges depend on the universe and the constraints, not on the weights. To
re-solve with other weights, measure once and pass the ranges in:

```rust
let ranges = model().compute_dimension_ranges()?;          // Option<Vec<DimensionRange>>
let sol = model().dimension_ranges(ranges.unwrap()).solve()?;
```

`compute_dimension_ranges` anchors every dimension, including those at weight
0. The frontier explorer does this itself: `sweep` and `simplex_grid` measure
the ranges once and use them at every point (`FrontierResult.dimension_ranges`),
which also keeps the points comparable. `solve_batch` measures them per item,
since each item has its own universe.

### The compiler does not normalise

`compile` and `compile_with_costs` are pure functions and need no solver, so
they always produce the raw problem, whatever `strategy.normalization` says.
Only `PortfolioModel::solve` (and what is built on it: batch, frontier, the
bindings) normalises. `docs/compiler.md` describes the raw problem; with
ranges, each dimension's coefficient is divided by its range and nothing else
changes.

## Edge cases

| Case | Behaviour |
|---|---|
| One weighted dimension | no anchor solves, `dimension_ranges` empty |
| Dimension at weight 0 | not anchored in `solve`; its entry is `best = worst = 0` (divisor 1) |
| Range of zero (constant score, or all anchors agree) | divisor 1: the dimension keeps its raw coefficient. It cannot trade off, so the scale does not matter |
| Problem infeasible | an anchor is infeasible, the ranges are dropped, and the solve returns `Ok` with status `Infeasible` as before |
| A dimension is unbounded alone | `PortfolioError::Normalization { dimension, status }` |

The last case is the one to know. Maximising return with shorting allowed and
no weight limit has no optimum, although the combined problem (bounded by the
risk term) has one. Such a dimension has no range. Either bound the problem
(long only, a maximum weight, group bounds) or use `normalization: raw`.

## Limits

- `worst` is an estimate. When an anchor problem has several optimal
  portfolios (typical for a linear dimension with ties), the other dimensions'
  values at that anchor depend on which one the solver returns. Clarabel is
  deterministic, so results are reproducible, but a small change in the data
  can move a range.
- The ranges are measured on the reported scores, which are gross of
  transaction costs. With costs, the cost term on the turnover variables is
  divided by the same range as the dimension it belongs to, so it stays in
  that dimension's units.
- A tactic that tightens a bound changes the feasible set and therefore the
  ranges. The same weights then mean a share of a smaller range.

## Migration

Strategies with two or more weighted dimensions give different portfolios than
before this change. To reproduce earlier results exactly, set
`normalization: raw`. Weights that had been tuned to compensate for score
units (for example 0.95 on return to make it count at all) should be reset to
the intended shares.
