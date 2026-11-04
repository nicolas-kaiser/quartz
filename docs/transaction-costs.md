# Transaction costs ("net of costs" scores)

Trading has a cost. With transaction costs, quartz optimises a score — typically expected
return — **net of the cost of getting there** from the current portfolio, instead of capping
turnover with an arbitrary limit and pricing the trades afterwards.

## Model

Costs need a turnover constraint, whose auxiliary variables `tᵢ ≥ |wᵢ − wᵢ_prev|` sit at
columns `n..2n` of the decision vector `[w, t, (y), (ζ, u)]`. A `TransactionCosts` value names
the score the costs reduce and the per-asset rates:

```
TransactionCosts { rates: [c₁ … cₙ], score_key: "expected_return" }
net score  =  sᵀw − cᵀt
```

`cᵢ` is in the units of `score_key` per unit of weight traded. For an annualised expected
return and a one-off fee `fᵢ` (fraction of the amount traded) spread over a holding period of
`H` years, `cᵢ = fᵢ / H`: a 10 bp fee kept for 2 years costs 5 bp of return a year.

### Where the score becomes net

Clarabel solves `min ½xᵀPx + qᵀx` with `P = (Σ_quad sign·weight)·Σ` and `q_w = Σ_linear
sign·weight·s`; the builder and the file loaders normalise dimension weights to sum to 1.
The cost enters exactly where the strategy wants **more** of `score_key`:

| strategy element on `score_key` | effect of costs |
|---|---|
| linear dimension, `maximize`, weight `ω` | objective `−ω·(sᵀw − cᵀt)`: `q[n+i] += ω·cᵢ` |
| `score_min(θ)`, lower side of a range | `sᵀw − cᵀt ≥ θ`: `+cᵢ` on `tᵢ` in that row |
| linear dimension, `minimize` | gross (unchanged) |
| `score_max(θ)`, upper side of a range | gross (unchanged) |
| quadratic (risk) dimension | unchanged — risk is not in return units |

The cost is weighted like the score it reduces, so a mixed strategy (`0.5` risk, `0.5` return)
trades risk against **net** return in the same proportion as against gross return.

Costs stay out of minimised scores and upper bounds on purpose: `t` is only a lower bound on
`|Δw|`, and the solver keeps it tight only when a larger `t` hurts. In a row where a larger `t`
would help, the solver could inflate `t` (up to the turnover cap) and the "net" figure would be
meaningless.

Consequences:

- A pure `minimize_risk` strategy is unaffected by costs, unless a `score_min` on `score_key`
  binds: the minimum-risk portfolio does not depend on what trading costs.
- "Minimum risk at a target return" with costs is minimum risk at a target **net** return
  (`score_min` row) — the efficient frontier reachable from today's portfolio after costs.
- Nothing else changes: no new variables, rows or cones; with no costs the QP is the same as
  without the feature.

### Caps

`TurnoverConstraint::max_turnover` stays a hard cap on `Σtᵢ`. To price trades without capping
them, pass a cap that cannot bind: between two fully invested long-only portfolios
`Σ|wᵢ − wᵢ_prev| ≤ 2`, so `2.0` is never active.

## Validation (compile time)

| error | when |
|---|---|
| `TransactionCostsWithoutTurnover` | costs without a turnover constraint |
| `TransactionCostDimensionMismatch` | `rates.len() != n_assets` |
| `InvalidTransactionCost { index, value }` | a rate is negative, NaN or infinite |

## Rust API

```rust
use quartz_portfolio::{PortfolioModel, Restrictions};

let solution = PortfolioModel::new(&universe)
    .strategy(&strategy)
    .restrictions(Restrictions::builder().long_only().fully_invested().build())
    .turnover(previous_weights.clone(), 2.0)              // 2.0: price, don't cap
    .transaction_costs(rates, "expected_return")          // any order w.r.t. turnover()
    .solve()?;
```

`BatchProblem::transaction_costs(rates, key)` and `FrontierExplorer::transaction_costs(rates, key)`
do the same per batch item / for every explored point. Lower level:
`compile_with_costs(universe, strategy, tactic, restrictions, turnover, Some(&costs))`
(`compile` is `compile_with_costs(.., None)`).

## Python API

`transaction_costs=(rates, score_key)` on `solve`, `Problem` (for `solve_batch`), `sweep` and
`simplex_grid`, next to `turnover=(previous_weights, max_turnover)`. Invalid input raises
`QuartzError`.

```python
sol = quartz.solve(
    universe, strategy,
    restrictions=restrictions,
    turnover=(prev_weights, 2.0),
    transaction_costs=([fee / holding_years for fee in fees], "expected_return"),
)
```

## Example

Returns 10 % / 5 % / 3 %, equal variances, held ⅓ each, maximise return, uniform cost 3 %:
moving C → A gains 7 % for 2 × 3 % of cost and happens; B → A gains 5 % < 6 % and does not.
The answer is `[⅔, ⅓, 0]` instead of `[1, 0, 0]` without costs
(`test_transaction_costs_trade_only_where_the_gain_pays`, in Rust and Python).

## History

- `ffa8540` first added `transaction_cost_rates`, an unscaled `q[n+i] += cᵢ` on the turnover
  constraint. It mixed units (a return-unit cost against ½·variance in every minimum-risk
  solve), was silently dropped without a prior `turnover()` call and accepted negative or NaN
  rates. Replaced by `TransactionCosts` (net of costs on a named score); the old keyword
  `transaction_cost_rates` no longer exists.
