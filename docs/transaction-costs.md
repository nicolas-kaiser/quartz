# Transaction costs in the quartz objective

## Why costs belong in the objective

Historically the turnover constraint capped `Σ|wᵢ − wᵢ_prev|` at a fixed limit and fees were
computed after the fact. This has two problems:

1. The cap is arbitrary — it doesn't reflect the actual cost of trading.
2. It treats all assets the same: a 1% trade in a liquid equity costs the same as a 1% trade
   in a thinly-traded bond, even though the real fee can differ by an order of magnitude.

The clean solution is to include transaction costs directly in the objective:

```
min  λ·wᵀΣw  −  μᵀw  +  Σᵢ cᵢ·tᵢ
```

where `tᵢ = |wᵢ − wᵢ_prev|` are the turnover auxiliary variables.  The optimizer then naturally
trades exactly as much as the improvement justifies.  No calibration of a turnover cap is needed
(though `max_turnover` remains a valid hard safety rail when wanted).

## Math

### Variable layout

When a `TurnoverConstraint` is present, quartz extends the decision vector to
`x = [w₁…wₙ, t₁…tₙ, (y…), (ζ,u…)]`.  The `tᵢ` variables sit at columns `n..2n`.

The constraint compiler encodes `tᵢ ≥ |wᵢ − wᵢ_prev|` as three groups of inequality rows:

| rows | meaning |
|------|---------|
| `wᵢ − tᵢ ≤ wᵢ_prev` | upper bound on change |
| `−wᵢ − tᵢ ≤ −wᵢ_prev` | lower bound on change |
| `−tᵢ ≤ 0` | non-negativity of t |
| `Σtᵢ ≤ T` | optional total-turnover cap |

### Cost term

Adding `Σᵢ cᵢ·tᵢ` to the objective is a purely linear change to the `q` vector:

```
q[n + i]  +=  cᵢ
```

No new variables, no new constraints, no change to the sparsity pattern.  When all `cᵢ = 0`
(i.e. `transaction_cost_rates` is absent) the QP is bit-identical to the previous behaviour.

### Holding-period scaling

The cost per unit of weight change is the fee rate (a fraction of notional, e.g. 10 bp = 0.001).
To make this commensurable with an annualised expected return, it must be amortised over the
holding period H (years):

```
cᵢ = fee_rate[asset_class_i] / H
```

A 10 bp fee amortised over 2 years = 5 bp per year, comparable to a 5 bp improvement in
expected return.  Setting H = 1 is a neutral default that neither over- nor under-penalises
trading.

## Rust API

### Building a constrained solve with costs

```rust
use quartz_portfolio::{PortfolioModel, Restrictions};

let solution = PortfolioModel::new(&universe)
    .strategy(&strategy)
    .restrictions(Restrictions::builder().long_only().fully_invested().build())
    .turnover(previous_weights.clone(), 0.30)   // optional hard cap
    .transaction_cost_rates(cost_rates)          // per-asset cᵢ
    .solve()?;
```

`transaction_cost_rates` must be called **after** `.turnover()`.  Calling it before is a no-op
(the `TurnoverConstraint` doesn't exist yet).

### `TurnoverConstraint::with_costs`

```rust
let tc = TurnoverConstraint::new(previous_weights, max_turnover)
    .with_costs(cost_rates);
```

### Batch solving

```rust
let p = BatchProblem::new(&universe, &strategy)
    .restrictions(r)
    .turnover(prev, 0.30)
    .transaction_cost_rates(cost_rates);
```

## Python API

```python
solution = quartz.solve(
    universe, strategy,
    restrictions=restrictions,
    turnover=(prev_weights, 0.30),          # optional hard cap
    transaction_cost_rates=cost_rates,      # list[float], one per asset
)

# or in a batch:
problem = quartz.Problem(
    universe, strategy,
    restrictions=restrictions,
    turnover=(prev_weights, 0.30),
    transaction_cost_rates=cost_rates,
)
solutions = quartz.solve_batch([problem])
```

`transaction_cost_rates` is silently ignored when `turnover` is not set (there are no `t`
variables to attach costs to).

## Integration in the PPI Simulation Studio

The API layer (`ip_ppi_api/app/models/simulation.py`) exposes a `holding_period` run parameter
(years, optional).  When set:

```python
c_i = fee_rate[asset_class_i] / holding_period
```

Fee rates come from the tenant's historical transaction data (median `|fee/amount|` over 2 years,
per asset class), falling back to 10 bp for sparse histories and 0 for cash-like classes.  The
`holding_period` field is disabled for the `closest_compliant` objective and for cash-flow runs
(same logic as `max_turnover`).

## Interaction with `max_turnover`

The two controls are complementary:

| scenario | recommendation |
|----------|---------------|
| costs only | optimizer self-limits; no hard cap needed |
| costs + cap | use the cap as a risk-management guardrail |
| cap only (legacy) | unchanged behaviour when `holding_period` is absent |

When both are set the optimizer minimises cost subject to the cap.  The cap may be slack (never
binding) if the cost alone deters the optimizer from trading that much.
