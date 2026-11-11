# The compiler: from a portfolio strategy to `P, q, A, b` + cones

Quartz does not solve anything itself. The compiler
(`crates/quartz-portfolio/src/compiler.rs`) turns a universe, a strategy and its
restrictions into one conic program, hands it to the solver crate, and the model
turns the solver's answer back into portfolio weights.

This document describes exactly what goes in, what the five outputs
`P`, `q`, `A`, `b` and the cone list contain, and what comes back.

```
Universe ─┐
Strategy ─┤
Tactic ───┤   compile_with_costs()        solve_qp_with()         PortfolioModel::solve()
Restrict. ┼──────────────────────▶ CompiledProblem ─────▶ RawSolution ─────▶ PortfolioSolution
Turnover ─┤   (quartz-portfolio)    P, q, A, b, cones     x, z, status       weights, scores
Costs ────┘                         n_assets, n_aux       (quartz-solver)
```

Contents:

1. [The mathematical form](#1-the-mathematical-form)
2. [Inputs](#2-inputs)
3. [The decision vector `x`](#3-the-decision-vector-x)
4. [`P` — quadratic objective](#4-p--quadratic-objective)
5. [`q` — linear objective](#5-q--linear-objective)
6. [`A` and `b` — constraint rows](#6-a-and-b--constraint-rows)
7. [The cone list](#7-the-cone-list)
8. [`CompiledProblem`](#8-compiledproblem)
9. [Worked example](#9-worked-example)
10. [Output](#10-output)
11. [OSQP backend](#11-osqp-backend)
12. [Compile errors](#12-compile-errors)
13. [Pitfalls](#13-pitfalls)
14. [Adding a constraint](#14-adding-a-constraint)

---

## 1. The mathematical form

The compiler produces the standard form of [Clarabel](https://clarabel.org):

```
minimise    ½ xᵀ P x + qᵀ x
subject to  A x + s = b
            s ∈ K
```

| Symbol | Size | Meaning |
|---|---|---|
| `x` | `n_vars` | decision vector: asset weights, then auxiliary variables (section 3) |
| `P` | `n_vars × n_vars` | quadratic cost, symmetric positive semidefinite, **upper triangle only** |
| `q` | `n_vars` | linear cost |
| `A` | `m × n_vars` | one row per scalar constraint row |
| `b` | `m` | right-hand side, one entry per row of `A` |
| `s` | `m` | slack, `s = b − A x`. Not stored: the solver creates it |
| `K` | — | product of cones; says which set each slice of `s` must lie in |

There are no separate "equality" and "inequality" matrices and no variable bounds.
Every constraint is a row of `A` and `b`, and **the cone the row belongs to decides
its meaning**:

| Cone on rows `i..j` | Condition on the slack | Meaning for the rows |
|---|---|---|
| `ZeroConeT(k)` | `s = 0` | `A x = b` |
| `NonnegativeConeT(k)` | `s ≥ 0` | `A x ≤ b` |
| `SecondOrderConeT(d)` | `s₀ ≥ ‖(s₁ … s_{d−1})‖₂` | `b₀ − a₀ᵀx ≥ ‖b₁.. − A₁.. x‖₂` |

Because the only inequality available is `A x ≤ b`, the compiler writes every
constraint in that direction:

| Wanted | Row in `A` | Entry in `b` |
|---|---|---|
| `aᵀx ≤ u` | `+a` | `u` |
| `aᵀx ≥ l` | `−a` | `−l` |
| `aᵀx = c` | `+a` (zero cone) | `c` |
| `l ≤ aᵀx ≤ u` | two rows: `−a` then `+a` | `−l` then `u` |

That is why lower bounds show up with negated coefficients and negated right-hand
sides everywhere below.

---

## 2. Inputs

```rust
pub fn compile_with_costs(
    universe: &Universe,
    strategy: &Strategy,
    tactic: Option<&Tactic>,
    restrictions: &Restrictions,
    turnover: Option<&TurnoverConstraint>,
    costs: Option<&TransactionCosts>,
) -> Result<CompiledProblem, CompileError>
```

`compile(...)` is the same without `costs`. `PortfolioModel::solve()` calls
`compile_with_costs` for you. The Python bindings and the JSON CLI build the same
Rust values, so everything here applies to them too.

**Asset order is the contract.** Asset `i` of `universe.assets` is column `i` of
`P` and `A`, entry `i` of `q`, and entry `i` of every per-asset vector you pass in
(benchmark weights, previous weights, cost rates, each scenario row). Nothing is
matched by id.

### 2.1 `Universe` (quartz-core)

| Field | Shape | Used for | Checked by `Universe::build()` |
|---|---|---|---|
| `assets[i].id` | string | labels the output weights; `exclude_asset` | unique |
| `assets[i].tags` | `key → value` strings | group constraints, `exclude_tag` | — |
| `assets[i].scores` | `key → f64` | linear dimensions, score constraints | finite |
| `covariance` | see below | risk dimension, tracking error | dimensions |
| `scenarios` | `S` rows × `n` returns, optional | CVaR only | non-empty, every row length `n`, finite |

An asset that lacks a score key counts as **0** for that key (in `q`, in score
constraints, and in the reported portfolio score). It is not an error.

Covariance, one of:

| Variant | Data | Requirements |
|---|---|---|
| `Full(Σ)` | `n × n` `CscMatrix` | square of size `n`. Must be stored **full-symmetric** (both triangles) and be positive semidefinite. Symmetry and PSD are not checked |
| `Factor { loadings, factor_cov, specific_variance }` | `B`: `n × k`, `F`: `k × k`, `D`: length `n`; `Σ = B F Bᵀ + D` | `B` has `n` rows, `F` is `k × k`, `D` has length `n` with finite entries `≥ 0`. `F` must be stored full-symmetric and be PSD (not checked) |

Units are yours to keep consistent: a tracking-error limit is in the units of
`√Σ`, a CVaR limit in the units of the scenario returns, a cost rate in the units
of the score it reduces.

### 2.2 `Strategy`

| Field | Type | Compiles to |
|---|---|---|
| `dimensions` | list of `Dimension { name, dim_type, sense, weight }` | the objective `P`, `q` |
| `group_constraints` | list of `GroupConstraint { tag_key, tag_value, lower, upper }` | 2 inequality rows each |
| `score_constraints` | list of `ScoreConstraint { score_key, bound }`, `bound` = `Min(θ)`, `Max(θ)` or `Range(l, u)` | 1 or 2 inequality rows each |
| `fully_invested` | bool (builder and file default: `true`) | 1 equality row |
| `tracking_error` | optional `{ benchmark_weights (n), max_te }` | 1 second-order cone |
| `cvar` | optional `{ alpha, max_cvar }` | `1 + S` variables, `2S + 1` inequality rows |

A dimension is either

- `Quadratic` — portfolio variance `wᵀΣw` (the builder's `minimize_risk`), or
- `Linear { score_key }` — the weighted score `Σᵢ wᵢ · scoreᵢ` (`maximize` / `minimize`).

`sense` gives the sign: `Minimize → +1`, `Maximize → −1`. `weight` is the
dimension's share in the objective. `Strategy::builder(..).build()` and the file
loaders in `config.rs` divide the weights by their sum so they add up to 1; a
`Strategy` built by hand or by raw `serde` is used as is.

### 2.3 `Tactic` (optional)

Merged into the strategy first (`tactic::merge`), before anything is compiled:

| Tactic field | Effect on the merged strategy |
|---|---|
| `group_overrides` | same `(tag_key, tag_value)`: bounds become `[max(lower), min(upper)]`; empty interval → `MergeError`. New groups are added |
| `score_overrides` | **replaces** the strategy's constraint on the same `score_key`; new ones are added |
| `dimension_weight_overrides` | sets the weight of the dimension with that `name`, then all weights are re-normalised to sum to 1 |

`tracking_error`, `cvar` and `fully_invested` pass through unchanged.

### 2.4 `Restrictions`

| Field | Compiles to |
|---|---|
| `fully_invested` | 1 equality row (one row only, even if the strategy also asks for it) |
| `long_only` | `n` inequality rows `wᵢ ≥ 0` |
| `max_single_weight: Some(u)` | `n` inequality rows `wᵢ ≤ u` |
| `exclusions` | 1 equality row `wᵢ = 0` per excluded asset |

### 2.5 `TurnoverConstraint` (optional)

`{ previous_weights (n), max_turnover }` — adds `n` variables `t` and `3n + 1`
inequality rows for `Σ|wᵢ − wᵢ_prev| ≤ max_turnover`.

### 2.6 `TransactionCosts` (optional)

`{ rates (n, finite, ≥ 0), score_key }` — adds no rows and no variables; it adds
coefficients on the turnover variables `t` in `q` and in some score rows. Requires
a turnover constraint. Full description: [transaction-costs.md](transaction-costs.md).

---

## 3. The decision vector `x`

`x` always starts with the `n` asset weights. Auxiliary blocks follow in a fixed
order, each present only when its feature is used:

```
x = [ w₀ … w_{n−1} | t₀ … t_{n−1} | y₀ … y_{k−1} | ζ, u₀ … u_{S−1} ]
      asset weights   turnover       factor          CVaR
```

| Block | Columns | Length | Present when | Meaning |
|---|---|---|---|---|
| `w` | `0 .. n` | `n` | always | asset weights |
| `t` | `n .. 2n` | `n` | a turnover constraint is given | `tᵢ ≥ |wᵢ − wᵢ_prev|` |
| `y` | `n + n_t ..` | `k` | factor covariance **and** a quadratic dimension | factor exposures `y = Bᵀw` |
| `ζ, u` | `n + n_t + k ..` | `1 + S` | `strategy.cvar` is set | `ζ` ≈ value-at-risk, `u_s` = loss beyond `ζ` in scenario `s` |

`n_aux` is the total length of the last three blocks and `n_vars = n + n_aux`.
All of `P`, `q` and `A` have `n_vars` columns.

The `y` block is deliberately not created without a quadratic dimension: a free
variable that appears in no constraint and no cost can make the solver's KKT
system singular.

---

## 4. `P` — quadratic objective

`P` is `n_vars × n_vars` and only the risk dimension writes into it. Let

```
λ = Σ over quadratic dimensions of  weight · sign        (sign: minimise +1, maximise −1)
```

**Full covariance**

```
P[0..n, 0..n] = 2λ · Σ      (upper triangle),      zero elsewhere
```

**Factor covariance** `Σ = B F Bᵀ + D` — since `wᵀΣw = yᵀFy + wᵀDw` with `y = Bᵀw`:

```
P[i, i]                   = 2λ · Dᵢ      for each asset i        (w block, diagonal)
P[y_off + i, y_off + j]   = 2λ · F[i,j]  for i ≤ j               (y block, upper triangle)
```

That is `n + k(k+1)/2` non-zeros instead of `n(n+1)/2`. The link `y = Bᵀw` is
enforced by equality rows in `A` (section 6.1).

Points to know:

- Clarabel reads the **upper triangle only**; the compiler keeps entries with
  `row ≤ col` from your matrix and drops the rest. A matrix stored lower-triangle
  only would silently lose its off-diagonal terms — store both triangles.
- The factor 2 cancels the solver's `½`: `½ xᵀPx = λ · wᵀΣw`. The risk term is
  therefore on the same scale as the linear terms `λ · sᵀw`, and with weights
  0.5 risk / 0.5 return the solved trade-off is `0.5 · variance − 0.5 · return`.
  (Before this was fixed `P` was `λ·Σ`, which halved the risk weight; an old risk
  weight `r` in a risk/return mix corresponds to `r / (2 − r)` now.)
- The `t`, `ζ` and `u` blocks have no quadratic cost: their rows and columns of
  `P` are empty.
- Without a quadratic dimension `P` is the all-zero matrix and the problem is a
  linear (or second-order cone) program.

---

## 5. `q` — linear objective

`q` has length `n_vars`, starts at zero, and receives:

| Source | Entries | Value added |
|---|---|---|
| each linear dimension on `score_key` | `q[i]`, `i < n` | `weight · sign · scoreᵢ` |
| transaction costs, for a linear dimension that **maximises** `costs.score_key` | `q[n + i]` (the `tᵢ`) | `+ weight · rateᵢ` |

So a maximised score enters `q` with a minus sign (the solver minimises), a
minimised one with a plus sign, and several dimensions simply add up. Entries for
`y`, `ζ`, `u` are always 0.

Example: `maximize("expected_return", 0.5)` with returns `[0.10, 0.05, 0.03]`
gives `q = [−0.05, −0.025, −0.015]`.

---

## 6. `A` and `b` — constraint rows

Each constraint type produces a `ConstraintContribution`
(`constraints/mod.rs`):

| Field | Meaning |
|---|---|
| `triplets` | `(row, col, val)` entries of `A`. `row` is local to the contribution (starts at 0); `col` is a global column of `x` |
| `b_entries` | one right-hand-side value per local row, in row order |
| `n_equality` | number of rows that go in the zero cone |
| `n_inequality` | number of rows that go in the nonnegative cone |
| `soc_blocks` | dimensions of second-order cones; their rows follow the others |

The compiler stacks the contributions, adding a running row offset to each local
`row`, in three groups: **all equalities, then all inequalities, then all
second-order cone blocks**. Within a group the order is fixed:

| # | Group | Contribution | Present when | Rows |
|---|---|---|---|---|
| 1 | equality | fully invested | strategy or restrictions ask for it | 1 |
| 2 | equality | exclusions | at least one excluded asset | 1 per excluded asset |
| 3 | equality | factor link `y = Bᵀw` | `y` block exists | `k` |
| 4 | inequality | group constraints, in merged order | per group with ≥ 1 matching asset | 2 each |
| 5 | inequality | score constraints, in merged order | per constraint | 1 (`Min`, `Max`) or 2 (`Range`) |
| 6 | inequality | long only | `restrictions.long_only` | `n` |
| 7 | inequality | max single weight | `restrictions.max_single_weight` | `n` |
| 8 | inequality | turnover | turnover given | `3n + 1` |
| 9 | inequality | CVaR | `strategy.cvar` | `2S + 1` |
| 10 | second-order cone | tracking error | `strategy.tracking_error` | `n + 1` (full) or `1 + k + n` (factor) |

`m` is the sum of the last column. `b[r]` always belongs to row `r` of `A`, and
entry `r` of the dual vector `z` returned by the solver belongs to the same row.

### 6.1 Equality rows (`A x = b`)

| Constraint | Row of `A` | `b` |
|---|---|---|
| Fully invested `Σ wᵢ = 1` | `+1` on every `w` column | `1` |
| Exclusion of asset `i`: `wᵢ = 0` | `+1` on column `i` | `0` |
| Factor link `j`: `Σᵢ B[i,j]·wᵢ − y_j = 0` | `B[i,j]` on column `i`, `−1` on column `y_off + j` | `0` |

### 6.2 Inequality rows (`A x ≤ b`)

`G` = the assets whose tag `tag_key` equals `tag_value`; `sᵢ` = asset `i`'s score.

| Constraint | Rows, in order | Row of `A` | `b` |
|---|---|---|---|
| Group `lower ≤ Σ_{i∈G} wᵢ ≤ upper` | lower | `−1` on columns in `G` | `−lower` |
| | upper | `+1` on columns in `G` | `upper` |
| Score `Min(θ)`: `Σ sᵢwᵢ ≥ θ` | 1 row | `−sᵢ` on column `i` | `−θ` |
| Score `Max(θ)`: `Σ sᵢwᵢ ≤ θ` | 1 row | `+sᵢ` on column `i` | `θ` |
| Score `Range(l, u)` | lower | `−sᵢ` | `−l` |
| | upper | `+sᵢ` | `u` |
| Long only `wᵢ ≥ 0` | row `i` | `−1` on column `i` | `0` |
| Max weight `wᵢ ≤ u` | row `i` | `+1` on column `i` | `u` |

With transaction costs on the same `score_key`, the `Min` row and the lower row
of a `Range` also get `+rateᵢ` on column `n + i`, which makes them
`Σ sᵢwᵢ − Σ rateᵢ·tᵢ ≥ θ`. `Max` rows are unchanged.

**Turnover** — `3n + 1` rows over `w` (columns `0..n`) and `t` (columns `n..2n`):

| Rows | Constraint | Row of `A` | `b` |
|---|---|---|---|
| `0 .. n` | `wᵢ − tᵢ ≤ wᵢ_prev` | `+1` on `wᵢ`, `−1` on `tᵢ` | `wᵢ_prev` |
| `n .. 2n` | `−wᵢ − tᵢ ≤ −wᵢ_prev` | `−1` on `wᵢ`, `−1` on `tᵢ` | `−wᵢ_prev` |
| `2n .. 3n` | `tᵢ ≥ 0` | `−1` on `tᵢ` | `0` |
| `3n` | `Σ tᵢ ≤ max_turnover` | `+1` on every `tᵢ` | `max_turnover` |

The first two blocks together say `tᵢ ≥ |wᵢ − wᵢ_prev|`. Nothing forces `tᵢ`
down to the exact absolute value unless the budget row or a cost makes a larger
`t` unattractive, so a returned `t` is an upper estimate of the trade, not the
trade itself.

**CVaR** (Rockafellar–Uryasev) — `2S + 1` rows over `w`, `ζ` (column `c`) and
`u_s` (column `c + 1 + s`), with `r_s` the returns of scenario `s` and
`κ = 1 / ((1 − alpha) · S)`:

| Rows | Constraint | Row of `A` | `b` |
|---|---|---|---|
| `0` | `ζ + κ · Σ u_s ≤ max_cvar` | `+1` on `ζ`, `κ` on every `u_s` | `max_cvar` |
| `1 .. S+1` | `−r_sᵀw − ζ − u_s ≤ 0` | `−r_s[j]` on `w_j`, `−1` on `ζ`, `−1` on `u_s` | `0` |
| `S+1 .. 2S+1` | `u_s ≥ 0` | `−1` on `u_s` | `0` |

The loss in scenario `s` is `−r_sᵀw`. These are plain linear rows, so CVaR works
on every backend.

### 6.3 Second-order cone rows (tracking error)

Tracking error `√((w − w_b)ᵀ Σ (w − w_b)) ≤ max_te` adds rows but no variables.
The cone requires `s₀ ≥ ‖(s₁, …)‖₂` for the slack `s = b − A x` of the block, so
the block is built to make `s₀ = max_te` and the remaining slacks a vector whose
norm is the tracking error.

**Full covariance**, with the Cholesky factor `Σ = L Lᵀ` — block of `n + 1` rows:

| Local row | Row of `A` | `b` | Slack `b − A x` |
|---|---|---|---|
| `0` | empty | `max_te` | `max_te` |
| `1 + i` | row `i` of `Lᵀ` (entries `L[j][i]` on columns `j ≥ i`) | `(Lᵀ w_b)ᵢ` | `(Lᵀ (w_b − w))ᵢ` |

and `‖Lᵀ(w_b − w)‖² = (w − w_b)ᵀ Σ (w − w_b)`.

**Factor covariance**, with `F = G Gᵀ` — block of `1 + k + n` rows:

| Local row | Row of `A` | `b` |
|---|---|---|
| `0` | empty | `max_te` |
| `1 + j`, `j < k` | row `j` of `M = GᵀBᵀ`, on the `w` columns | `(M w_b)_j` |
| `1 + k + i` | `√Dᵢ` on column `i` | `√Dᵢ · w_b[i]` |

using `(w − w_b)ᵀΣ(w − w_b) = ‖GᵀBᵀ(w − w_b)‖² + ‖D^½(w − w_b)‖²`. These rows are
written on the `w` columns directly and never use the `y` variables, which may
not exist.

The Cholesky factorisation (`math.rs`) is dense and retries with a small diagonal
jitter (`1e-12 … 1e-6` × mean diagonal) when the matrix is singular; it fails
only on a genuinely indefinite matrix (`CholeskyFailure`).

---

## 7. The cone list

`cones` is an ordered list that partitions the `m` rows of `A` and `b` from top
to bottom. The compiler emits at most:

```
[ ZeroConeT(total_eq),            // rows 0 .. total_eq                 →  A x = b
  NonnegativeConeT(total_ineq),   // next total_ineq rows               →  A x ≤ b
  SecondOrderConeT(d) ]           // next d rows (tracking error)       →  s₀ ≥ ‖s₁..‖
```

- A cone whose row count would be 0 is left out, so the list has between 0 and 3
  entries.
- All equality rows share **one** zero cone and all inequality rows share **one**
  nonnegative cone, whatever their origin. The cone list does not tell you which
  constraint a row came from; the row order of section 6 does.
- The numbers inside the cones always add up to `m = b.len()`.
- `d` is `n + 1` for a full covariance and `1 + k + n` for a factor model.

To find a constraint's rows, add up the row counts of everything before it in
the table of section 6. Example with `n = 3`, fully invested, one group, long
only, turnover: row 0 budget; rows 1–2 the group; rows 3–5 long only; rows 6–15
turnover; cones `[ZeroConeT(1), NonnegativeConeT(15)]`.

---

## 8. `CompiledProblem`

Defined in `crates/quartz-solver/src/lib.rs`; this is the whole interface between
compiler and solver.

| Field | Type | Content |
|---|---|---|
| `p` | `CscMatrix<f64>` | `n_vars × n_vars`, upper triangle (section 4) |
| `q` | `Vec<f64>` | length `n_vars` (section 5) |
| `a` | `CscMatrix<f64>` | `m × n_vars` (section 6) |
| `b` | `Vec<f64>` | length `m` |
| `cones` | `Vec<SupportedConeT<f64>>` | ordered partition of the `m` rows (section 7) |
| `n_assets` | `usize` | `n`; the weights are `x[0..n_assets]` |
| `n_aux` | `usize` | number of auxiliary variables; `n_vars() = n_assets + n_aux` |

`CscMatrix` is Clarabel's compressed-sparse-column type: `m`, `n`, `colptr`
(length `n + 1`), `rowval` and `nzval`. Column `j` holds the entries
`colptr[j] .. colptr[j+1]`.

The problem carries no names: no asset ids, no constraint labels. Rows and
columns are identified by position only.

---

## 9. Worked example

Three assets, full covariance:

| | A | B | C |
|---|---|---|---|
| currency | USD | EUR | EUR |
| expected_return | 0.10 | 0.05 | 0.03 |

```
        ⎡0.040 0.006 0.002⎤
    Σ = ⎢0.006 0.090 0.009⎥
        ⎣0.002 0.009 0.010⎦
```

```rust
let strategy = Strategy::builder("example")
    .minimize_risk(0.5)
    .maximize("expected_return", 0.5)
    .group("currency", "EUR", 0.6, 0.9)
    .max_tracking_error(vec![0.4, 0.3, 0.3], 0.05)
    .build();
let restrictions = Restrictions::builder().long_only().fully_invested().build();
let problem = compile(&universe, &strategy, None, &restrictions, None)?;
```

No turnover, no factor model, no CVaR, so `x = [w_A, w_B, w_C]`, `n_aux = 0`.
The compiled problem (values as produced by the code):

```
P = ⎡0.040 0.006  0.002⎤      = 2 · 0.5 · Σ, upper triangle
    ⎢  ·   0.090  0.009⎥
    ⎣  ·     ·    0.010⎦

q = [−0.05, −0.025, −0.015]   = −0.5 · expected_return
```

| Row | `A` (w_A, w_B, w_C) | `b` | Cone | Meaning |
|---|---|---|---|---|
| 0 | `[ 1, 1, 1 ]` | `1` | Zero | `w_A + w_B + w_C = 1` |
| 1 | `[ 0, −1, −1 ]` | `−0.6` | Nonnegative | EUR `≥ 0.6` |
| 2 | `[ 0, 1, 1 ]` | `0.9` | Nonnegative | EUR `≤ 0.9` |
| 3 | `[ −1, 0, 0 ]` | `0` | Nonnegative | `w_A ≥ 0` |
| 4 | `[ 0, −1, 0 ]` | `0` | Nonnegative | `w_B ≥ 0` |
| 5 | `[ 0, 0, −1 ]` | `0` | Nonnegative | `w_C ≥ 0` |
| 6 | `[ 0, 0, 0 ]` | `0.05` | SOC row 0 | `s₀ = max_te` |
| 7 | `[ 0.2, 0.03, 0.01 ]` | `0.092` | SOC row 1 | `(Lᵀ(w_b − w))₀` |
| 8 | `[ 0, 0.29850, 0.02915 ]` | `0.09829` | SOC row 2 | `(Lᵀ(w_b − w))₁` |
| 9 | `[ 0, 0, 0.09513 ]` | `0.02854` | SOC row 3 | `(Lᵀ(w_b − w))₂` |

```
cones = [ZeroConeT(1), NonnegativeConeT(5), SecondOrderConeT(4)]      1 + 5 + 4 = 10 = m
```

Rows 7–9 of `A` are `Lᵀ` (`Σ = LLᵀ`; `0.2 = √0.04`) and their `b` entries are
`Lᵀ · [0.4, 0.3, 0.3]`.

Solver result (Clarabel):

```
status = Optimal
x      = [0.4000, 0.1254, 0.4746]
z      = [0.0324, 0.0241, ≈0, ≈0, ≈0, ≈0, 0.0045, −0.0003, −0.0042, 0.0015]
obj    = −0.02400
```

Reading it: the EUR bucket sits at its lower bound (`w_B + w_C = 0.6`), which is
why row 1 has a non-zero dual; the tracking-error cone is binding too (non-zero
duals on rows 6–9, reported `tracking_error` = 0.0500); rows 2–5 are slack.
`obj = 0.5·wᵀΣw − 0.5·μᵀw = 0.5·0.012500 − 0.5·0.060508`, i.e. the weighted sum
of the reported `financial_risk` and `expected_return`.

---

## 10. Output

### 10.1 `RawSolution` (solver crate)

| Field | Content |
|---|---|
| `status` | `Optimal`, `AlmostOptimal`, `Infeasible`, `DualInfeasible`, `MaxIterations`, `NumericalError` |
| `x` | primal vector, length `n_vars`, in the layout of section 3 |
| `z` | dual vector, length `m`, one entry per row of `A`, in the row order of section 6 |
| `obj_val` | `½ xᵀPx + qᵀx` at the returned `x` |
| `solve_time_s`, `iterations` | solver statistics |

The slack `s` is not returned; compute `b − A x` if you need it.

Duals: `z ≥ 0` on nonnegative-cone rows, and a positive value means the
constraint is binding; on zero-cone rows the sign is free; on a second-order cone
block `z` lies in the same cone. At the optimum `P x + q + Aᵀ z = 0`.

### 10.2 `PortfolioSolution` (what `PortfolioModel::solve()` returns)

| Field | Content |
|---|---|
| `status` | copied from the solver |
| `weights` | `(AssetId, weight)` for every asset, in universe order — `x[0..n]` |
| `weights_vec` | the same weights as a plain vector |
| `portfolio_scores` | see below |
| `objective_value` | `obj_val` — the scalarised objective `Σ weight · sign · metric` over the dimensions (variance for the risk dimension), less `weight · cᵀt` with transaction costs |
| `solve_time_s`, `iterations` | copied |
| `raw_x` | the full `x` including auxiliary variables |
| `raw_z` | the full `z` |

`portfolio_scores` is computed after the solve from the weights, not read from
the solver:

| Key | Value | Present when |
|---|---|---|
| every score key found on any asset | `Σᵢ wᵢ · scoreᵢ` (gross of transaction costs) | always |
| `financial_risk` | variance `wᵀΣw` — not a volatility, no dimension weight | the strategy has a quadratic dimension |
| `tracking_error` | `√((w − w_b)ᵀΣ(w − w_b))` | `strategy.tracking_error` is set |
| `cvar` | exact discrete CVaR of the scenarios at `alpha` | `strategy.cvar` is set and scenarios exist |

**Check `status` before using the weights.** An infeasible problem is not an
`Err`: `solve()` returns `Ok` with `status = Infeasible` and weights that mean
nothing (under OSQP they are `NaN`). `Err` is reserved for compile errors and
solver set-up failures.

`raw_x` and `raw_z` can be fed back through `PortfolioModel::warm_start(&prev)`;
they must come from a problem with the same `n_vars` and `m`.

---

## 11. OSQP backend

OSQP solves `min ½xᵀPx + qᵀx  s.t.  l ≤ A x ≤ u`. The backend
(`quartz-solver/src/osqp_backend.rs`) reuses `P`, `q`, `A` unchanged and derives
the bounds from `b` and the cone list:

| Cone | `l` | `u` |
|---|---|---|
| `ZeroConeT` rows | `b` | `b` |
| `NonnegativeConeT` rows | `−1e30` | `b` |
| `SecondOrderConeT` | — | `SolverError::Unsupported` |

So a strategy with a tracking-error limit cannot be solved with OSQP; everything
else, CVaR included, can. `z` has the same sign convention on both backends.

---

## 12. Compile errors

Checked before any matrix is built:

| `CompileError` | Cause |
|---|---|
| `NoDimensions` | `strategy.dimensions` is empty |
| `MergeError` | a tactic group override leaves an empty interval |
| `TurnoverDimensionMismatch` | `previous_weights.len() != n` |
| `TransactionCostsWithoutTurnover` | costs given without a turnover constraint |
| `TransactionCostDimensionMismatch` | `rates.len() != n` |
| `InvalidTransactionCost` | a rate is negative or not finite |
| `CvarRequiresScenarios` | `strategy.cvar` set but the universe has no scenarios |
| `InvalidCvarAlpha` | `alpha` not in `(0, 1)` or `max_cvar` not finite |
| `BenchmarkDimensionMismatch` | `benchmark_weights.len() != n` |
| `InvalidTrackingErrorLimit` | `max_te` not finite or `≤ 0` |
| `CholeskyFailure` | covariance not PSD (tracking error only) |

---

## 13. Pitfalls

Things the compiler accepts without complaint:

- **Unknown names match nothing.** A group constraint whose tag matches no asset
  produces **no rows** — even with `lower > 0`. An exclusion naming an unknown
  asset or tag is ignored. A score key no asset has is a vector of zeros.
- **Contradictory bounds compile.** `lower > upper` in a group or range, or a
  tracking-error limit that conflicts with an exclusion, gives a valid problem
  that the solver reports as `Infeasible`.
- **Covariance storage.** `Σ` and `F` must hold both triangles. The compiler
  keeps the upper one for `P`, while the Cholesky factor and the reported
  variance read the matrix as stored.
- **Maximising risk.** A quadratic dimension with `sense = Maximize` gives a
  negative `λ` and a `P` that is not PSD; the solver will fail or return
  nonsense.
- **Weights are not always normalised.** Only the builder, the file loaders and
  a tactic merge normalise dimension weights. They scale `P` and `q` directly, so
  un-normalised weights change the objective value (not the optimum, if all are
  scaled alike).
- **A tactic can loosen a score constraint.** Group bounds only tighten, but a
  score override replaces the strategy's constraint outright.
- **Exclusion rows have no stable order.** Excluded indices pass through a
  `HashSet`, so the order of those equality rows — and of their duals in `z` —
  can differ from run to run. The solution is unaffected.
- **`t` is not the trade.** See section 6.2; compute `|w − w_prev|` yourself.

---

## 14. Adding a constraint

1. Create `crates/quartz-portfolio/src/constraints/<name>.rs` with a `compile()`
   that returns a `ConstraintContribution`: local rows from 0, global columns,
   one `b` entry per row, and the right counter (`n_equality`, `n_inequality` or
   a `soc_blocks` entry). Write inequalities as `A x ≤ b`.
2. Re-export it from `constraints/mod.rs`.
3. In `compile_with_costs`, push it into `eq_contributions`,
   `ineq_contributions` or `soc_contributions`. Do not mix row kinds in one
   contribution: the assembler places a whole contribution in one group.
4. If it needs auxiliary variables, append the block **after** the existing
   ones, add its length to `n_aux`, and pass its column offset to `compile()`.
   Make sure every new variable appears in at least one row.
5. Add it to the tables in sections 3 and 6 of this document.
