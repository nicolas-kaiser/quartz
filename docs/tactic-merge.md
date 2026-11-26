# Merging a tactic into a strategy

A tactic is a short-term overlay on a strategy. `tactic::merge`
(`crates/quartz-portfolio/src/tactic.rs`) combines the two before anything is
compiled. There are three rules, one per kind of override, and every conflict
is an error (`MergeError`) raised before the solver runs.

| Tactic field | Rule |
|---|---|
| `group_overrides` | intersect with the strategy's bounds |
| `score_overrides` | intersect with the strategy's bounds |
| `dimension_weight_overrides` | the value is the dimension's final weight |

Tracking error, CVaR, `fully_invested` and `normalization` are not affected by
a tactic.

## Group bounds: intersection

For the same `(tag_key, tag_value)` the merged bounds are
`[max(lower), min(upper)]`. A group only the tactic names is added as is; the
match is exact, so `usd` does not tighten a strategy group on `USD`. That case
is caught at compile time (`TagValueNearMiss`), as is a tag key no asset has
(`UnknownTagKey`); see [reference-validation.md](reference-validation.md).

Strategy EUR 30–60%, tactic EUR 50–70% → EUR 50–60%. Strategy EUR 30–60%,
tactic EUR 70–80% → `EmptyGroupInterval`.

## Score bounds: intersection

All bounds on one score key, from the strategy and from the tactic, are
combined into a single constraint: the highest minimum and the lowest maximum
win. The result is a `Min`, a `Max` or a `Range`, placed where the strategy
first had that score (or appended for a score only the tactic constrains).

| Strategy | Tactic | Merged |
|---|---|---|
| `esg ≥ 5` | `esg ≥ 4` | `esg ≥ 5` (a tactic cannot loosen) |
| `esg ≥ 5` | `esg ≤ 7` | `5 ≤ esg ≤ 7` |
| `carbon ≥ 1`, `carbon ≤ 3` | `carbon ≤ 2` | `1 ≤ carbon ≤ 2` |
| `esg ≥ 5` | `esg ≤ 4` | `EmptyScoreInterval` |

So a tactic can only tighten group and score bounds.

## Dimension weights: the override is the final share

`override_weight("financial_risk", 0.55)` means risk has 55% of the objective.
The dimensions the tactic does not name keep their proportions to each other
and share what is left:

```
base      = the strategy's weights
fixed     = sum of the override values
rest      = sum of the base weights of the dimensions not overridden
weight_k  = override_k                        if overridden
          = base_k · (1 − fixed) / rest       otherwise
```

Example, strategy 40% risk / 25% return / 15% transition / 20% environment:

| Tactic | Risk | Return | Transition | Environment |
|---|---|---|---|---|
| none | 0.40 | 0.25 | 0.15 | 0.20 |
| risk → 0.55 | 0.55 | 0.1875 | 0.1125 | 0.15 |
| risk → 0.55, return → 0.15 | 0.55 | 0.15 | 0.1286 | 0.1714 |
| environment → 0 | 0.50 | 0.3125 | 0.1875 | 0 |

The merged weights always sum to 1 when there is at least one weight override.
A tactic without weight overrides leaves the weights exactly as they are.

Errors:

| `MergeError` | When |
|---|---|
| `UnknownDimension { name, available }` | the override names a dimension the strategy does not have. The name is the dimension's `name`: `financial_risk` for the risk dimension, the score key for dimensions added with `maximize` / `minimize` |
| `InvalidWeightOverride { name, weight }` | the value is negative or not finite |
| `WeightOverridesExceedOne { total }` | the overrides sum to more than 1 |
| `WeightOverridesLeaveRemainder { total }` | the overrides sum to less than 1 and nothing can take the rest: every dimension is overridden, or the others all have weight 0 |

If two dimensions share a name, both get the override and both count in the
sum.

## Where the errors surface

`compile` and `PortfolioModel::solve` return them as
`CompileError::MergeError`, the frontier explorer as `FrontierError::Merge`,
the Python bindings as `QuartzError`.

## Change of behaviour

Until this change the override replaced the weight and all weights were then
divided by their sum, so "risk to 55%" on the example above gave
0.55 / 1.15 = 47.8%, and an override naming an unknown dimension did nothing.
A tactic with weight overrides therefore gives a different portfolio now; a
tactic whose overrides sum to more than 1, or that names an unknown dimension,
used to solve and now fails with one of the errors above.
