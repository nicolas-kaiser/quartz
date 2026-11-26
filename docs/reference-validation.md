# Reference validation: names that match nothing are errors

A strategy, a tactic and the restrictions refer to things in the universe by
name: score keys, tag keys and values, asset ids. A name that matches nothing
used to be accepted and to do nothing, so a typo produced a different
portfolio with status `Optimal`. Such names are now errors, raised before the
solver runs. The same applies one level up: a misspelt key in a strategy,
tactic or restrictions file is rejected instead of being ignored.

Code: `validate_references` in `crates/quartz-portfolio/src/compiler.rs`
(called by `compile`, and once by `PortfolioModel::solve` before the
normalisation anchor solves); strict file parsing in the serde attributes of
the types (`deny_unknown_fields`) and `DimensionRepr` in
`crates/quartz-core/src/dimension.rs`.

## Rules at compile time

All of these are `CompileError` variants. They are checked on the strategy
with the tactic merged in, so tactic groups and score bounds are covered, and
they do not depend on dimension weights: a dimension at weight 0 is held to
the same rules.

### Score keys

| Case | Result |
|---|---|
| A linear dimension, a score constraint or the transaction costs use a score key **no asset** has | `UnknownScoreKey { score_key, used_by, available }` |
| A linear dimension or a score constraint uses a score that is missing on **an asset that can be held** | `MissingScores { score_key, used_by, assets }` |
| An **excluded** asset lacks the score | accepted: its weight is fixed at 0 |
| The transaction-cost key exists but no dimension or bound consumes it | accepted: the costs do nothing, as before (see [transaction-costs.md](transaction-costs.md)) |

Why a partly missing score is an error: a missing value was read as 0. For a
minimised score (a risk, a carbon intensity) 0 is the best possible value, so
the unrated asset was favoured; for a maximised score it was the worst; inside
a score bound it shifted the portfolio average. None of these is a neutral
default. The two ways out are named in the message: give the asset a score, or
exclude it.

"Can be held" means "not excluded by the restrictions". An asset whose weight
happens to be forced to 0 by a group cap or a weight limit still needs the
score.

`financial_risk`, `tracking_error` and `cvar` are reported metrics, not
scores. Using one as a score key gives `UnknownScoreKey` with a hint
(`minimize_risk`, or the tracking-error / CVaR constraint).

Reported `portfolio_scores` still read a missing value as 0, for score keys
the strategy does not use. A reported score of a partly rated universe is
therefore a sum over the rated holdings only.

### Tags

| Case | Result |
|---|---|
| A group constraint or a tag exclusion uses a **tag key** no asset has (`sectr`) | `UnknownTagKey { tag_key, tag_value, used_by, available }` |
| Known tag key, **value** that no asset has | accepted |
| … and the group's bounds exclude a weight of 0 | `EmptyGroup` (unchanged) |
| … and the value equals an existing one except for case or surrounding spaces (`usd` for `USD`) | `TagValueNearMiss { tag_key, tag_value, existing, used_by }` |

An absent value is legitimate and common: "no Tobacco" applied to a universe
that holds none, or a `JPY 0–20%` cap on a universe without yen assets. It
cannot be told apart from a misspelt value ("Tobaco"), so only the near-miss
is rejected. `Restrictions::unmatched_exclusions(&universe)` returns the
exclusions that hit nothing, for callers who want to log them.

**An asset without the tag key is outside every rule on that key.** A cap
"USD ≤ 20%" and an exclusion "sector = Tobacco" only see assets that carry
`currency` and `sector`. This is not checked (universes are often partially
tagged); tag every asset on the keys that restrictions use.

### Excluded asset ids

| Case | Result |
|---|---|
| An exclusion names an id that is not in the universe | `UnknownExcludedAssets { ids }`, all unknown ids in one error |
| … with `ignore_unknown_assets` set on the restrictions | accepted, the id is skipped |
| The id equals an existing one except for case or surrounding spaces | `ExcludedAssetNearMiss { id, existing }`, with or without the flag |

`ignore_unknown_assets` (default off) is for one restricted list applied to
changing universes, such as a firm-wide list in a per-date backtest:

```rust
let restrictions = Restrictions::builder()
    .long_only()
    .exclude_asset("XYZ")
    .ignore_unknown_assets()
    .build();
let skipped = restrictions.unmatched_exclusions(&universe);   // log these
```

```python
r = quartz.Restrictions(long_only=True, exclude_assets=ids, ignore_unknown_assets=True)
r.unmatched_exclusions(universe)       # [["asset", "XYZ"], ["tag", "sector", "Tobacco"]]
```

```yaml
exclusions:
- !by_asset XYZ
ignore_unknown_assets: true
```

The flag covers asset ids only. An unknown tag key stays an error.

## Rules at load time

Strategy, tactic and restrictions files (YAML and JSON), and the JSON spec of
the `quartz-demo` bridge, reject keys they do not know:

```
unknown field `exclusion`, expected one of `exclusions`, `max_single_weight`, …
```

Before, every section was optional and unknown keys were dropped, so
`exclusion:` loaded as "no exclusions", `score_constrains:` as "no score
bounds", and a strategy file passed as a tactic loaded as an empty tactic.

A dimension accepts exactly `name`, `type`, `score_key`, `sense`, `weight`:

- `type: linear` needs a `score_key`;
- `type: quadratic` takes none (it is the portfolio variance);
- anything else, such as `wieght`, is an error.

The written format is unchanged, and files written by `to_yaml_string` /
`to_json_string` load as before.

## Where the errors surface

| Caller | Error |
|---|---|
| `compile`, `compile_with_costs` | `CompileError` |
| `PortfolioModel::solve` | `PortfolioError::Compile` |
| `solve_batch` | per item, the other items are solved |
| `FrontierExplorer` | `FrontierError::Portfolio` |
| File loaders | `ConfigError::Yaml` / `ConfigError::Json` |
| Python | `QuartzError` |
| `quartz-demo` | `{"error": …}` |

## Change of behaviour

Input that used to solve can now fail:

- a strategy or tactic that uses a score key, tag key or excluded asset id the
  universe does not have;
- a universe in which a score that is optimised or bounded is missing on an
  investable asset;
- a file with a key the format does not define, including keys that were
  harmlessly ignored (comments stored as extra keys, for instance).

Each message names what did not match and what exists.

## Not covered

Found in the same review, not changed here: duplicate dimension names; a
second tactic override on the same group; negative or all-zero dimension
weights; `NaN` in bounds; vectors given per asset (benchmark, previous
weights, cost rates) checked by length only.
