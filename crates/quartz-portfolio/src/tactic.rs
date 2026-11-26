use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::constraints::{
    CvarConstraint, GroupConstraint, ScoreBound, ScoreConstraint, TrackingErrorConstraint,
};
use crate::strategy::Strategy;

/// A tactical overlay that adjusts a strategy's parameters for a shorter horizon.
///
/// Tactics can tighten group and score bounds and override dimension weights.
/// When merged with a strategy, bounds are intersected: a tactic never
/// loosens a strategy bound. An overridden weight is the dimension's final
/// share of the objective; the other dimensions keep their proportions and
/// share the rest. See `docs/tactic-merge.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tactic {
    pub name: String,
    #[serde(default)]
    pub group_overrides: Vec<GroupConstraint>,
    #[serde(default)]
    pub score_overrides: Vec<ScoreConstraint>,
    #[serde(default)]
    pub dimension_weight_overrides: HashMap<String, f64>,
}

impl Tactic {
    #[must_use]
    pub fn builder(name: impl Into<String>) -> TacticBuilder {
        TacticBuilder {
            name: name.into(),
            group_overrides: Vec::new(),
            score_overrides: Vec::new(),
            dimension_weight_overrides: HashMap::new(),
        }
    }
}

#[derive(Clone)]
pub struct TacticBuilder {
    name: String,
    group_overrides: Vec<GroupConstraint>,
    score_overrides: Vec<ScoreConstraint>,
    dimension_weight_overrides: HashMap<String, f64>,
}

impl TacticBuilder {
    /// Override a group constraint bound. During merge, the interval will be
    /// intersected with the strategy's bound.
    #[must_use]
    pub fn override_group(
        mut self,
        tag_key: impl Into<String>,
        tag_value: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> Self {
        self.group_overrides
            .push(GroupConstraint::new(tag_key, tag_value, lower, upper));
        self
    }

    /// Tighten a score constraint. During merge, the bound is intersected
    /// with the strategy's bounds on the same score.
    #[must_use]
    pub fn override_score(mut self, constraint: ScoreConstraint) -> Self {
        self.score_overrides.push(constraint);
        self
    }

    /// Set the weight of a dimension by name. The value is the dimension's
    /// final share ("risk to 55%" → 0.55): the dimensions that are not
    /// overridden keep their proportions to each other and share the rest.
    /// An unknown name, a negative value or overrides that sum to more than
    /// 1 are a [`MergeError`] at solve time.
    #[must_use]
    pub fn override_weight(mut self, dimension_name: impl Into<String>, weight: f64) -> Self {
        self.dimension_weight_overrides
            .insert(dimension_name.into(), weight);
        self
    }

    pub fn build(self) -> Tactic {
        Tactic {
            name: self.name,
            group_overrides: self.group_overrides,
            score_overrides: self.score_overrides,
            dimension_weight_overrides: self.dimension_weight_overrides,
        }
    }
}

/// Error when a tactic cannot be merged into a strategy: an empty interval or
/// an invalid weight override.
///
/// Bounds are `(lower, upper)`; an open side of a score bound is ±∞.
#[derive(Debug, Clone)]
pub enum MergeError {
    /// A group override does not overlap the strategy's group bounds.
    EmptyGroupInterval {
        tag_key: String,
        tag_value: String,
        strategy_bounds: (f64, f64),
        tactic_bounds: (f64, f64),
    },
    /// A score override does not overlap the strategy's score bounds.
    EmptyScoreInterval {
        score_key: String,
        strategy_bounds: (f64, f64),
        tactic_bounds: (f64, f64),
    },
    /// A weight override names a dimension the strategy does not have.
    UnknownDimension {
        name: String,
        available: Vec<String>,
    },
    /// A weight override is negative or not finite.
    InvalidWeightOverride { name: String, weight: f64 },
    /// The weight overrides assign more than the whole objective.
    WeightOverridesExceedOne { total: f64 },
    /// The weight overrides assign less than the whole objective and no
    /// other dimension can take the rest (every dimension is overridden, or
    /// the others all have weight 0).
    WeightOverridesLeaveRemainder { total: f64 },
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::EmptyGroupInterval {
                tag_key,
                tag_value,
                strategy_bounds: s,
                tactic_bounds: t,
            } => write!(
                f,
                "Empty interval after merging tactic for ({tag_key}, {tag_value}): \
                 strategy [{}, {}] ∩ tactic [{}, {}] = ∅",
                s.0, s.1, t.0, t.1,
            ),
            MergeError::EmptyScoreInterval {
                score_key,
                strategy_bounds: s,
                tactic_bounds: t,
            } => write!(
                f,
                "Empty interval after merging tactic for score {score_key}: \
                 strategy [{}, {}] ∩ tactic [{}, {}] = ∅",
                s.0, s.1, t.0, t.1,
            ),
            MergeError::UnknownDimension { name, available } => write!(
                f,
                "Tactic overrides the weight of dimension '{name}', which the strategy \
                 does not have (dimensions: {})",
                available.join(", "),
            ),
            MergeError::InvalidWeightOverride { name, weight } => write!(
                f,
                "Tactic weight override for dimension '{name}' is {weight}; it must be \
                 a finite share of at least 0"
            ),
            MergeError::WeightOverridesExceedOne { total } => write!(
                f,
                "Tactic weight overrides sum to {total}; they are shares of the objective \
                 and cannot exceed 1"
            ),
            MergeError::WeightOverridesLeaveRemainder { total } => write!(
                f,
                "Tactic weight overrides sum to {total} and no other dimension has weight \
                 to take the remaining {}",
                1.0 - total,
            ),
        }
    }
}

impl std::error::Error for MergeError {}

/// Tolerance on sums of dimension weights.
const WEIGHT_TOL: f64 = 1e-9;

/// Apply the tactic's weight overrides.
///
/// An override is the dimension's final share. The dimensions that are not
/// overridden keep their proportions to each other and share `1 − Σ overrides`.
/// Without overrides the weights are returned untouched.
fn merge_weights(
    dimensions: &[quartz_core::Dimension],
    overrides: &HashMap<String, f64>,
) -> Result<Vec<quartz_core::Dimension>, MergeError> {
    let mut dims = dimensions.to_vec();
    if overrides.is_empty() {
        return Ok(dims);
    }

    // Sorted, so the reported name does not depend on HashMap iteration order
    let mut names: Vec<&String> = overrides.keys().collect();
    names.sort();
    for name in names {
        if !dims.iter().any(|d| &d.name == name) {
            return Err(MergeError::UnknownDimension {
                name: name.clone(),
                available: dims.iter().map(|d| d.name.clone()).collect(),
            });
        }
        let weight = overrides[name];
        if !weight.is_finite() || weight < 0.0 {
            return Err(MergeError::InvalidWeightOverride {
                name: name.clone(),
                weight,
            });
        }
    }

    // Shares assigned by the tactic (a name shared by two dimensions counts
    // twice), and the strategy weight of the dimensions left alone.
    let mut fixed = 0.0;
    let mut rest = 0.0;
    for d in &dims {
        match overrides.get(&d.name) {
            Some(w) => fixed += w,
            None => rest += d.weight,
        }
    }
    if fixed > 1.0 + WEIGHT_TOL {
        return Err(MergeError::WeightOverridesExceedOne { total: fixed });
    }
    let remainder = (1.0 - fixed).max(0.0);
    if remainder > WEIGHT_TOL && rest <= 0.0 {
        return Err(MergeError::WeightOverridesLeaveRemainder { total: fixed });
    }
    let scale = if rest > 0.0 { remainder / rest } else { 0.0 };
    for d in &mut dims {
        d.weight = match overrides.get(&d.name) {
            Some(&w) => w,
            None => d.weight * scale,
        };
    }
    Ok(dims)
}

/// The interval `(lower, upper)` allowed by all `constraints` on `score_key`
/// together (±∞ where a side is unbounded).
fn score_interval(constraints: &[ScoreConstraint], score_key: &str) -> (f64, f64) {
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for c in constraints.iter().filter(|c| c.score_key == score_key) {
        match c.bound {
            ScoreBound::Min(l) => lo = lo.max(l),
            ScoreBound::Max(u) => hi = hi.min(u),
            ScoreBound::Range(l, u) => {
                lo = lo.max(l);
                hi = hi.min(u);
            }
        }
    }
    (lo, hi)
}

/// Intersect the strategy's and the tactic's bounds on `score_key` into one
/// constraint. The tactic can only tighten.
fn merge_score(
    strategy: &[ScoreConstraint],
    tactic: &[ScoreConstraint],
    score_key: &str,
) -> Result<ScoreConstraint, MergeError> {
    let s = score_interval(strategy, score_key);
    let t = score_interval(tactic, score_key);
    let (lo, hi) = (s.0.max(t.0), s.1.min(t.1));
    if lo > hi {
        return Err(MergeError::EmptyScoreInterval {
            score_key: score_key.to_string(),
            strategy_bounds: s,
            tactic_bounds: t,
        });
    }
    let bound = match (lo.is_finite(), hi.is_finite()) {
        (true, true) => ScoreBound::Range(lo, hi),
        (true, false) => ScoreBound::Min(lo),
        // (false, false) cannot happen: the tactic has a constraint on this key
        _ => ScoreBound::Max(hi),
    };
    Ok(ScoreConstraint::new(score_key, bound))
}

/// The result of merging a strategy with an optional tactic.
pub struct MergedStrategy {
    pub dimensions: Vec<quartz_core::Dimension>,
    pub group_constraints: Vec<GroupConstraint>,
    pub score_constraints: Vec<ScoreConstraint>,
    pub fully_invested: bool,
    /// Passed through unchanged (no tactic override in v1; the natural future
    /// semantic is min-tightening of the limits).
    pub tracking_error: Option<TrackingErrorConstraint>,
    pub cvar: Option<CvarConstraint>,
}

/// Merge a strategy with an optional tactic.
///
/// - Group constraints are intersected: [max(l_s, l_t), min(u_s, u_t)]
/// - Score constraints are intersected the same way, per score key: the
///   higher minimum and the lower maximum win
/// - A dimension weight override is that dimension's final share; the other
///   dimensions keep their proportions and share the rest (`merge_weights`)
pub fn merge(strategy: &Strategy, tactic: Option<&Tactic>) -> Result<MergedStrategy, MergeError> {
    let tactic = match tactic {
        Some(t) => t,
        None => {
            return Ok(MergedStrategy {
                dimensions: strategy.dimensions.clone(),
                group_constraints: strategy.group_constraints.clone(),
                score_constraints: strategy.score_constraints.clone(),
                fully_invested: strategy.fully_invested,
                tracking_error: strategy.tracking_error.clone(),
                cvar: strategy.cvar.clone(),
            })
        }
    };

    // Merge group constraints by intersection
    let mut merged_groups = Vec::new();
    for sg in &strategy.group_constraints {
        let key = sg.group_key();
        if let Some(tg) = tactic
            .group_overrides
            .iter()
            .find(|g| g.group_key() == key)
        {
            let lower = sg.lower.max(tg.lower);
            let upper = sg.upper.min(tg.upper);
            if lower > upper {
                return Err(MergeError::EmptyGroupInterval {
                    tag_key: sg.tag_key.clone(),
                    tag_value: sg.tag_value.clone(),
                    strategy_bounds: (sg.lower, sg.upper),
                    tactic_bounds: (tg.lower, tg.upper),
                });
            }
            merged_groups.push(GroupConstraint::new(
                &sg.tag_key,
                &sg.tag_value,
                lower,
                upper,
            ));
        } else {
            merged_groups.push(sg.clone());
        }
    }

    // Add tactic group constraints that don't exist in strategy
    for tg in &tactic.group_overrides {
        let key = tg.group_key();
        if !strategy.group_constraints.iter().any(|g| g.group_key() == key) {
            merged_groups.push(tg.clone());
        }
    }

    // Merge score constraints by intersection. Every score the tactic touches
    // collapses into one constraint, placed where the strategy first had it
    // (or appended, for a score only the tactic constrains).
    let touched = |key: &str| tactic.score_overrides.iter().any(|t| t.score_key == key);
    let mut merged_scores: Vec<ScoreConstraint> = Vec::new();
    let mut done: Vec<&str> = Vec::new();
    for sc in strategy.score_constraints.iter().chain(&tactic.score_overrides) {
        let key = sc.score_key.as_str();
        if !touched(key) {
            merged_scores.push(sc.clone());
        } else if !done.contains(&key) {
            done.push(key);
            merged_scores.push(merge_score(
                &strategy.score_constraints,
                &tactic.score_overrides,
                key,
            )?);
        }
    }

    // Merge dimension weights
    let merged_dims = merge_weights(&strategy.dimensions, &tactic.dimension_weight_overrides)?;

    Ok(MergedStrategy {
        dimensions: merged_dims,
        group_constraints: merged_groups,
        score_constraints: merged_scores,
        fully_invested: strategy.fully_invested,
        tracking_error: strategy.tracking_error.clone(),
        cvar: strategy.cvar.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_no_tactic() {
        let s = Strategy::builder("Test")
            .minimize_risk(0.5)
            .maximize("return", 0.5)
            .group("currency", "USD", 0.10, 0.20)
            .build();

        let merged = merge(&s, None).unwrap();
        assert_eq!(merged.group_constraints.len(), 1);
        assert_eq!(merged.group_constraints[0].lower, 0.10);
    }

    #[test]
    fn test_merge_tighten_bounds() {
        let s = Strategy::builder("Test")
            .minimize_risk(1.0)
            .group("currency", "USD", 0.10, 0.30)
            .build();

        let t = Tactic::builder("Tactical")
            .override_group("currency", "USD", 0.15, 0.25)
            .build();

        let merged = merge(&s, Some(&t)).unwrap();
        assert_eq!(merged.group_constraints[0].lower, 0.15);
        assert_eq!(merged.group_constraints[0].upper, 0.25);
    }

    #[test]
    fn test_merge_empty_interval_error() {
        let s = Strategy::builder("Test")
            .minimize_risk(1.0)
            .group("currency", "USD", 0.10, 0.20)
            .build();

        let t = Tactic::builder("Bad")
            .override_group("currency", "USD", 0.30, 0.40)
            .build();

        let result = merge(&s, Some(&t));
        assert!(result.is_err());
    }

    fn bound_of(merged: &MergedStrategy, key: &str) -> ScoreBound {
        let all: Vec<_> = merged
            .score_constraints
            .iter()
            .filter(|c| c.score_key == key)
            .collect();
        assert_eq!(all.len(), 1, "one merged constraint per touched score");
        all[0].bound.clone()
    }

    #[test]
    fn test_merge_score_tightens_never_loosens() {
        let s = Strategy::builder("Test")
            .minimize_risk(1.0)
            .score_min("esg", 5.0)
            .score_max("carbon", 3.0)
            .score_min("untouched", 1.0)
            .build();

        // esg: a lower tactic minimum must not loosen; carbon: tighter max + new min
        let t = Tactic::builder("T")
            .override_score(ScoreConstraint::min("esg", 4.0))
            .override_score(ScoreConstraint::range("carbon", 1.0, 2.0))
            .override_score(ScoreConstraint::max("new", 9.0))
            .build();

        let merged = merge(&s, Some(&t)).unwrap();
        assert!(matches!(bound_of(&merged, "esg"), ScoreBound::Min(v) if v == 5.0));
        assert!(matches!(bound_of(&merged, "carbon"), ScoreBound::Range(l, u) if l == 1.0 && u == 2.0));
        assert!(matches!(bound_of(&merged, "untouched"), ScoreBound::Min(v) if v == 1.0));
        assert!(matches!(bound_of(&merged, "new"), ScoreBound::Max(v) if v == 9.0));
        // strategy order kept, tactic-only score appended
        let keys: Vec<_> = merged.score_constraints.iter().map(|c| c.score_key.as_str()).collect();
        assert_eq!(keys, ["esg", "carbon", "untouched", "new"]);
    }

    #[test]
    fn test_merge_score_combines_min_and_max() {
        let s = Strategy::builder("Test")
            .minimize_risk(1.0)
            .score_min("esg", 5.0)
            .build();
        let t = Tactic::builder("T")
            .override_score(ScoreConstraint::max("esg", 7.0))
            .build();

        let merged = merge(&s, Some(&t)).unwrap();
        assert!(matches!(bound_of(&merged, "esg"), ScoreBound::Range(l, u) if l == 5.0 && u == 7.0));
    }

    #[test]
    fn test_merge_score_empty_interval_error() {
        let s = Strategy::builder("Test")
            .minimize_risk(1.0)
            .score_min("esg", 5.0)
            .build();
        let t = Tactic::builder("Bad")
            .override_score(ScoreConstraint::max("esg", 4.0))
            .build();

        assert!(matches!(
            merge(&s, Some(&t)),
            Err(MergeError::EmptyScoreInterval { strategy_bounds, tactic_bounds, .. })
                if strategy_bounds.0 == 5.0 && tactic_bounds.1 == 4.0
        ));
    }

    // ---- weight overrides --------------------------------------------------

    /// The README strategy: 40% risk, 25% return, 15% transition, 20% environment.
    fn readme_strategy() -> Strategy {
        Strategy::builder("ESG Balanced")
            .minimize_risk(0.40)
            .maximize("expected_return", 0.25)
            .minimize("transition_risk", 0.15)
            .maximize("environmental_impact", 0.20)
            .build()
    }

    fn weights(merged: &MergedStrategy) -> Vec<f64> {
        merged.dimensions.iter().map(|d| d.weight).collect()
    }

    fn assert_weights(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn test_weight_override_is_the_final_share() {
        // "Risk to 55%" is 55%, not 0.55 / 1.15. The others keep their
        // proportions and share the remaining 45% (factor 0.45 / 0.60).
        let t = Tactic::builder("T").override_weight("financial_risk", 0.55).build();
        let merged = merge(&readme_strategy(), Some(&t)).unwrap();
        assert_weights(&weights(&merged), &[0.55, 0.1875, 0.1125, 0.15]);
    }

    #[test]
    fn test_several_weight_overrides() {
        let t = Tactic::builder("T")
            .override_weight("financial_risk", 0.55)
            .override_weight("expected_return", 0.15)
            .build();
        let merged = merge(&readme_strategy(), Some(&t)).unwrap();
        // 0.30 left for transition : environment = 15 : 20
        assert_weights(&weights(&merged), &[0.55, 0.15, 0.30 * 15.0 / 35.0, 0.30 * 20.0 / 35.0]);
    }

    #[test]
    fn test_weight_override_to_zero_and_to_one() {
        let s = readme_strategy();
        let off = Tactic::builder("T").override_weight("environmental_impact", 0.0).build();
        assert_weights(&weights(&merge(&s, Some(&off)).unwrap()), &[0.5, 0.3125, 0.1875, 0.0]);
        let all = Tactic::builder("T").override_weight("financial_risk", 1.0).build();
        assert_weights(&weights(&merge(&s, Some(&all)).unwrap()), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_weight_overrides_on_every_dimension() {
        let s = Strategy::builder("S").minimize_risk(0.5).maximize("ret", 0.5).build();
        let ok = Tactic::builder("T")
            .override_weight("financial_risk", 0.7)
            .override_weight("ret", 0.3)
            .build();
        assert_weights(&weights(&merge(&s, Some(&ok)).unwrap()), &[0.7, 0.3]);

        let short = Tactic::builder("T")
            .override_weight("financial_risk", 0.6)
            .override_weight("ret", 0.3)
            .build();
        assert!(matches!(
            merge(&s, Some(&short)),
            Err(MergeError::WeightOverridesLeaveRemainder { total }) if (total - 0.9).abs() < 1e-12
        ));
    }

    #[test]
    fn test_weight_override_errors() {
        let s = readme_strategy();
        let merge_with = |name: &str, w: f64| {
            let t = Tactic::builder("T").override_weight(name, w).build();
            merge(&s, Some(&t)).err()
        };
        match merge_with("risk", 0.55) {
            Some(MergeError::UnknownDimension { name, available }) => {
                assert_eq!(name, "risk");
                assert_eq!(available[0], "financial_risk");
                assert_eq!(available.len(), 4);
            }
            other => panic!("expected UnknownDimension, got {other:?}"),
        }
        assert!(matches!(
            merge_with("financial_risk", -0.1),
            Some(MergeError::InvalidWeightOverride { .. })
        ));
        assert!(matches!(
            merge_with("financial_risk", f64::NAN),
            Some(MergeError::InvalidWeightOverride { .. })
        ));
        assert!(matches!(
            merge_with("financial_risk", 1.2),
            Some(MergeError::WeightOverridesExceedOne { total }) if total == 1.2
        ));
        // The message names the dimension and what exists
        let msg = merge_with("risk", 0.55).unwrap().to_string();
        assert!(msg.contains("'risk'") && msg.contains("financial_risk"), "{msg}");
    }

    #[test]
    fn test_weight_override_on_unnormalised_strategy() {
        // Hand-built weights 2 / 2 / 4: the override is still a share of 1.
        let mut s = Strategy::builder("S")
            .minimize_risk(1.0)
            .maximize("a", 1.0)
            .maximize("b", 1.0)
            .build();
        for (d, w) in s.dimensions.iter_mut().zip([2.0, 2.0, 4.0]) {
            d.weight = w;
        }
        let t = Tactic::builder("T").override_weight("financial_risk", 0.4).build();
        assert_weights(&weights(&merge(&s, Some(&t)).unwrap()), &[0.4, 0.2, 0.4]);
    }

    #[test]
    fn test_tactic_without_weight_overrides_leaves_weights_alone() {
        let mut s = readme_strategy();
        s.dimensions[0].weight = 2.0; // not normalised, on purpose
        let t = Tactic::builder("T").override_group("currency", "EUR", 0.1, 0.2).build();
        let merged = merge(&s, Some(&t)).unwrap();
        assert_eq!(weights(&merged), vec![2.0, 0.25, 0.15, 0.20]);
    }
}
