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
/// loosens a strategy bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Override the weight of a dimension by name.
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

/// Error when merging a tactic with a strategy leaves an empty interval.
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
        }
    }
}

impl std::error::Error for MergeError {}

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
/// - Dimension weights from tactic override matching ones by name
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
    let mut merged_dims = strategy.dimensions.clone();
    for dim in &mut merged_dims {
        if let Some(&new_weight) = tactic.dimension_weight_overrides.get(&dim.name) {
            dim.weight = new_weight;
        }
    }
    // Re-normalize weights
    let total_weight: f64 = merged_dims.iter().map(|d| d.weight).sum();
    if total_weight > 0.0 && (total_weight - 1.0).abs() > 1e-10 {
        for d in &mut merged_dims {
            d.weight /= total_weight;
        }
    }

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
}
