use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::constraints::{GroupConstraint, ScoreConstraint};
use crate::strategy::Strategy;

/// A tactical overlay that adjusts a strategy's parameters for a shorter horizon.
///
/// Tactics can override group constraint bounds and dimension weights.
/// When merged with a strategy, bounds are intersected (tightened).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tactic {
    pub name: String,
    pub group_overrides: Vec<GroupConstraint>,
    pub score_overrides: Vec<ScoreConstraint>,
    pub dimension_weight_overrides: HashMap<String, f64>,
}

impl Tactic {
    pub fn builder(name: impl Into<String>) -> TacticBuilder {
        TacticBuilder {
            name: name.into(),
            group_overrides: Vec::new(),
            score_overrides: Vec::new(),
            dimension_weight_overrides: HashMap::new(),
        }
    }
}

pub struct TacticBuilder {
    name: String,
    group_overrides: Vec<GroupConstraint>,
    score_overrides: Vec<ScoreConstraint>,
    dimension_weight_overrides: HashMap<String, f64>,
}

impl TacticBuilder {
    /// Override a group constraint bound. During merge, the interval will be
    /// intersected with the strategy's bound.
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

    /// Override a score constraint.
    pub fn override_score(mut self, constraint: ScoreConstraint) -> Self {
        self.score_overrides.push(constraint);
        self
    }

    /// Override the weight of a dimension by name.
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

/// Error when merging a tactic with a strategy results in empty intervals.
#[derive(Debug, Clone)]
pub struct MergeError {
    pub tag_key: String,
    pub tag_value: String,
    pub strategy_bounds: (f64, f64),
    pub tactic_bounds: (f64, f64),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Empty interval after merging tactic for ({}, {}): strategy [{}, {}] ∩ tactic [{}, {}] = ∅",
            self.tag_key, self.tag_value,
            self.strategy_bounds.0, self.strategy_bounds.1,
            self.tactic_bounds.0, self.tactic_bounds.1,
        )
    }
}

impl std::error::Error for MergeError {}

/// The result of merging a strategy with an optional tactic.
pub struct MergedStrategy {
    pub dimensions: Vec<quartz_core::Dimension>,
    pub group_constraints: Vec<GroupConstraint>,
    pub score_constraints: Vec<ScoreConstraint>,
    pub fully_invested: bool,
}

/// Merge a strategy with an optional tactic.
///
/// - Group constraints are intersected: [max(l_s, l_t), min(u_s, u_t)]
/// - Score constraints from tactic replace matching ones from strategy
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
                return Err(MergeError {
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

    // Merge score constraints: tactic overrides by score_key
    let mut merged_scores = Vec::new();
    for ss in &strategy.score_constraints {
        if let Some(ts) = tactic
            .score_overrides
            .iter()
            .find(|s| s.score_key == ss.score_key)
        {
            merged_scores.push(ts.clone());
        } else {
            merged_scores.push(ss.clone());
        }
    }
    for ts in &tactic.score_overrides {
        if !strategy
            .score_constraints
            .iter()
            .any(|s| s.score_key == ts.score_key)
        {
            merged_scores.push(ts.clone());
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
}
