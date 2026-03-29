use quartz_core::{Dimension, DimensionType, Sense};
use serde::{Deserialize, Serialize};

use crate::constraints::{GroupConstraint, ScoreConstraint};

/// A portfolio strategy: the long-term investment approach.
///
/// Defines which dimensions to optimize (with weights), and strategic
/// allocation constraints (currency, sector, asset class bounds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
    pub dimensions: Vec<Dimension>,
    pub group_constraints: Vec<GroupConstraint>,
    pub score_constraints: Vec<ScoreConstraint>,
    pub fully_invested: bool,
}

impl Strategy {
    pub fn builder(name: impl Into<String>) -> StrategyBuilder {
        StrategyBuilder {
            name: name.into(),
            dimensions: Vec::new(),
            group_constraints: Vec::new(),
            score_constraints: Vec::new(),
            fully_invested: true,
        }
    }
}

/// Builder for constructing a Strategy fluently.
pub struct StrategyBuilder {
    name: String,
    dimensions: Vec<Dimension>,
    group_constraints: Vec<GroupConstraint>,
    score_constraints: Vec<ScoreConstraint>,
    fully_invested: bool,
}

impl StrategyBuilder {
    /// Add the quadratic risk dimension (minimize portfolio variance).
    pub fn minimize_risk(mut self, weight: f64) -> Self {
        self.dimensions.push(Dimension::quadratic(
            "financial_risk",
            Sense::Minimize,
            weight,
        ));
        self
    }

    /// Add a linear dimension to minimize (e.g. physical risk, transition risk).
    pub fn minimize(mut self, score_key: impl Into<String>, weight: f64) -> Self {
        let key: String = score_key.into();
        self.dimensions
            .push(Dimension::linear(&key, &key, Sense::Minimize, weight));
        self
    }

    /// Add a linear dimension to maximize (e.g. expected return, ESG score).
    pub fn maximize(mut self, score_key: impl Into<String>, weight: f64) -> Self {
        let key: String = score_key.into();
        self.dimensions
            .push(Dimension::linear(&key, &key, Sense::Maximize, weight));
        self
    }

    /// Add a custom dimension with full control.
    pub fn dimension(mut self, dim: Dimension) -> Self {
        self.dimensions.push(dim);
        self
    }

    /// Add a group allocation constraint.
    pub fn group(
        mut self,
        tag_key: impl Into<String>,
        tag_value: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> Self {
        self.group_constraints
            .push(GroupConstraint::new(tag_key, tag_value, lower, upper));
        self
    }

    /// Add a minimum portfolio score constraint.
    pub fn score_min(mut self, score_key: impl Into<String>, threshold: f64) -> Self {
        self.score_constraints
            .push(ScoreConstraint::min(score_key, threshold));
        self
    }

    /// Add a maximum portfolio score constraint.
    pub fn score_max(mut self, score_key: impl Into<String>, threshold: f64) -> Self {
        self.score_constraints
            .push(ScoreConstraint::max(score_key, threshold));
        self
    }

    /// Set whether the portfolio must be fully invested (Σw = 1).
    pub fn fully_invested(mut self, val: bool) -> Self {
        self.fully_invested = val;
        self
    }

    /// Build the strategy. Normalizes dimension weights to sum to 1.0.
    pub fn build(mut self) -> Strategy {
        // Normalize dimension weights
        let total_weight: f64 = self.dimensions.iter().map(|d| d.weight).sum();
        if total_weight > 0.0 && (total_weight - 1.0).abs() > 1e-10 {
            for d in &mut self.dimensions {
                d.weight /= total_weight;
            }
        }

        Strategy {
            name: self.name,
            dimensions: self.dimensions,
            group_constraints: self.group_constraints,
            score_constraints: self.score_constraints,
            fully_invested: self.fully_invested,
        }
    }
}

/// Returns the quadratic dimension from a strategy, if any.
pub fn find_quadratic_dimension(strategy: &Strategy) -> Option<&Dimension> {
    strategy
        .dimensions
        .iter()
        .find(|d| matches!(d.dim_type, DimensionType::Quadratic))
}

/// Returns all linear dimensions from a strategy.
pub fn find_linear_dimensions(strategy: &Strategy) -> Vec<&Dimension> {
    strategy
        .dimensions
        .iter()
        .filter(|d| matches!(d.dim_type, DimensionType::Linear { .. }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_builder() {
        let s = Strategy::builder("ESG Balanced")
            .minimize_risk(0.4)
            .maximize("expected_return", 0.3)
            .minimize("transition_risk", 0.15)
            .maximize("environmental_impact", 0.15)
            .group("currency", "USD", 0.10, 0.20)
            .group("currency", "EUR", 0.40, 0.60)
            .score_min("environmental_impact", 5.0)
            .build();

        assert_eq!(s.name, "ESG Balanced");
        assert_eq!(s.dimensions.len(), 4);
        assert_eq!(s.group_constraints.len(), 2);
        assert_eq!(s.score_constraints.len(), 1);

        // Weights should sum to 1.0
        let total: f64 = s.dimensions.iter().map(|d| d.weight).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_dimensions() {
        let s = Strategy::builder("Test")
            .minimize_risk(0.5)
            .maximize("return", 0.5)
            .build();

        assert!(find_quadratic_dimension(&s).is_some());
        assert_eq!(find_linear_dimensions(&s).len(), 1);
    }
}
