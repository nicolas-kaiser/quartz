use super::{ConstraintContribution, Triplet};
use quartz_core::Universe;
use serde::{Deserialize, Serialize};

/// Bound type for a portfolio-level score constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreBound {
    /// Portfolio-weighted score must be ≥ threshold.
    Min(f64),
    /// Portfolio-weighted score must be ≤ threshold.
    Max(f64),
    /// Portfolio-weighted score must be in [lower, upper].
    Range(f64, f64),
}

/// Constraint on the portfolio's aggregate score for a given dimension.
///
/// Example: ScoreConstraint::new("environmental_impact", ScoreBound::Min(5.0))
/// means Σᵢ(wᵢ * enviro_scoreᵢ) ≥ 5.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreConstraint {
    pub score_key: String,
    pub bound: ScoreBound,
}

impl ScoreConstraint {
    pub fn new(score_key: impl Into<String>, bound: ScoreBound) -> Self {
        Self {
            score_key: score_key.into(),
            bound,
        }
    }

    pub fn min(score_key: impl Into<String>, threshold: f64) -> Self {
        Self::new(score_key, ScoreBound::Min(threshold))
    }

    pub fn max(score_key: impl Into<String>, threshold: f64) -> Self {
        Self::new(score_key, ScoreBound::Max(threshold))
    }

    pub fn range(score_key: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self::new(score_key, ScoreBound::Range(lower, upper))
    }

    /// Compile against a universe.
    ///
    /// For Min(θ): -Σ(sᵢwᵢ) + s = -θ, s ≥ 0
    /// For Max(θ):  Σ(sᵢwᵢ) + s = θ,  s ≥ 0
    /// For Range(l,u): both rows
    pub fn compile(&self, universe: &Universe) -> ConstraintContribution {
        let scores = universe.score_vector(&self.score_key);
        let mut contrib = ConstraintContribution::new();
        let mut row = 0;

        match &self.bound {
            ScoreBound::Min(threshold) => {
                for (i, &s) in scores.iter().enumerate() {
                    if s != 0.0 {
                        contrib.triplets.push(Triplet {
                            row,
                            col: i,
                            val: -s,
                        });
                    }
                }
                contrib.b_entries.push(-threshold);
                contrib.n_inequality += 1;
            }
            ScoreBound::Max(threshold) => {
                for (i, &s) in scores.iter().enumerate() {
                    if s != 0.0 {
                        contrib.triplets.push(Triplet {
                            row,
                            col: i,
                            val: s,
                        });
                    }
                }
                contrib.b_entries.push(*threshold);
                contrib.n_inequality += 1;
            }
            ScoreBound::Range(lower, upper) => {
                // Lower: -sᵀw ≤ -lower
                for (i, &s) in scores.iter().enumerate() {
                    if s != 0.0 {
                        contrib.triplets.push(Triplet {
                            row,
                            col: i,
                            val: -s,
                        });
                    }
                }
                contrib.b_entries.push(-lower);
                contrib.n_inequality += 1;
                row += 1;

                // Upper: sᵀw ≤ upper
                for (i, &s) in scores.iter().enumerate() {
                    if s != 0.0 {
                        contrib.triplets.push(Triplet {
                            row,
                            col: i,
                            val: s,
                        });
                    }
                }
                contrib.b_entries.push(*upper);
                contrib.n_inequality += 1;
                let _ = row;
            }
        }

        contrib
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::CscMatrix;
    use quartz_core::Asset;

    fn test_universe() -> Universe {
        Universe::builder()
            .add_asset(Asset::new("A").score("esg", 8.0))
            .add_asset(Asset::new("B").score("esg", 5.0))
            .add_asset(Asset::new("C").score("esg", 3.0))
            .covariance_full(CscMatrix::identity(3))
            .build()
            .unwrap()
    }

    #[test]
    fn test_score_min() {
        let sc = ScoreConstraint::min("esg", 6.0);
        let contrib = sc.compile(&test_universe());

        assert_eq!(contrib.n_inequality, 1);
        assert_eq!(contrib.b_entries, vec![-6.0]);
        // -8w0 -5w1 -3w2 ≤ -6
        assert_eq!(contrib.triplets.len(), 3);
        assert_eq!(contrib.triplets[0].val, -8.0);
        assert_eq!(contrib.triplets[1].val, -5.0);
    }

    #[test]
    fn test_score_range() {
        let sc = ScoreConstraint::range("esg", 4.0, 7.0);
        let contrib = sc.compile(&test_universe());

        assert_eq!(contrib.n_inequality, 2);
        assert_eq!(contrib.b_entries, vec![-4.0, 7.0]);
    }
}
