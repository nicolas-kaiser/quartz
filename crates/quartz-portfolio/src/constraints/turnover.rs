use super::{ConstraintContribution, Triplet};
use serde::{Deserialize, Serialize};

/// Turnover constraint: limits the total change from a previous portfolio.
///
/// |wᵢ - wᵢ_prev| ≤ tᵢ, Σtᵢ ≤ max_turnover
///
/// This introduces n auxiliary variables t (one per asset), expanding
/// the decision vector from [w] (n) to [w, t] (2n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnoverConstraint {
    pub previous_weights: Vec<f64>,
    pub max_turnover: f64,
}

impl TurnoverConstraint {
    pub fn new(previous_weights: Vec<f64>, max_turnover: f64) -> Self {
        Self {
            previous_weights,
            max_turnover,
        }
    }

    /// Compile turnover constraints.
    ///
    /// Given n assets (columns 0..n for w, columns n..2n for t):
    ///
    /// 1) wᵢ - wᵢ_prev ≤ tᵢ  →  wᵢ - tᵢ + s = wᵢ_prev, s ≥ 0   (n rows)
    /// 2) wᵢ_prev - wᵢ ≤ tᵢ  →  -wᵢ - tᵢ + s = -wᵢ_prev, s ≥ 0  (n rows)
    /// 3) tᵢ ≥ 0              →  -tᵢ + s = 0, s ≥ 0               (n rows)
    /// 4) Σtᵢ ≤ max_turnover  →  Σtᵢ + s = max_turnover, s ≥ 0    (1 row)
    ///
    /// Total: 3n + 1 inequality rows, 0 equality rows.
    pub fn compile(&self, n_assets: usize) -> ConstraintContribution {
        let n = n_assets;
        let mut contrib = ConstraintContribution::new();
        let mut row = 0;

        // 1) wᵢ - tᵢ ≤ wᵢ_prev  →  wᵢ - tᵢ + s = wᵢ_prev
        for i in 0..n {
            contrib.triplets.push(Triplet {
                row,
                col: i,
                val: 1.0,
            });
            contrib.triplets.push(Triplet {
                row,
                col: n + i, // t_i
                val: -1.0,
            });
            contrib.b_entries.push(self.previous_weights[i]);
            row += 1;
        }
        contrib.n_inequality += n;

        // 2) -wᵢ - tᵢ ≤ -wᵢ_prev  →  -wᵢ - tᵢ + s = -wᵢ_prev
        for i in 0..n {
            contrib.triplets.push(Triplet {
                row,
                col: i,
                val: -1.0,
            });
            contrib.triplets.push(Triplet {
                row,
                col: n + i,
                val: -1.0,
            });
            contrib.b_entries.push(-self.previous_weights[i]);
            row += 1;
        }
        contrib.n_inequality += n;

        // 3) tᵢ ≥ 0  →  -tᵢ + s = 0
        for i in 0..n {
            contrib.triplets.push(Triplet {
                row,
                col: n + i,
                val: -1.0,
            });
            contrib.b_entries.push(0.0);
            row += 1;
        }
        contrib.n_inequality += n;

        // 4) Σtᵢ ≤ max_turnover  →  Σtᵢ + s = max_turnover
        for i in 0..n {
            contrib.triplets.push(Triplet {
                row,
                col: n + i,
                val: 1.0,
            });
        }
        contrib.b_entries.push(self.max_turnover);
        contrib.n_inequality += 1;

        contrib
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turnover_compile() {
        let tc = TurnoverConstraint::new(vec![0.5, 0.3, 0.2], 0.20);
        let contrib = tc.compile(3);

        // 3*3 + 1 = 10 inequality rows
        assert_eq!(contrib.n_inequality, 10);
        assert_eq!(contrib.n_equality, 0);
        assert_eq!(contrib.b_entries.len(), 10);
        // Last entry is max_turnover
        assert_eq!(*contrib.b_entries.last().unwrap(), 0.20);
    }
}
