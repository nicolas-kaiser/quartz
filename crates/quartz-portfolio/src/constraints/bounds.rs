use super::{ConstraintContribution, Triplet};
use serde::{Deserialize, Serialize};

/// Weight bounds for individual assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBounds {
    /// Lower bound for each asset weight (None = no lower bound constraint added).
    pub lower: Option<f64>,
    /// Upper bound for each asset weight (None = no upper bound constraint added).
    pub upper: Option<f64>,
}

impl WeightBounds {
    /// Long-only constraint: wᵢ ≥ 0 for all i.
    pub fn long_only() -> Self {
        Self {
            lower: Some(0.0),
            upper: None,
        }
    }

    /// Box constraints: lower ≤ wᵢ ≤ upper for all i.
    pub fn boxed(lower: f64, upper: f64) -> Self {
        Self {
            lower: Some(lower),
            upper: Some(upper),
        }
    }

    /// Max single weight: wᵢ ≤ max for all i.
    pub fn max_weight(max: f64) -> Self {
        Self {
            lower: None,
            upper: Some(max),
        }
    }

    /// Compile to constraint rows for n_assets variables.
    ///
    /// Convention for Clarabel NonnegativeConeT: Ax + s = b, s ≥ 0
    /// For upper bound wᵢ ≤ uᵢ:  wᵢ + s = uᵢ  →  s = uᵢ - wᵢ ≥ 0  ✓
    /// For lower bound wᵢ ≥ lᵢ:  -wᵢ + s = -lᵢ →  s = wᵢ - lᵢ ≥ 0  ✓
    pub fn compile(&self, n_assets: usize) -> ConstraintContribution {
        let mut contrib = ConstraintContribution::new();
        let mut row = 0;

        // Lower bounds: -wᵢ + s = -lᵢ, s ≥ 0
        if let Some(lb) = self.lower {
            for i in 0..n_assets {
                contrib.triplets.push(Triplet {
                    row,
                    col: i,
                    val: -1.0,
                });
                contrib.b_entries.push(-lb);
                row += 1;
            }
            contrib.n_inequality += n_assets;
        }

        // Upper bounds: wᵢ + s = uᵢ, s ≥ 0
        if let Some(ub) = self.upper {
            for i in 0..n_assets {
                contrib.triplets.push(Triplet {
                    row,
                    col: i,
                    val: 1.0,
                });
                contrib.b_entries.push(ub);
                row += 1;
            }
            contrib.n_inequality += n_assets;
        }

        contrib
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_only() {
        let bounds = WeightBounds::long_only();
        let contrib = bounds.compile(3);

        assert_eq!(contrib.n_equality, 0);
        assert_eq!(contrib.n_inequality, 3);
        assert_eq!(contrib.b_entries, vec![0.0, 0.0, 0.0]);
        // -w0, -w1, -w2
        assert_eq!(contrib.triplets.len(), 3);
        assert_eq!(contrib.triplets[0].val, -1.0);
    }

    #[test]
    fn test_boxed() {
        let bounds = WeightBounds::boxed(0.01, 0.10);
        let contrib = bounds.compile(2);

        assert_eq!(contrib.n_inequality, 4); // 2 lower + 2 upper
        assert_eq!(contrib.b_entries, vec![-0.01, -0.01, 0.10, 0.10]);
    }
}
