use super::{ConstraintContribution, Triplet};
use quartz_core::Universe;
use serde::{Deserialize, Serialize};

/// A group allocation constraint: the sum of weights for assets matching
/// a tag key-value pair must be within [lower, upper].
///
/// Example: GroupConstraint::new("currency", "USD", 0.10, 0.20) means
/// 10% ≤ Σ(weights of USD assets) ≤ 20%.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConstraint {
    pub tag_key: String,
    pub tag_value: String,
    pub lower: f64,
    pub upper: f64,
}

impl GroupConstraint {
    pub fn new(
        tag_key: impl Into<String>,
        tag_value: impl Into<String>,
        lower: f64,
        upper: f64,
    ) -> Self {
        Self {
            tag_key: tag_key.into(),
            tag_value: tag_value.into(),
            lower,
            upper,
        }
    }

    /// Compile against a universe. Looks up asset indices matching (tag_key, tag_value)
    /// and creates inequality rows.
    ///
    /// Lower bound: -Σwᵢ + s = -lower, s ≥ 0
    /// Upper bound:  Σwᵢ + s = upper,  s ≥ 0
    pub fn compile(&self, universe: &Universe) -> ConstraintContribution {
        let indices = universe.asset_indices(&self.tag_key, &self.tag_value);
        let mut contrib = ConstraintContribution::new();

        if indices.is_empty() {
            return contrib;
        }

        let mut row = 0;

        // Lower bound: -Σwᵢ + s = -lower
        for &i in &indices {
            contrib.triplets.push(Triplet {
                row,
                col: i,
                val: -1.0,
            });
        }
        contrib.b_entries.push(-self.lower);
        contrib.n_inequality += 1;
        row += 1;

        // Upper bound: Σwᵢ + s = upper
        for &i in &indices {
            contrib.triplets.push(Triplet {
                row,
                col: i,
                val: 1.0,
            });
        }
        contrib.b_entries.push(self.upper);
        contrib.n_inequality += 1;
        let _ = row;

        contrib
    }

    /// Returns a unique key identifying this group (tag_key, tag_value).
    pub fn group_key(&self) -> (&str, &str) {
        (&self.tag_key, &self.tag_value)
    }
}

/// Fully invested constraint: Σwᵢ = 1 (equality).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FullyInvested;

impl FullyInvested {
    /// Compile: 1ᵀw + s = 1, s ∈ ZeroCone (equality).
    pub fn compile(&self, n_assets: usize) -> ConstraintContribution {
        let mut contrib = ConstraintContribution::new();

        for i in 0..n_assets {
            contrib.triplets.push(Triplet {
                row: 0,
                col: i,
                val: 1.0,
            });
        }
        contrib.b_entries.push(1.0);
        contrib.n_equality = 1;

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
            .add_asset(Asset::new("A").tag("currency", "USD"))
            .add_asset(Asset::new("B").tag("currency", "EUR"))
            .add_asset(Asset::new("C").tag("currency", "EUR"))
            .covariance_full(CscMatrix::from(&[
                [0.04, 0.006, 0.002],
                [0.006, 0.09, 0.009],
                [0.002, 0.009, 0.01],
            ]))
            .build()
            .unwrap()
    }

    #[test]
    fn test_group_constraint_eur() {
        let gc = GroupConstraint::new("currency", "EUR", 0.40, 0.60);
        let contrib = gc.compile(&test_universe());

        assert_eq!(contrib.n_equality, 0);
        assert_eq!(contrib.n_inequality, 2);
        // Lower: -(w1+w2) ≤ -0.40 → cols 1,2 with val -1.0
        // Upper: (w1+w2) ≤ 0.60 → cols 1,2 with val +1.0
        assert_eq!(contrib.triplets.len(), 4);
        assert_eq!(contrib.b_entries, vec![-0.40, 0.60]);
    }

    #[test]
    fn test_group_constraint_no_match() {
        let gc = GroupConstraint::new("currency", "GBP", 0.0, 0.10);
        let contrib = gc.compile(&test_universe());

        assert_eq!(contrib.n_rows(), 0);
    }

    #[test]
    fn test_fully_invested() {
        let fi = FullyInvested;
        let contrib = fi.compile(3);

        assert_eq!(contrib.n_equality, 1);
        assert_eq!(contrib.n_inequality, 0);
        assert_eq!(contrib.b_entries, vec![1.0]);
        assert_eq!(contrib.triplets.len(), 3);
    }
}
