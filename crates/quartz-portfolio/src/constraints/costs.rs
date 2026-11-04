use serde::{Deserialize, Serialize};

use super::{ConstraintContribution, ScoreBound, ScoreConstraint, Triplet};

/// Transaction costs that reduce one score: wherever the strategy asks for
/// more of `score_key`, it gets that score **net of costs**, `sᵀw − cᵀt`,
/// with `tᵢ = |wᵢ − wᵢ_prev|` the turnover auxiliary variables.
///
/// `rates[i]` is in the units of `score_key` per unit of weight traded. For a
/// return score in annual terms, a one-off fee `f` amortised over a holding
/// period of `H` years is `cᵢ = f / H`.
///
/// Costs need a turnover constraint (the t variables) and apply only where a
/// larger score is better, because t is a lower bound on |Δw| that the solver
/// keeps tight only when a higher t hurts:
/// - a linear dimension **maximising** `score_key`: objective term `−cᵀt`;
/// - a score constraint **minimum** on `score_key` (`Min`, the lower side of
///   `Range`): `sᵀw − cᵀt ≥ θ`.
///
/// Minimising `score_key` and upper bounds on it stay gross. A pure
/// minimum-risk objective is therefore unaffected unless a minimum on
/// `score_key` binds. See `quartz/docs/transaction-costs.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransactionCosts {
    pub rates: Vec<f64>,
    pub score_key: String,
}

impl TransactionCosts {
    pub fn new(rates: Vec<f64>, score_key: impl Into<String>) -> Self {
        Self {
            rates,
            score_key: score_key.into(),
        }
    }

    /// Make the minimum rows of a compiled score constraint on `score_key`
    /// net of costs. `contrib` is `sc.compile(..)`'s output (local rows);
    /// `t_offset` is the column of t₀ (= n).
    ///
    /// Min(θ) is `−sᵀw ≤ −θ` (row 0): adding `+cᵢ·tᵢ` gives `−(sᵀw − cᵀt) ≤ −θ`.
    /// Range(l, u) has its lower row first (row 0); its upper row stays gross.
    pub(crate) fn net_score_rows(
        &self,
        sc: &ScoreConstraint,
        contrib: &mut ConstraintContribution,
        t_offset: usize,
    ) {
        if sc.score_key != self.score_key {
            return;
        }
        let min_row = match sc.bound {
            ScoreBound::Min(_) | ScoreBound::Range(..) => 0,
            ScoreBound::Max(_) => return,
        };
        for (i, &c) in self.rates.iter().enumerate() {
            if c != 0.0 {
                contrib.triplets.push(Triplet {
                    row: min_row,
                    col: t_offset + i,
                    val: c,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::CscMatrix;
    use quartz_core::{Asset, Universe};

    fn universe() -> Universe {
        Universe::builder()
            .add_asset(Asset::new("A").score("ret", 0.10))
            .add_asset(Asset::new("B").score("ret", 0.05))
            .covariance_full(CscMatrix::identity(2))
            .build()
            .unwrap()
    }

    fn t_entries(contrib: &ConstraintContribution) -> Vec<(usize, usize, f64)> {
        contrib
            .triplets
            .iter()
            .filter(|t| t.col >= 2)
            .map(|t| (t.row, t.col, t.val))
            .collect()
    }

    #[test]
    fn min_row_is_net_of_costs() {
        let costs = TransactionCosts::new(vec![0.01, 0.0], "ret");
        let sc = ScoreConstraint::min("ret", 0.07);
        let mut c = sc.compile(&universe());
        costs.net_score_rows(&sc, &mut c, 2);
        // zero rates add no entries
        assert_eq!(t_entries(&c), vec![(0, 2, 0.01)]);
    }

    #[test]
    fn range_nets_only_its_lower_row() {
        let costs = TransactionCosts::new(vec![0.01, 0.02], "ret");
        let sc = ScoreConstraint::range("ret", 0.06, 0.09);
        let mut c = sc.compile(&universe());
        costs.net_score_rows(&sc, &mut c, 2);
        assert_eq!(t_entries(&c), vec![(0, 2, 0.01), (0, 3, 0.02)]);
    }

    #[test]
    fn max_rows_and_other_keys_stay_gross() {
        let costs = TransactionCosts::new(vec![0.01, 0.02], "ret");
        for sc in [
            ScoreConstraint::max("ret", 0.09),
            ScoreConstraint::min("esg", 1.0),
        ] {
            let mut c = sc.compile(&universe());
            costs.net_score_rows(&sc, &mut c, 2);
            assert!(t_entries(&c).is_empty());
        }
    }

    #[test]
    fn serde_round_trip() {
        let costs = TransactionCosts::new(vec![0.001, 0.0], "expected_return");
        let json = serde_json::to_string(&costs).unwrap();
        assert_eq!(
            serde_json::from_str::<TransactionCosts>(&json).unwrap(),
            costs
        );
    }
}
