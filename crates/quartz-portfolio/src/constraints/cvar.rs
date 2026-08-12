//! CVaR constraint via the Rockafellar–Uryasev linearization.
//!
//! Given S return scenarios r_s, portfolio loss in scenario s is −r_sᵀw.
//! CVaR_α (expected loss over the worst (1−α) tail) ≤ c holds iff there exist
//! ζ and u ≥ 0 with u_s ≥ loss_s − ζ and ζ + 1/((1−α)S)·Σ u_s ≤ c.
//!
//! Adds 1+S auxiliary variables (ζ at `aux_offset`, u_s after it) and 2S+1
//! inequality rows — purely linear, so it works on every solver backend.
//! If (1−α)·S < 1 the formulation degenerates gracefully to a max-loss
//! constraint.

use serde::{Deserialize, Serialize};

use super::{ConstraintContribution, Triplet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvarConstraint {
    /// Confidence level in (0, 1); 0.95 = expected loss over the worst 5%.
    pub alpha: f64,
    /// Maximum CVaR, in the scenarios' return units (e.g. daily loss).
    pub max_cvar: f64,
}

impl CvarConstraint {
    pub fn new(alpha: f64, max_cvar: f64) -> Self {
        Self { alpha, max_cvar }
    }

    /// Number of auxiliary variables for S scenarios: ζ plus one u per scenario.
    pub fn n_aux(n_scenarios: usize) -> usize {
        1 + n_scenarios
    }

    /// Compile to 2S+1 inequality rows. ζ lives at column `aux_offset`,
    /// u_s at `aux_offset + 1 + s`.
    pub fn compile(&self, scenarios: &[Vec<f64>], aux_offset: usize) -> ConstraintContribution {
        let s_count = scenarios.len();
        let q = 1.0 / ((1.0 - self.alpha) * s_count as f64);
        let z0 = aux_offset;
        let mut contrib = ConstraintContribution::new();

        // Row 0: ζ + q·Σ u_s ≤ max_cvar
        contrib.triplets.push(Triplet { row: 0, col: z0, val: 1.0 });
        for s in 0..s_count {
            contrib.triplets.push(Triplet {
                row: 0,
                col: z0 + 1 + s,
                val: q,
            });
        }
        contrib.b_entries.push(self.max_cvar);

        // Rows 1..=S: u_s ≥ −r_sᵀw − ζ  →  −r_sᵀw − ζ − u_s ≤ 0
        for (s, r) in scenarios.iter().enumerate() {
            let row = 1 + s;
            for (j, &ret) in r.iter().enumerate() {
                if ret != 0.0 {
                    contrib.triplets.push(Triplet { row, col: j, val: -ret });
                }
            }
            contrib.triplets.push(Triplet { row, col: z0, val: -1.0 });
            contrib.triplets.push(Triplet {
                row,
                col: z0 + 1 + s,
                val: -1.0,
            });
            contrib.b_entries.push(0.0);
        }

        // Rows S+1..=2S: u_s ≥ 0  →  −u_s ≤ 0
        for s in 0..s_count {
            contrib.triplets.push(Triplet {
                row: 1 + s_count + s,
                col: z0 + 1 + s,
                val: -1.0,
            });
            contrib.b_entries.push(0.0);
        }

        contrib.n_inequality = 2 * s_count + 1;
        contrib
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvar_compile_layout() {
        let scenarios = vec![vec![0.10, 0.01], vec![-0.20, 0.01], vec![0.05, 0.01]];
        let cvar = CvarConstraint::new(0.75, 0.05);
        let c = cvar.compile(&scenarios, 4); // ζ at col 4, u at 5..8

        assert_eq!(c.n_inequality, 7); // 2*3 + 1
        assert_eq!(c.n_equality, 0);
        assert!(c.soc_blocks.is_empty());
        assert_eq!(c.b_entries.len(), 7);
        assert_eq!(c.b_entries[0], 0.05);

        // Row 0: q = 1/((1-0.75)*3) = 4/3 on each u column
        let q_entries: Vec<_> = c.triplets.iter().filter(|t| t.row == 0 && t.col >= 5).collect();
        assert_eq!(q_entries.len(), 3);
        for e in q_entries {
            assert!((e.val - 4.0 / 3.0).abs() < 1e-12);
        }
        // Row 2 (scenario 1): -r coefficients on w cols, -1 on ζ and u_1
        assert!(c.triplets.iter().any(|t| t.row == 2 && t.col == 0 && (t.val - 0.20).abs() < 1e-12));
        assert!(c.triplets.iter().any(|t| t.row == 2 && t.col == 4 && t.val == -1.0));
        assert!(c.triplets.iter().any(|t| t.row == 2 && t.col == 6 && t.val == -1.0));
    }
}
