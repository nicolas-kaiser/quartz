//! Tracking-error constraint: ‖w − w_b‖_Σ ≤ max_te as a second-order cone.
//!
//! With Σ = LLᵀ (Cholesky), (w−w_b)ᵀΣ(w−w_b) = ‖Lᵀ(w−w_b)‖², so the SOC block
//! is s₀ = max_te (constant row) followed by s_i = (Lᵀ(w_b − w))_i, and
//! s₀ ≥ ‖s₁..‖ is exactly the constraint. No auxiliary variables.
//!
//! For the factor model Σ = BFBᵀ + D the norm splits:
//! (w−d)ᵀΣ(w−d) = ‖GᵀBᵀ(w−d)‖² + ‖D^½(w−d)‖² with F = GGᵀ, giving a SOC of
//! dimension 1+k+n. The rows are written over the w columns directly — the
//! factor y-variables are NOT reused (they only exist when a quadratic
//! dimension is present, and dead variables make the KKT system singular).

use quartz_core::CovarianceModel;
use serde::{Deserialize, Serialize};

use super::{ConstraintContribution, Triplet};
use crate::math::{cholesky_lower, csc_to_dense, CholeskyError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingErrorConstraint {
    /// Benchmark weights, ordered by universe index (length n).
    pub benchmark_weights: Vec<f64>,
    /// Maximum tracking error, in the covariance's units (e.g. annualized).
    pub max_te: f64,
}

impl TrackingErrorConstraint {
    pub fn new(benchmark_weights: Vec<f64>, max_te: f64) -> Self {
        Self {
            benchmark_weights,
            max_te,
        }
    }

    /// Compile to one SOC block. Dimension/positivity validation happens in
    /// the compiler; this can only fail on a non-PSD covariance.
    pub(crate) fn compile(
        &self,
        covariance: &CovarianceModel,
    ) -> Result<ConstraintContribution, CholeskyError> {
        let wb = &self.benchmark_weights;
        let n = wb.len();
        let mut contrib = ConstraintContribution::new();

        // Local row 0: s0 = max_te (empty A row; CSC rows are implicit)
        contrib.b_entries.push(self.max_te);

        match covariance {
            CovarianceModel::Full(cov) => {
                // Rows 1+i: row i of Lᵀ (= column i of L) over w columns
                let dense = csc_to_dense(cov);
                let l = cholesky_lower(&dense, n)?;
                for i in 0..n {
                    let row = 1 + i;
                    let mut b = 0.0;
                    for j in i..n {
                        let v = l[j * n + i]; // L[j][i] = (Lᵀ)[i][j]
                        if v != 0.0 {
                            contrib.triplets.push(Triplet { row, col: j, val: v });
                            b += v * wb[j];
                        }
                    }
                    contrib.b_entries.push(b);
                }
                contrib.soc_blocks.push(n + 1);
            }
            CovarianceModel::Factor {
                loadings,
                factor_cov,
                specific_variance,
            } => {
                let k = loadings.n;
                // M = GᵀBᵀ (k×n), F = GGᵀ
                let f_dense = csc_to_dense(factor_cov);
                let g = cholesky_lower(&f_dense, k)?;
                let b_dense = csc_to_dense(loadings); // n×k row-major
                // Rows 1+j: M[j] over w columns
                for j in 0..k {
                    let row = 1 + j;
                    let mut b = 0.0;
                    for i in 0..n {
                        // M[j][i] = Σ_{l>=j} G[l][j] * B[i][l]
                        let mut v = 0.0;
                        for l_idx in j..k {
                            v += g[l_idx * k + j] * b_dense[i * k + l_idx];
                        }
                        if v != 0.0 {
                            contrib.triplets.push(Triplet { row, col: i, val: v });
                            b += v * wb[i];
                        }
                    }
                    contrib.b_entries.push(b);
                }
                // Rows 1+k+i: sqrt(d_i) over w column i
                for (i, &d) in specific_variance.iter().enumerate() {
                    let row = 1 + k + i;
                    let v = d.sqrt();
                    if v != 0.0 {
                        contrib.triplets.push(Triplet { row, col: i, val: v });
                        contrib.b_entries.push(v * wb[i]);
                    } else {
                        contrib.b_entries.push(0.0);
                    }
                }
                contrib.soc_blocks.push(1 + k + n);
            }
        }
        Ok(contrib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::CscMatrix;

    #[test]
    fn test_full_covariance_layout() {
        // Diagonal Σ: L = diag(0.2, 0.1)
        let cov = CovarianceModel::Full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]]));
        let te = TrackingErrorConstraint::new(vec![1.0, 0.0], 0.05);
        let c = te.compile(&cov).unwrap();

        assert_eq!(c.soc_blocks, vec![3]); // n+1
        assert_eq!(c.n_equality, 0);
        assert_eq!(c.n_inequality, 0);
        assert_eq!(c.n_rows(), 3);
        assert_eq!(c.b_entries[0], 0.05); // s0 = max_te
        // Row 1: Lᵀ row 0 = [0.2, 0]; b = 0.2 * wb[0] = 0.2
        let r1: Vec<_> = c.triplets.iter().filter(|t| t.row == 1).collect();
        assert_eq!(r1.len(), 1);
        assert!((r1[0].val - 0.2).abs() < 1e-12 && r1[0].col == 0);
        assert!((c.b_entries[1] - 0.2).abs() < 1e-12);
        // Row 2: [0, 0.1]; b = 0.1 * wb[1] = 0
        assert!((c.b_entries[2]).abs() < 1e-12);
    }

    #[test]
    fn test_factor_covariance_layout() {
        // 3 assets, 2 factors
        let cov = CovarianceModel::Factor {
            loadings: CscMatrix::from(&[[1.0, 0.2], [0.8, -0.1], [0.5, 0.7]]),
            factor_cov: CscMatrix::from(&[[0.04, 0.01], [0.01, 0.02]]),
            specific_variance: vec![0.01, 0.02, 0.015],
        };
        let te = TrackingErrorConstraint::new(vec![0.4, 0.3, 0.3], 0.10);
        let c = te.compile(&cov).unwrap();

        assert_eq!(c.soc_blocks, vec![6]); // 1 + k + n = 1 + 2 + 3
        assert_eq!(c.n_rows(), 6);
        assert_eq!(c.b_entries.len(), 6);
        assert_eq!(c.b_entries[0], 0.10);
        // Specific-variance rows are diagonal: row 1+k+i touches only col i
        for i in 0..3 {
            let row = 3 + i;
            let entries: Vec<_> = c.triplets.iter().filter(|t| t.row == row).collect();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].col, i);
            assert!((entries[0].val - (if i == 0 { 0.01f64 } else if i == 1 { 0.02 } else { 0.015 }).sqrt()).abs() < 1e-12);
        }
    }
}
