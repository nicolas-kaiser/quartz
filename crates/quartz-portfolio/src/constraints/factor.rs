use clarabel::algebra::CscMatrix;

use super::{ConstraintContribution, Triplet};

/// Links factor-exposure auxiliary variables to asset weights.
///
/// When the covariance is a factor model Σ = BFBᵀ + D, the compiler introduces
/// k auxiliary variables y = Bᵀw so the quadratic objective becomes
/// yᵀFy + wᵀDw (O(nk²) instead of O(n²)). This constraint defines y:
///
///   for each factor j:  Σᵢ B[i,j]·wᵢ − y_j = 0    (k equality rows)
///
/// y_j lives at column `y_offset + j` of the decision vector.
pub struct FactorLink;

impl FactorLink {
    pub fn compile(&self, loadings: &CscMatrix<f64>, y_offset: usize) -> ConstraintContribution {
        let k = loadings.n;
        let mut contrib = ConstraintContribution::new();
        for j in 0..k {
            for idx in loadings.colptr[j]..loadings.colptr[j + 1] {
                contrib.triplets.push(Triplet {
                    row: j,
                    col: loadings.rowval[idx],
                    val: loadings.nzval[idx],
                });
            }
            contrib.triplets.push(Triplet {
                row: j,
                col: y_offset + j,
                val: -1.0,
            });
            contrib.b_entries.push(0.0);
        }
        contrib.n_equality = k;
        contrib
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_link_compile() {
        // 3 assets, 2 factors, y at columns 3..5
        let loadings = CscMatrix::from(&[[1.0, 0.2], [0.8, -0.1], [0.5, 0.7]]);
        let contrib = FactorLink.compile(&loadings, 3);

        assert_eq!(contrib.n_equality, 2);
        assert_eq!(contrib.n_inequality, 0);
        assert_eq!(contrib.b_entries, vec![0.0, 0.0]);

        // Each factor row has 3 loading entries + one -1.0 on its y column
        assert_eq!(contrib.triplets.len(), 8);
        let y_entries: Vec<_> = contrib
            .triplets
            .iter()
            .filter(|t| t.val == -1.0 && t.col >= 3)
            .collect();
        assert_eq!(y_entries.len(), 2);
        assert_eq!((y_entries[0].row, y_entries[0].col), (0, 3));
        assert_eq!((y_entries[1].row, y_entries[1].col), (1, 4));

        // B[2,1] = 0.7 lands in row 1 (factor 1), col 2 (asset 2)
        assert!(contrib
            .triplets
            .iter()
            .any(|t| t.row == 1 && t.col == 2 && t.val == 0.7));
    }
}
