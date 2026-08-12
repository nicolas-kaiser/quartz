//! Small dense linear-algebra helpers for constraint compilation.

use clarabel::algebra::CscMatrix;

#[derive(Debug, Clone)]
pub(crate) struct CholeskyError(pub String);

impl std::fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Expand a (small) CSC matrix to dense row-major storage.
pub(crate) fn csc_to_dense(m: &CscMatrix<f64>) -> Vec<f64> {
    let mut dense = vec![0.0; m.m * m.n];
    for j in 0..m.n {
        for idx in m.colptr[j]..m.colptr[j + 1] {
            dense[m.rowval[idx] * m.n + j] = m.nzval[idx];
        }
    }
    dense
}

/// Dense Cholesky factorization A = LLᵀ (L lower triangular, row-major).
///
/// Sample and PCA-truncated covariance matrices are routinely PSD-singular,
/// so non-positive pivots are retried with escalating diagonal jitter
/// (relative to the mean diagonal). Jitter slightly inflates the measured
/// tracking error — the conservative direction. Fails only on genuinely
/// indefinite input.
pub(crate) fn cholesky_lower(a: &[f64], n: usize) -> Result<Vec<f64>, CholeskyError> {
    debug_assert_eq!(a.len(), n * n);
    let mean_diag = (0..n).map(|i| a[i * n + i].abs()).sum::<f64>() / n.max(1) as f64;

    'jitter: for &scale in &[0.0, 1e-12, 1e-10, 1e-8, 1e-6] {
        let jitter = scale * mean_diag.max(f64::MIN_POSITIVE);
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                if i == j {
                    sum += jitter;
                }
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum <= 0.0 {
                        continue 'jitter;
                    }
                    l[i * n + i] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        return Ok(l);
    }
    Err(CholeskyError(
        "matrix is not positive semidefinite (Cholesky failed even with jitter)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(l: &[f64], n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    a[i * n + j] += l[i * n + k] * l[j * n + k];
                }
            }
        }
        a
    }

    #[test]
    fn test_cholesky_spd() {
        // Hand-checkable SPD 3x3
        let a = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let l = cholesky_lower(&a, 3).unwrap();
        assert!((l[0] - 2.0).abs() < 1e-12); // L[0][0] = sqrt(4)
        let back = reconstruct(&l, 3);
        for (x, y) in a.iter().zip(&back) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_rank_deficient_jitter() {
        // Rank-1 PSD matrix: vvᵀ with v = (1, 2)
        let a = [1.0, 2.0, 2.0, 4.0];
        let l = cholesky_lower(&a, 2).unwrap();
        let back = reconstruct(&l, 2);
        for (x, y) in a.iter().zip(&back) {
            assert!((x - y).abs() < 1e-5, "jittered reconstruction too far: {x} vs {y}");
        }
    }

    #[test]
    fn test_cholesky_indefinite_fails() {
        let a = [1.0, 2.0, 2.0, 1.0]; // eigenvalues 3, -1
        assert!(cholesky_lower(&a, 2).is_err());
    }

    #[test]
    fn test_csc_to_dense() {
        let m = CscMatrix::from(&[[1.0, 2.0], [0.0, 3.0]]);
        assert_eq!(csc_to_dense(&m), vec![1.0, 2.0, 0.0, 3.0]);
    }
}
