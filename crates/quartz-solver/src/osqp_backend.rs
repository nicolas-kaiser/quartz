//! OSQP backend: solves the Clarabel-form problem `Ax + s = b, s ∈ K` as
//! OSQP-form `l ≤ Ax ≤ u`.
//!
//! Row conversion: equality rows (ZeroConeT) become `l = u = b`; inequality
//! rows (NonnegativeConeT, `Ax + s = b, s ≥ 0` i.e. `Ax ≤ b`) become
//! `l = −∞, u = b`. Dual sign conventions agree with Clarabel's (an active
//! upper bound gives y ≥ 0, matching z ≥ 0 on the nonnegative cone), so
//! `RawSolution.z` carries the same semantics from either backend.

use std::borrow::Cow;

use crate::{CompiledProblem, RawSolution, SolveStatus, SolverError, SolverSettings, WarmStart};

/// OSQP's infinity convention (OSQP_INFTY in the C core). Passing IEEE ±inf
/// instead poisons the duality-gap termination check with NaN (inf · 0), and
/// the solver then runs to max_iter despite converged residuals.
const OSQP_INFTY: f64 = 1e30;

/// Borrow a clarabel CSC matrix as an osqp CSC matrix (zero copy).
fn borrow_csc(m: &clarabel::algebra::CscMatrix<f64>) -> osqp::CscMatrix<'_> {
    osqp::CscMatrix {
        nrows: m.m,
        ncols: m.n,
        indptr: Cow::Borrowed(&m.colptr),
        indices: Cow::Borrowed(&m.rowval),
        data: Cow::Borrowed(&m.nzval),
    }
}

/// Convert the cone list to OSQP bounds. Only Zero (equality) followed by
/// Nonnegative (inequality) cones are expressible; anything else errors so
/// future cone types fail loudly on this path.
fn cones_to_bounds(problem: &CompiledProblem) -> Result<(Vec<f64>, Vec<f64>), SolverError> {
    use clarabel::solver::SupportedConeT;

    let m = problem.b.len();
    let mut l = vec![-OSQP_INFTY; m];
    let mut u = vec![0.0; m];
    let mut row = 0;
    for cone in &problem.cones {
        match cone {
            SupportedConeT::ZeroConeT(k) => {
                for i in row..row + k {
                    l[i] = problem.b[i];
                    u[i] = problem.b[i];
                }
                row += k;
            }
            SupportedConeT::NonnegativeConeT(k) => {
                for i in row..row + k {
                    u[i] = problem.b[i];
                }
                row += k;
            }
            other => {
                return Err(SolverError::Unsupported(format!(
                    "OSQP backend supports only Zero and Nonnegative cones, got {other:?}"
                )))
            }
        }
    }
    if row != m {
        return Err(SolverError::Setup(format!(
            "cone rows ({row}) do not cover all constraints ({m})"
        )));
    }
    Ok((l, u))
}

pub(crate) fn solve(
    problem: &CompiledProblem,
    settings: &SolverSettings,
    warm_start: Option<&WarmStart>,
) -> Result<RawSolution, SolverError> {
    let (l, u) = cones_to_bounds(problem)?;
    let n_vars = problem.n_vars();
    let m = problem.b.len();

    if let Some(ws) = warm_start {
        if ws.x.len() != n_vars {
            return Err(SolverError::WarmStart(format!(
                "warm-start x has length {}, problem has {n_vars} variables \
                 (chained across structurally different problems?)",
                ws.x.len()
            )));
        }
        if !ws.y.is_empty() && ws.y.len() != m {
            return Err(SolverError::WarmStart(format!(
                "warm-start y has length {}, problem has {m} constraints",
                ws.y.len()
            )));
        }
    }

    let osqp_settings = osqp::Settings::default()
        .verbose(settings.verbose)
        .max_iter(settings.max_iter)
        .eps_abs(settings.tol_gap_abs)
        .eps_rel(settings.tol_gap_rel)
        .eps_prim_inf(settings.tol_feas)
        .eps_dual_inf(settings.tol_feas)
        // Polishing recovers high accuracy from the first-order solution.
        .polishing(true)
        .warm_starting(true);

    let mut prob = osqp::Problem::new(
        borrow_csc(&problem.p),
        &problem.q,
        borrow_csc(&problem.a),
        &l,
        &u,
        &osqp_settings,
    )
    .map_err(|e| SolverError::Setup(format!("{e:?}")))?;

    if let Some(ws) = warm_start {
        if ws.y.is_empty() {
            prob.warm_start_x(&ws.x);
        } else {
            prob.warm_start(&ws.x, &ws.y);
        }
    }

    let status = prob.solve();
    let iterations = status.iter();
    let solve_time_s = status.run_time().as_secs_f64();

    let mapped = match status {
        osqp::Status::Solved(_) => SolveStatus::Optimal,
        osqp::Status::SolvedInaccurate(_) => SolveStatus::AlmostOptimal,
        osqp::Status::MaxIterationsReached(_) | osqp::Status::TimeLimitReached(_) => {
            SolveStatus::MaxIterations
        }
        osqp::Status::PrimalInfeasible(_) | osqp::Status::PrimalInfeasibleInaccurate(_) => {
            SolveStatus::Infeasible
        }
        osqp::Status::DualInfeasible(_) | osqp::Status::DualInfeasibleInaccurate(_) => {
            SolveStatus::DualInfeasible
        }
        _ => SolveStatus::NumericalError,
    };

    // Infeasibility certificates carry no primal iterate; return NaN-filled
    // full-length vectors so downstream `x[..n]` indexing never panics.
    let (x, z, obj_val) = match status.solution() {
        Some(sol) => (sol.x().to_vec(), sol.y().to_vec(), sol.obj_val()),
        None => (vec![f64::NAN; n_vars], vec![f64::NAN; m], f64::NAN),
    };

    Ok(RawSolution {
        status: mapped,
        x,
        z,
        obj_val,
        solve_time_s,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{solve_qp_with, Backend, SolverSettings, WarmStart};
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::SupportedConeT;

    /// min (x1-1)² + (x2-2)²  s.t. x1+x2 = 1, 0 ≤ xi ≤ 0.8.
    /// Optimum: x = (0.2, 0.8), the x2 ≤ 0.8 bound active.
    fn fixture() -> CompiledProblem {
        CompiledProblem {
            p: CscMatrix::from(&[[2.0, 0.0], [0.0, 2.0]]),
            q: vec![-2.0, -4.0],
            a: CscMatrix::from(&[
                [1.0, 1.0],   // eq: x1 + x2 = 1
                [1.0, 0.0],   // x1 <= 0.8
                [0.0, 1.0],   // x2 <= 0.8
                [-1.0, 0.0],  // x1 >= 0
                [0.0, -1.0],  // x2 >= 0
            ]),
            b: vec![1.0, 0.8, 0.8, 0.0, 0.0],
            cones: vec![
                SupportedConeT::ZeroConeT(1),
                SupportedConeT::NonnegativeConeT(4),
            ],
            n_assets: 2,
            n_aux: 0,
        }
    }

    fn osqp_settings() -> SolverSettings {
        SolverSettings::default_for(Backend::Osqp)
    }

    #[test]
    fn test_cones_to_bounds() {
        let problem = fixture();
        let (l, u) = cones_to_bounds(&problem).unwrap();
        assert_eq!(u, problem.b);
        assert_eq!(l[0], 1.0); // equality row: l = u = b
        for &v in &l[1..] {
            assert_eq!(v, -OSQP_INFTY);
        }
    }

    #[test]
    fn test_parity_with_clarabel() {
        let problem = fixture();
        let cl = solve_qp_with(&problem, &SolverSettings::default(), Backend::Clarabel, None)
            .unwrap();
        let os = solve_qp_with(&problem, &osqp_settings(), Backend::Osqp, None).unwrap();

        assert_eq!(os.status, SolveStatus::Optimal);
        for (a, b) in cl.x.iter().zip(&os.x) {
            assert!((a - b).abs() < 1e-5, "x differs: {a} vs {b}");
        }
        assert!((cl.obj_val - os.obj_val).abs() < 1e-5);
        // Dual sign conventions must agree (active x2 <= 0.8 row: z > 0)
        for (a, b) in cl.z.iter().zip(&os.z) {
            assert!((a - b).abs() < 1e-4, "z differs: {a} vs {b}");
        }
        assert!(cl.z[2] > 1e-3, "expected active inequality dual");
    }

    #[test]
    fn test_warm_start_reduces_iterations() {
        let problem = fixture();
        let settings = osqp_settings();
        let first = solve_qp_with(&problem, &settings, Backend::Osqp, None).unwrap();

        // Perturb the linear cost by ~1%
        let mut perturbed = fixture();
        for v in &mut perturbed.q {
            *v *= 1.01;
        }
        let cold = solve_qp_with(&perturbed, &settings, Backend::Osqp, None).unwrap();
        let ws = WarmStart {
            x: first.x.clone(),
            y: first.z.clone(),
        };
        let warm = solve_qp_with(&perturbed, &settings, Backend::Osqp, Some(&ws)).unwrap();

        assert_eq!(warm.status, SolveStatus::Optimal);
        assert!(
            warm.iterations < cold.iterations,
            "warm ({}) should beat cold ({})",
            warm.iterations,
            cold.iterations
        );
    }

    #[test]
    fn test_warm_start_dimension_mismatch() {
        let problem = fixture();
        let ws = WarmStart {
            x: vec![0.0; 5], // wrong: problem has 2 vars
            y: vec![],
        };
        let result = solve_qp_with(&problem, &osqp_settings(), Backend::Osqp, Some(&ws));
        assert!(matches!(result, Err(SolverError::WarmStart(_))));
    }

    #[test]
    fn test_clarabel_ignores_warm_start() {
        let problem = fixture();
        let settings = SolverSettings::default();
        let plain = solve_qp_with(&problem, &settings, Backend::Clarabel, None).unwrap();
        // Even a mis-sized hint is fine: Clarabel discards it.
        let ws = WarmStart {
            x: vec![0.0; 99],
            y: vec![],
        };
        let hinted = solve_qp_with(&problem, &settings, Backend::Clarabel, Some(&ws)).unwrap();
        assert_eq!(plain.x, hinted.x);
        assert_eq!(plain.iterations, hinted.iterations);
    }

    #[test]
    fn test_infeasible_returns_nan_full_length() {
        // x1 + x2 = 1 and x1 + x2 <= -1: infeasible
        let problem = CompiledProblem {
            p: CscMatrix::from(&[[2.0, 0.0], [0.0, 2.0]]),
            q: vec![0.0, 0.0],
            a: CscMatrix::from(&[[1.0, 1.0], [1.0, 1.0]]),
            b: vec![1.0, -1.0],
            cones: vec![
                SupportedConeT::ZeroConeT(1),
                SupportedConeT::NonnegativeConeT(1),
            ],
            n_assets: 2,
            n_aux: 0,
        };
        let result = solve_qp_with(&problem, &osqp_settings(), Backend::Osqp, None).unwrap();
        assert_eq!(result.status, SolveStatus::Infeasible);
        assert_eq!(result.x.len(), 2);
        assert!(result.x.iter().all(|v| v.is_nan()));
    }
}
