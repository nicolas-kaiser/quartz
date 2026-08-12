use std::collections::HashMap;

use quartz_core::{DimensionType, Universe};
use quartz_solver::{self, Backend, SolverSettings, WarmStart};

use crate::compiler::{self, CompileError};
use crate::constraints::TurnoverConstraint;
use crate::restriction::Restrictions;
use crate::solution::PortfolioSolution;
use crate::strategy::Strategy;
use crate::tactic::Tactic;

/// Error from the portfolio model.
#[derive(Debug)]
pub enum PortfolioError {
    Compile(CompileError),
    Solver(quartz_solver::SolverError),
}

impl std::fmt::Display for PortfolioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortfolioError::Compile(e) => write!(f, "Compilation error: {e}"),
            PortfolioError::Solver(e) => write!(f, "Solver error: {e}"),
        }
    }
}

impl std::error::Error for PortfolioError {}

impl From<CompileError> for PortfolioError {
    fn from(e: CompileError) -> Self {
        PortfolioError::Compile(e)
    }
}

impl From<quartz_solver::SolverError> for PortfolioError {
    fn from(e: quartz_solver::SolverError) -> Self {
        PortfolioError::Solver(e)
    }
}

/// The main entry point for portfolio optimization.
///
/// # Example
/// ```ignore
/// let solution = PortfolioModel::new(&universe)
///     .strategy(&strategy)
///     .restrictions(&restrictions)
///     .solve()?;
/// ```
pub struct PortfolioModel<'a> {
    universe: &'a Universe,
    strategy: Option<&'a Strategy>,
    tactic: Option<&'a Tactic>,
    restrictions: Restrictions,
    turnover: Option<TurnoverConstraint>,
    solver_settings: SolverSettings,
    backend: Backend,
    warm_start: Option<WarmStart>,
}

impl<'a> PortfolioModel<'a> {
    pub fn new(universe: &'a Universe) -> Self {
        Self {
            universe,
            strategy: None,
            tactic: None,
            restrictions: Restrictions::default(),
            turnover: None,
            solver_settings: SolverSettings::default(),
            backend: Backend::default(),
            warm_start: None,
        }
    }

    pub fn strategy(mut self, strategy: &'a Strategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    pub fn tactic(mut self, tactic: &'a Tactic) -> Self {
        self.tactic = Some(tactic);
        self
    }

    pub fn restrictions(mut self, restrictions: Restrictions) -> Self {
        self.restrictions = restrictions;
        self
    }

    pub fn turnover(mut self, previous_weights: Vec<f64>, max_turnover: f64) -> Self {
        self.turnover = Some(TurnoverConstraint::new(previous_weights, max_turnover));
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.solver_settings.verbose = verbose;
        self
    }

    pub fn solver_settings(mut self, settings: SolverSettings) -> Self {
        self.solver_settings = settings;
        self
    }

    /// Select the QP solver backend. When using OSQP, pair with
    /// `SolverSettings::default_for(Backend::Osqp)` — Clarabel-tuned settings
    /// (max_iter 200, tol 1e-8) usually end in `MaxIterations` under ADMM.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Seed the solver with a previous solution of a structurally identical
    /// problem (same universe size, same turnover/factor setup). Only OSQP
    /// can exploit the hint; Clarabel accepts and ignores it.
    pub fn warm_start(mut self, previous: &PortfolioSolution) -> Self {
        self.warm_start = Some(WarmStart {
            x: previous.raw_x.clone(),
            y: previous.raw_z.clone(),
        });
        self
    }

    /// Compile, solve, and return the enriched portfolio solution.
    pub fn solve(self) -> Result<PortfolioSolution, PortfolioError> {
        let strategy = self.strategy.ok_or(PortfolioError::Compile(
            CompileError::NoDimensions,
        ))?;

        // Compile to QP
        let problem = compiler::compile(
            self.universe,
            strategy,
            self.tactic,
            &self.restrictions,
            self.turnover.as_ref(),
        )?;

        let n = problem.n_assets;

        // Solve
        let raw = quartz_solver::solve_qp_with(
            &problem,
            &self.solver_settings,
            self.backend,
            self.warm_start.as_ref(),
        )?;

        // Extract asset weights (first n entries)
        let weights_vec: Vec<f64> = raw.x[..n].to_vec();
        let weights: Vec<_> = self
            .universe
            .assets
            .iter()
            .zip(weights_vec.iter())
            .map(|(a, &w)| (a.id.clone(), w))
            .collect();

        // Compute portfolio scores for all known score keys
        let mut portfolio_scores = HashMap::new();
        let mut all_score_keys: Vec<String> = self
            .universe
            .assets
            .iter()
            .flat_map(|a| a.scores.keys().cloned())
            .collect();
        all_score_keys.sort();
        all_score_keys.dedup();

        for key in &all_score_keys {
            let score = self.universe.portfolio_score(&weights_vec, key);
            portfolio_scores.insert(key.clone(), score);
        }

        // Add financial_risk (portfolio variance) if quadratic dimension exists
        if strategy
            .dimensions
            .iter()
            .any(|d| matches!(d.dim_type, DimensionType::Quadratic))
        {
            let variance = compute_portfolio_variance(self.universe, &weights_vec);
            portfolio_scores.insert("financial_risk".to_string(), variance);
        }

        // Ex-post risk-constraint values, gated on constraint presence only
        // (unlike financial_risk, which is gated on the quadratic dimension)
        if let Some(te) = &strategy.tracking_error {
            let diff: Vec<f64> = weights_vec
                .iter()
                .zip(&te.benchmark_weights)
                .map(|(w, b)| w - b)
                .collect();
            let te_val = compute_portfolio_variance(self.universe, &diff).max(0.0).sqrt();
            portfolio_scores.insert("tracking_error".to_string(), te_val);
        }
        if let Some(cvar) = &strategy.cvar {
            if let Some(scenarios) = &self.universe.scenarios {
                portfolio_scores.insert(
                    "cvar".to_string(),
                    compute_cvar(scenarios, &weights_vec, cvar.alpha),
                );
            }
        }

        Ok(PortfolioSolution {
            status: raw.status,
            weights,
            weights_vec,
            portfolio_scores,
            objective_value: raw.obj_val,
            solve_time_s: raw.solve_time_s,
            iterations: raw.iterations,
            raw_x: raw.x,
            raw_z: raw.z,
        })
    }
}

/// Exact discrete CVaR at level alpha: the mean loss over the worst (1−α)
/// tail, with fractional-tail interpolation — this matches the value the
/// Rockafellar–Uryasev constraint bounds at the optimum (a plain "mean of the
/// worst ⌈m⌉ scenarios" would not, for fractional m).
fn compute_cvar(scenarios: &[Vec<f64>], weights: &[f64], alpha: f64) -> f64 {
    let mut losses: Vec<f64> = scenarios
        .iter()
        .map(|r| -r.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>())
        .collect();
    losses.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let m = ((1.0 - alpha) * losses.len() as f64).min(losses.len() as f64);
    if m <= 0.0 {
        return losses[0];
    }
    let full = m.floor() as usize;
    let mut total: f64 = losses[..full].iter().sum();
    let frac = m - full as f64;
    if frac > 0.0 && full < losses.len() {
        total += frac * losses[full];
    }
    total / m
}

/// Compute wᵀΣw for the full covariance model.
fn compute_portfolio_variance(universe: &Universe, weights: &[f64]) -> f64 {
    match &universe.covariance {
        quartz_core::CovarianceModel::Full(cov) => {
            // Compute wᵀΣw via CSC traversal
            let n = weights.len();
            let mut result = 0.0;
            for j in 0..n {
                let col_start = cov.colptr[j];
                let col_end = cov.colptr[j + 1];
                for idx in col_start..col_end {
                    let i = cov.rowval[idx];
                    let v = cov.nzval[idx];
                    result += weights[i] * v * weights[j];
                }
            }
            result
        }
        quartz_core::CovarianceModel::Factor {
            loadings,
            factor_cov,
            specific_variance,
        } => {
            // wᵀΣw = vᵀFv + Σ dᵢwᵢ²  with v = Bᵀw
            let k = loadings.n;
            let mut v = vec![0.0; k];
            for j in 0..k {
                for idx in loadings.colptr[j]..loadings.colptr[j + 1] {
                    v[j] += loadings.nzval[idx] * weights[loadings.rowval[idx]];
                }
            }
            let mut result = 0.0;
            for j in 0..k {
                for idx in factor_cov.colptr[j]..factor_cov.colptr[j + 1] {
                    result += v[factor_cov.rowval[idx]] * factor_cov.nzval[idx] * v[j];
                }
            }
            result
                + weights
                    .iter()
                    .zip(specific_variance)
                    .map(|(w, d)| d * w * w)
                    .sum::<f64>()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::CscMatrix;
    use quartz_core::Asset;

    // 3 assets, 2 factors — shared fixture
    const B: [[f64; 2]; 3] = [[1.0, 0.2], [0.8, -0.1], [0.5, 0.7]];
    const F: [[f64; 2]; 2] = [[0.04, 0.01], [0.01, 0.02]];
    const D: [f64; 3] = [0.01, 0.02, 0.015];

    /// Σ = BFBᵀ + D, densified with plain loops.
    fn densified_sigma() -> [[f64; 3]; 3] {
        let mut sigma = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for r in 0..2 {
                    for c in 0..2 {
                        sigma[i][j] += B[i][r] * F[r][c] * B[j][c];
                    }
                }
            }
            sigma[i][i] += D[i];
        }
        sigma
    }

    fn assets() -> Vec<Asset> {
        vec![
            Asset::new("A").score("expected_return", 0.10),
            Asset::new("B").score("expected_return", 0.05),
            Asset::new("C").score("expected_return", 0.03),
        ]
    }

    fn factor_universe() -> Universe {
        Universe::builder()
            .assets(assets())
            .covariance_factor(CscMatrix::from(&B), CscMatrix::from(&F), D.to_vec())
            .build()
            .unwrap()
    }

    fn dense_universe() -> Universe {
        Universe::builder()
            .assets(assets())
            .covariance_full(CscMatrix::from(&densified_sigma()))
            .build()
            .unwrap()
    }

    #[test]
    fn test_factor_solve_matches_densified() {
        let strategy = Strategy::builder("MinVar")
            .minimize_risk(0.7)
            .maximize("expected_return", 0.3)
            .build();
        let restrictions = || Restrictions::builder().long_only().fully_invested().build();

        let factor_uni = factor_universe();
        let dense_uni = dense_universe();
        let sol_factor = PortfolioModel::new(&factor_uni)
            .strategy(&strategy)
            .restrictions(restrictions())
            .solve()
            .unwrap();
        let sol_dense = PortfolioModel::new(&dense_uni)
            .strategy(&strategy)
            .restrictions(restrictions())
            .solve()
            .unwrap();

        assert_eq!(sol_factor.status, crate::solution::SolveStatus::Optimal);
        assert_eq!(sol_dense.status, crate::solution::SolveStatus::Optimal);
        for (wf, wd) in sol_factor.weights_vec.iter().zip(&sol_dense.weights_vec) {
            assert!((wf - wd).abs() < 1e-6, "weights differ: {wf} vs {wd}");
        }
        assert!((sol_factor.objective_value - sol_dense.objective_value).abs() < 1e-6);
        // Reported variance must also agree across the two covariance models
        let vf = sol_factor.variance().unwrap();
        let vd = sol_dense.variance().unwrap();
        assert!((vf - vd).abs() < 1e-8, "variance differs: {vf} vs {vd}");
    }

    #[test]
    fn test_factor_solve_with_turnover() {
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let previous = vec![0.4, 0.3, 0.3];
        let max_turnover = 0.10;

        let universe = factor_universe();
        let solution = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .turnover(previous.clone(), max_turnover)
            .solve()
            .unwrap();

        assert_eq!(solution.status, crate::solution::SolveStatus::Optimal);
        let turnover: f64 = solution
            .weights_vec
            .iter()
            .zip(&previous)
            .map(|(w, p)| (w - p).abs())
            .sum();
        assert!(turnover <= max_turnover + 1e-6, "turnover {turnover} exceeds budget");
    }

    #[cfg(feature = "osqp")]
    #[test]
    fn test_osqp_matches_clarabel() {
        let strategy = Strategy::builder("Mix")
            .minimize_risk(0.7)
            .maximize("expected_return", 0.3)
            .build();
        let restrictions = || Restrictions::builder().long_only().fully_invested().build();

        for universe in [dense_universe(), factor_universe()] {
            let cl = PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(restrictions())
                .solve()
                .unwrap();
            let os = PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(restrictions())
                .backend(Backend::Osqp)
                .solver_settings(SolverSettings::default_for(Backend::Osqp))
                .solve()
                .unwrap();
            assert_eq!(os.status, crate::solution::SolveStatus::Optimal);
            for (a, b) in cl.weights_vec.iter().zip(&os.weights_vec) {
                assert!((a - b).abs() < 1e-5, "weights differ: {a} vs {b}");
            }
        }
    }

    #[cfg(feature = "osqp")]
    #[test]
    fn test_osqp_warm_start_round_trip_with_turnover() {
        // Warm-starting through PortfolioSolution must round-trip the full
        // [w, t] aux layout and reduce iterations on a perturbed re-solve.
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let restrictions = || Restrictions::builder().long_only().fully_invested().build();
        let settings = SolverSettings::default_for(Backend::Osqp);
        let previous = vec![0.4, 0.3, 0.3];

        let day1 = PortfolioModel::new(&dense_universe())
            .strategy(&strategy)
            .restrictions(restrictions())
            .turnover(previous.clone(), 0.15)
            .backend(Backend::Osqp)
            .solver_settings(settings.clone())
            .solve()
            .unwrap();
        // raw_x covers [w (3), t (3)]
        assert_eq!(day1.raw_x.len(), 6);

        // Day 2: slightly perturbed universe (same structure)
        let mut assets2 = assets();
        assets2[0] = Asset::new("A").score("expected_return", 0.11);
        let universe2 = Universe::builder()
            .assets(assets2)
            .covariance_full(CscMatrix::from(&densified_sigma()))
            .build()
            .unwrap();

        let solve_day2 = |warm: Option<&PortfolioSolution>| {
            let mut model = PortfolioModel::new(&universe2)
                .strategy(&strategy)
                .restrictions(restrictions())
                .turnover(day1.weights_vec.clone(), 0.15)
                .backend(Backend::Osqp)
                .solver_settings(settings.clone());
            if let Some(prev) = warm {
                model = model.warm_start(prev);
            }
            model.solve().unwrap()
        };
        let cold = solve_day2(None);
        let warm = solve_day2(Some(&day1));

        assert_eq!(warm.status, crate::solution::SolveStatus::Optimal);
        assert!(
            warm.iterations <= cold.iterations,
            "warm ({}) should not exceed cold ({})",
            warm.iterations,
            cold.iterations
        );
        for (a, b) in cold.weights_vec.iter().zip(&warm.weights_vec) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    fn te_universe() -> Universe {
        // 2 assets, diagonal Σ; A low return, B high return
        Universe::builder()
            .add_asset(Asset::new("A").score("expected_return", 0.02))
            .add_asset(Asset::new("B").score("expected_return", 0.10))
            .covariance_full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]]))
            .build()
            .unwrap()
    }

    #[test]
    fn test_tracking_error_analytic() {
        // Benchmark = 100% A; fully invested ⇒ TE² = 0.05·(1−w_A)².
        // Max-return objective favors B, so TE ≤ 0.05 binds:
        // w_A = 1 − 0.05/√0.05 ≈ 0.77639
        let universe = te_universe();
        let strategy = Strategy::builder("TE")
            .maximize("expected_return", 1.0)
            .max_tracking_error(vec![1.0, 0.0], 0.05)
            .build();
        let solution = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();

        assert_eq!(solution.status, crate::solution::SolveStatus::Optimal);
        let expected_wa = 1.0 - 0.05 / 0.05_f64.sqrt();
        assert!(
            (solution.weights_vec[0] - expected_wa).abs() < 1e-5,
            "w_A = {}, expected {expected_wa}",
            solution.weights_vec[0]
        );
        let te = solution.portfolio_scores["tracking_error"];
        assert!(te <= 0.05 + 1e-6, "TE {te} exceeds limit");
        assert!(te >= 0.05 - 1e-4, "TE should bind, got {te}");
    }

    #[test]
    fn test_tracking_error_factor_matches_densified() {
        let strategy = Strategy::builder("TE")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .max_tracking_error(vec![0.4, 0.3, 0.3], 0.03)
            .build();
        let restrictions = || Restrictions::builder().long_only().fully_invested().build();

        let factor_uni = factor_universe();
        let dense_uni = dense_universe();
        let sol_f = PortfolioModel::new(&factor_uni)
            .strategy(&strategy)
            .restrictions(restrictions())
            .solve()
            .unwrap();
        let sol_d = PortfolioModel::new(&dense_uni)
            .strategy(&strategy)
            .restrictions(restrictions())
            .solve()
            .unwrap();

        assert_eq!(sol_f.status, crate::solution::SolveStatus::Optimal);
        for (wf, wd) in sol_f.weights_vec.iter().zip(&sol_d.weights_vec) {
            assert!((wf - wd).abs() < 1e-5, "weights differ: {wf} vs {wd}");
        }
        assert!(
            (sol_f.portfolio_scores["tracking_error"] - sol_d.portfolio_scores["tracking_error"])
                .abs()
                < 1e-6
        );
    }

    fn cvar_universe() -> Universe {
        Universe::builder()
            .add_asset(Asset::new("A").score("expected_return", 0.10))
            .add_asset(Asset::new("B").score("expected_return", 0.01))
            .covariance_full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]]))
            .scenarios(vec![
                vec![0.10, 0.01],
                vec![0.05, 0.01],
                vec![-0.20, 0.01],
                vec![0.02, 0.01],
            ])
            .build()
            .unwrap()
    }

    #[test]
    fn test_cvar_analytic() {
        // α = 0.75, S = 4 ⇒ (1−α)S = 1 ⇒ CVaR = worst single scenario loss
        // = 0.20·w_A − 0.01·w_B = 0.21·w_A − 0.01. Limit 0.05 ⇒ w_A ≤ 0.06/0.21.
        // Max-return objective favors A, so the limit binds.
        let universe = cvar_universe();
        let strategy = Strategy::builder("CVaR")
            .maximize("expected_return", 1.0)
            .max_cvar(0.75, 0.05)
            .build();
        let solution = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();

        assert_eq!(solution.status, crate::solution::SolveStatus::Optimal);
        let expected_wa = 0.06 / 0.21;
        assert!(
            (solution.weights_vec[0] - expected_wa).abs() < 1e-5,
            "w_A = {}, expected {expected_wa}",
            solution.weights_vec[0]
        );
        let cvar = solution.portfolio_scores["cvar"];
        assert!((cvar - 0.05).abs() < 1e-5, "CVaR should bind at 0.05, got {cvar}");
    }

    #[test]
    fn test_compute_cvar_fractional_tail() {
        // Weights (1, 0): losses = -r_A = [0.20, 0.03, -0.01, -0.05, -0.10]
        // sorted desc. α = 0.7, S = 5 ⇒ m = 1.5 ⇒ CVaR = (0.20 + 0.5·0.03)/1.5
        let scenarios = vec![
            vec![-0.20, 0.0],
            vec![-0.03, 0.0],
            vec![0.01, 0.0],
            vec![0.05, 0.0],
            vec![0.10, 0.0],
        ];
        let cvar = compute_cvar(&scenarios, &[1.0, 0.0], 0.7);
        let expected = (0.20 + 0.5 * 0.03) / 1.5;
        assert!((cvar - expected).abs() < 1e-12, "{cvar} vs {expected}");
    }

    #[test]
    fn test_risk_constraints_infeasible_when_impossible() {
        let universe = cvar_universe();
        let strategy = Strategy::builder("Impossible")
            .maximize("expected_return", 1.0)
            .max_cvar(0.75, -10.0)
            .build();
        let solution = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();
        assert_eq!(solution.status, crate::solution::SolveStatus::Infeasible);
    }

    #[cfg(feature = "osqp")]
    #[test]
    fn test_osqp_rejects_tracking_error() {
        let universe = te_universe();
        let strategy = Strategy::builder("TE")
            .maximize("expected_return", 1.0)
            .max_tracking_error(vec![1.0, 0.0], 0.05)
            .build();
        let result = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .backend(Backend::Osqp)
            .solver_settings(SolverSettings::default_for(Backend::Osqp))
            .solve();
        assert!(matches!(
            result,
            Err(PortfolioError::Solver(quartz_solver::SolverError::Unsupported(_)))
        ));
    }

    #[cfg(feature = "osqp")]
    #[test]
    fn test_osqp_cvar_matches_clarabel() {
        let universe = cvar_universe();
        let strategy = Strategy::builder("CVaR")
            .maximize("expected_return", 1.0)
            .max_cvar(0.75, 0.05)
            .build();
        let restrictions = || Restrictions::builder().long_only().fully_invested().build();
        let cl = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(restrictions())
            .solve()
            .unwrap();
        let os = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(restrictions())
            .backend(Backend::Osqp)
            .solver_settings(SolverSettings::default_for(Backend::Osqp))
            .solve()
            .unwrap();
        for (a, b) in cl.weights_vec.iter().zip(&os.weights_vec) {
            assert!((a - b).abs() < 1e-4, "weights differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_frontier_respects_risk_constraints() {
        let universe = cvar_universe();
        let strategy = Strategy::builder("Both")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .max_cvar(0.75, 0.05)
            .max_tracking_error(vec![0.5, 0.5], 0.10)
            .build();
        let result = crate::FrontierExplorer::new(&universe, &strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .sweep("expected_return", "financial_risk", 7)
            .unwrap();
        for p in &result.points {
            assert!(p.portfolio_scores["cvar"] <= 0.05 + 1e-5);
            assert!(p.portfolio_scores["tracking_error"] <= 0.10 + 1e-5);
        }
    }

    #[test]
    fn test_factor_variance_computation() {
        let w = [0.5, 0.3, 0.2];

        // Hand computation: wᵀ(BFBᵀ + D)w via the densified matrix
        let sigma = densified_sigma();
        let mut expected = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                expected += w[i] * sigma[i][j] * w[j];
            }
        }

        let vf = compute_portfolio_variance(&factor_universe(), &w);
        let vd = compute_portfolio_variance(&dense_universe(), &w);
        assert!((vf - expected).abs() < 1e-12, "factor arm: {vf} vs {expected}");
        assert!((vd - expected).abs() < 1e-12, "full arm: {vd} vs {expected}");
    }
}
