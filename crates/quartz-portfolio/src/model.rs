use std::collections::HashMap;

use quartz_core::{DimensionType, Universe};
use quartz_solver::{self, SolverSettings};

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
        let raw = quartz_solver::solve_qp(&problem, &self.solver_settings)?;

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

        Ok(PortfolioSolution {
            status: raw.status,
            weights,
            weights_vec,
            portfolio_scores,
            objective_value: raw.obj_val,
            solve_time_s: raw.solve_time_s,
            iterations: raw.iterations,
        })
    }
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
