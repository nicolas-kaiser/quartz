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
        quartz_core::CovarianceModel::Factor { .. } => {
            // TODO: factor model variance computation
            0.0
        }
    }
}
