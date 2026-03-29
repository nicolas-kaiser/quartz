use quartz_core::AssetId;
use std::collections::HashMap;

pub use quartz_solver::SolveStatus;

/// The enriched solution of a portfolio optimization.
#[derive(Debug, Clone)]
pub struct PortfolioSolution {
    /// Solver status.
    pub status: SolveStatus,
    /// Optimal weights per asset (only non-excluded, non-zero weights shown).
    pub weights: Vec<(AssetId, f64)>,
    /// All weights as a dense vector (ordered by universe index).
    pub weights_vec: Vec<f64>,
    /// Portfolio-level scores for each dimension.
    pub portfolio_scores: HashMap<String, f64>,
    /// Objective function value.
    pub objective_value: f64,
    /// Solve time in seconds.
    pub solve_time_s: f64,
    /// Number of solver iterations.
    pub iterations: u32,
}

impl PortfolioSolution {
    /// Get the weight for a specific asset.
    pub fn weight(&self, id: &AssetId) -> Option<f64> {
        self.weights
            .iter()
            .find(|(aid, _)| aid == id)
            .map(|(_, w)| *w)
    }

    /// Get a portfolio score by dimension name.
    pub fn score(&self, key: &str) -> Option<f64> {
        self.portfolio_scores.get(key).copied()
    }

    /// Portfolio variance (if financial_risk is in scores).
    pub fn variance(&self) -> Option<f64> {
        self.portfolio_scores.get("financial_risk").copied()
    }

    /// Portfolio expected return (if expected_return is in scores).
    pub fn expected_return(&self) -> Option<f64> {
        self.portfolio_scores.get("expected_return").copied()
    }
}
