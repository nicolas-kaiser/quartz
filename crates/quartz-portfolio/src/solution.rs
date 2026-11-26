use quartz_core::AssetId;

use crate::normalization::DimensionRange;
use std::collections::HashMap;

pub use quartz_solver::SolveStatus;

/// The enriched solution of a portfolio optimization.
#[derive(Debug, Clone)]
pub struct PortfolioSolution {
    /// Solver status.
    pub status: SolveStatus,
    /// Weight of every asset, in universe order (excluded and zero-weight
    /// assets included).
    pub weights: Vec<(AssetId, f64)>,
    /// All weights as a dense vector (ordered by universe index).
    pub weights_vec: Vec<f64>,
    /// Portfolio-level scores for each dimension.
    pub portfolio_scores: HashMap<String, f64>,
    /// Objective function value: `Σ weight · sign · value` over the
    /// dimensions, each value divided by its range when `dimension_ranges`
    /// is not empty.
    pub objective_value: f64,
    /// Solve time in seconds.
    pub solve_time_s: f64,
    /// Number of solver iterations.
    pub iterations: u32,
    /// Full primal vector `[w, t?, y_factor?]` including auxiliary variables.
    /// Feed into `PortfolioModel::warm_start` to seed the next solve.
    pub raw_x: Vec<f64>,
    /// Full dual vector (same sign convention on all backends).
    pub raw_z: Vec<f64>,
    /// The feasible range each dimension was divided by, in strategy order.
    /// Empty when the weights were applied raw (`Normalization::Raw`, fewer
    /// than two weighted dimensions, or an infeasible problem).
    pub dimension_ranges: Vec<DimensionRange>,
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
