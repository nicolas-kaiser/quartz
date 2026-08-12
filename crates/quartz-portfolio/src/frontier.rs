//! Pareto frontier exploration: re-solve the portfolio across a family of
//! dimension-weight combinations and report the non-dominated trade-off surface.
//!
//! Two generation modes:
//! - [`FrontierExplorer::sweep`] trades two dimensions against each other while
//!   holding all other dimension weights fixed.
//! - [`FrontierExplorer::simplex_grid`] enumerates a full lattice over all of
//!   the strategy's dimensions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use quartz_core::{AssetId, Dimension, DimensionType, Universe};

use crate::model::{PortfolioError, PortfolioModel};
use crate::restriction::Restrictions;
use crate::solution::SolveStatus;
use crate::strategy::Strategy;
use crate::tactic::{self, MergeError, MergedStrategy, Tactic};

/// Hard cap on the number of solves per exploration.
pub const MAX_POINTS: usize = 10_000;

/// Errors from frontier exploration.
#[derive(Debug)]
pub enum FrontierError {
    UnknownDimension(String),
    SameDimension(String),
    /// The simplex grid needs at least two dimensions to explore.
    TooFewDimensions { found: usize },
    /// Sweep needs n_points >= 2; grid needs resolution >= 1.
    TooFewPoints { requested: usize },
    TooManyPoints { requested: usize, max: usize },
    /// The two swept dimensions carry no weight in the base strategy.
    ZeroPairWeight { dim_a: String, dim_b: String },
    /// Every explored point was infeasible or otherwise non-optimal.
    NoFeasiblePoints { attempted: usize },
    Merge(MergeError),
    Portfolio(PortfolioError),
}

impl std::fmt::Display for FrontierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrontierError::UnknownDimension(name) => {
                write!(f, "Dimension '{name}' not found in strategy")
            }
            FrontierError::SameDimension(name) => {
                write!(f, "Cannot sweep dimension '{name}' against itself")
            }
            FrontierError::TooFewDimensions { found } => {
                write!(f, "Grid exploration needs at least 2 dimensions, found {found}")
            }
            FrontierError::TooFewPoints { requested } => {
                write!(f, "Too few points requested ({requested})")
            }
            FrontierError::TooManyPoints { requested, max } => {
                write!(f, "Requested {requested} points, maximum is {max}")
            }
            FrontierError::ZeroPairWeight { dim_a, dim_b } => {
                write!(f, "Dimensions '{dim_a}' and '{dim_b}' have zero combined weight")
            }
            FrontierError::NoFeasiblePoints { attempted } => {
                write!(f, "All {attempted} explored points were infeasible or non-optimal")
            }
            FrontierError::Merge(e) => write!(f, "Tactic merge error: {e}"),
            FrontierError::Portfolio(e) => write!(f, "Portfolio error: {e}"),
        }
    }
}

impl std::error::Error for FrontierError {}

impl From<PortfolioError> for FrontierError {
    fn from(e: PortfolioError) -> Self {
        FrontierError::Portfolio(e)
    }
}

impl From<MergeError> for FrontierError {
    fn from(e: MergeError) -> Self {
        FrontierError::Merge(e)
    }
}

/// One solved point on the exploration path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierPoint {
    /// Dimension weights used for this solve (all dimensions, strategy order).
    pub dimension_weights: Vec<(String, f64)>,
    pub weights: Vec<(AssetId, f64)>,
    pub weights_vec: Vec<f64>,
    pub portfolio_scores: HashMap<String, f64>,
    pub objective_value: f64,
    /// True if no other point dominates this one over the run's objective dims.
    pub is_efficient: bool,
}

/// The result of a frontier exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierResult {
    /// Optimally solved points, in generation order.
    pub points: Vec<FrontierPoint>,
    /// Dimension names used for the dominance filter.
    pub objective_dims: Vec<String>,
    /// Number of explored points skipped because the solve was not optimal.
    pub n_skipped: usize,
}

/// Explores the Pareto frontier of a multi-dimensional strategy.
///
/// Mirrors [`PortfolioModel`]'s builder API; tactic, restrictions, and turnover
/// apply identically to every explored point.
pub struct FrontierExplorer<'a> {
    universe: &'a Universe,
    strategy: &'a Strategy,
    tactic: Option<&'a Tactic>,
    restrictions: Restrictions,
    turnover: Option<(Vec<f64>, f64)>,
}

impl<'a> FrontierExplorer<'a> {
    pub fn new(universe: &'a Universe, strategy: &'a Strategy) -> Self {
        Self {
            universe,
            strategy,
            tactic: None,
            restrictions: Restrictions::default(),
            turnover: None,
        }
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
        self.turnover = Some((previous_weights, max_turnover));
        self
    }

    /// Trade `dim_a` against `dim_b` over `n_points` steps.
    ///
    /// With s = weight(a) + weight(b) in the (normalized) base strategy, point i
    /// uses weight(a) = αᵢ·s and weight(b) = (1−αᵢ)·s with αᵢ = i/(n−1); all
    /// other dimension weights stay fixed, so the total remains 1 and the base
    /// strategy itself lies on the sweep at α = weight(a)/s.
    pub fn sweep(
        &self,
        dim_a: &str,
        dim_b: &str,
        n_points: usize,
    ) -> Result<FrontierResult, FrontierError> {
        if n_points < 2 {
            return Err(FrontierError::TooFewPoints { requested: n_points });
        }
        if n_points > MAX_POINTS {
            return Err(FrontierError::TooManyPoints {
                requested: n_points,
                max: MAX_POINTS,
            });
        }
        if dim_a == dim_b {
            return Err(FrontierError::SameDimension(dim_a.to_string()));
        }

        let base = self.resolve_base()?;
        let ia = find_dimension(&base.dimensions, dim_a)?;
        let ib = find_dimension(&base.dimensions, dim_b)?;
        let s = base.dimensions[ia].weight + base.dimensions[ib].weight;
        if s <= 0.0 {
            return Err(FrontierError::ZeroPairWeight {
                dim_a: dim_a.to_string(),
                dim_b: dim_b.to_string(),
            });
        }

        let mut points = Vec::new();
        let mut n_skipped = 0;
        for i in 0..n_points {
            let alpha = i as f64 / (n_points - 1) as f64;
            let mut dims = base.dimensions.clone();
            dims[ia].weight = alpha * s;
            dims[ib].weight = (1.0 - alpha) * s;
            match self.solve_point(&base, dims)? {
                Some(p) => points.push(p),
                None => n_skipped += 1,
            }
        }

        let objective_dims = vec![
            base.dimensions[ia].clone(),
            base.dimensions[ib].clone(),
        ];
        finish(points, &objective_dims, n_skipped, n_points)
    }

    /// Enumerate all weight combinations on a simplex lattice over every
    /// dimension of the strategy: compositions of `resolution` into m parts,
    /// C(resolution+m−1, m−1) points in total.
    pub fn simplex_grid(&self, resolution: usize) -> Result<FrontierResult, FrontierError> {
        if resolution < 1 {
            return Err(FrontierError::TooFewPoints { requested: resolution });
        }

        let base = self.resolve_base()?;
        let m = base.dimensions.len();
        if m < 2 {
            return Err(FrontierError::TooFewDimensions { found: m });
        }

        let n_points = n_compositions(resolution, m);
        if n_points > MAX_POINTS {
            return Err(FrontierError::TooManyPoints {
                requested: n_points,
                max: MAX_POINTS,
            });
        }

        let mut points = Vec::new();
        let mut n_skipped = 0;
        for comp in compositions(resolution, m) {
            let mut dims = base.dimensions.clone();
            for (d, &c) in dims.iter_mut().zip(&comp) {
                d.weight = c as f64 / resolution as f64;
            }
            match self.solve_point(&base, dims)? {
                Some(p) => points.push(p),
                None => n_skipped += 1,
            }
        }

        finish(points, &base.dimensions, n_skipped, n_points)
    }

    /// Merge the tactic once up front and normalize dimension weights.
    ///
    /// Per-point solves must NOT pass the tactic through: `tactic::merge`
    /// re-normalizes dimension weights and would silently distort the swept
    /// weights. Group/score overrides are baked into the merged result here.
    fn resolve_base(&self) -> Result<MergedStrategy, FrontierError> {
        let mut merged = tactic::merge(self.strategy, self.tactic)?;
        let total: f64 = merged.dimensions.iter().map(|d| d.weight).sum();
        if total > 0.0 && (total - 1.0).abs() > 1e-12 {
            for d in &mut merged.dimensions {
                d.weight /= total;
            }
        }
        Ok(merged)
    }

    /// Solve one weight combination. Returns Ok(None) for non-optimal statuses.
    fn solve_point(
        &self,
        base: &MergedStrategy,
        dims: Vec<Dimension>,
    ) -> Result<Option<FrontierPoint>, FrontierError> {
        let strategy = Strategy {
            name: self.strategy.name.clone(),
            dimensions: dims,
            group_constraints: base.group_constraints.clone(),
            score_constraints: base.score_constraints.clone(),
            fully_invested: base.fully_invested,
        };
        let mut model = PortfolioModel::new(self.universe)
            .strategy(&strategy)
            .restrictions(self.restrictions.clone());
        if let Some((prev, max)) = &self.turnover {
            model = model.turnover(prev.clone(), *max);
        }
        let sol = model.solve()?;
        if sol.status != SolveStatus::Optimal {
            return Ok(None);
        }
        Ok(Some(FrontierPoint {
            dimension_weights: strategy
                .dimensions
                .iter()
                .map(|d| (d.name.clone(), d.weight))
                .collect(),
            weights: sol.weights,
            weights_vec: sol.weights_vec,
            portfolio_scores: sol.portfolio_scores,
            objective_value: sol.objective_value,
            is_efficient: false,
        }))
    }
}

/// The portfolio_scores key a dimension is measured by.
fn metric_key(dim: &Dimension) -> &str {
    match &dim.dim_type {
        DimensionType::Quadratic => "financial_risk",
        DimensionType::Linear { score_key } => score_key,
    }
}

fn find_dimension(dims: &[Dimension], name: &str) -> Result<usize, FrontierError> {
    dims.iter()
        .position(|d| d.name == name)
        .ok_or_else(|| FrontierError::UnknownDimension(name.to_string()))
}

/// Apply the dominance filter and assemble the result.
fn finish(
    mut points: Vec<FrontierPoint>,
    objective_dims: &[Dimension],
    n_skipped: usize,
    attempted: usize,
) -> Result<FrontierResult, FrontierError> {
    if points.is_empty() {
        return Err(FrontierError::NoFeasiblePoints { attempted });
    }

    // Canonicalize each point to a minimization vector over the objective dims.
    let metrics: Vec<Vec<f64>> = points
        .iter()
        .map(|p| {
            objective_dims
                .iter()
                .map(|d| {
                    d.sense.sign() * p.portfolio_scores.get(metric_key(d)).copied().unwrap_or(0.0)
                })
                .collect()
        })
        .collect();
    let flags = pareto_flags(&metrics);
    for (p, flag) in points.iter_mut().zip(flags) {
        p.is_efficient = flag;
    }

    Ok(FrontierResult {
        points,
        objective_dims: objective_dims.iter().map(|d| d.name.clone()).collect(),
        n_skipped,
    })
}

/// Non-dominated flags over canonical minimization vectors.
///
/// A dominates B ⇔ A is no worse in every coordinate (within tolerance) and
/// strictly better in at least one. O(p²), fine at the MAX_POINTS cap.
pub fn pareto_flags(metrics: &[Vec<f64>]) -> Vec<bool> {
    let dominates = |a: &[f64], b: &[f64]| -> bool {
        let mut strictly_better = false;
        for (&ma, &mb) in a.iter().zip(b) {
            let eps = 1e-9 * ma.abs().max(mb.abs()).max(1.0);
            if ma > mb + eps {
                return false;
            }
            if ma < mb - eps {
                strictly_better = true;
            }
        }
        strictly_better
    };
    (0..metrics.len())
        .map(|i| {
            !metrics
                .iter()
                .enumerate()
                .any(|(j, m)| j != i && dominates(m, &metrics[i]))
        })
        .collect()
}

/// Number of compositions of g into m non-negative parts: C(g+m−1, m−1).
fn n_compositions(g: usize, m: usize) -> usize {
    // Multiplicative binomial, saturating well above MAX_POINTS.
    let mut result: usize = 1;
    for i in 0..(m - 1) {
        result = result.saturating_mul(g + m - 1 - i) / (i + 1);
        if result > 100 * MAX_POINTS {
            return result;
        }
    }
    result
}

/// All compositions of g into m non-negative parts, lexicographic.
fn compositions(g: usize, m: usize) -> Vec<Vec<usize>> {
    fn rec(g: usize, m: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if m == 1 {
            prefix.push(g);
            out.push(prefix.clone());
            prefix.pop();
            return;
        }
        for c in 0..=g {
            prefix.push(c);
            rec(g - c, m - 1, prefix, out);
            prefix.pop();
        }
    }
    let mut out = Vec::new();
    rec(g, m, &mut Vec::new(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::CscMatrix;
    use quartz_core::Asset;

    /// 2 assets, zero covariance. With w_A = x: return = 0.02 + 0.08x,
    /// variance = 0.04x² + 0.01(1−x)², minimized at x* = 0.2.
    fn two_asset_universe() -> Universe {
        Universe::builder()
            .add_asset(Asset::new("A").score("expected_return", 0.10).score("esg", 3.0))
            .add_asset(Asset::new("B").score("expected_return", 0.02).score("esg", 8.0))
            .covariance_full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]]))
            .build()
            .unwrap()
    }

    fn base_strategy() -> Strategy {
        Strategy::builder("Base")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .build()
    }

    fn base_restrictions() -> Restrictions {
        Restrictions::builder().long_only().fully_invested().build()
    }

    #[test]
    fn test_sweep_endpoints() {
        let universe = two_asset_universe();
        let strategy = base_strategy();
        let result = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 11)
            .unwrap();

        assert_eq!(result.points.len(), 11);
        assert_eq!(result.n_skipped, 0);
        // α=0: pure min-variance → x* = 0.2
        assert!((result.points[0].weights_vec[0] - 0.2).abs() < 1e-4);
        // α=1: pure max-return → all in A
        assert!((result.points[10].weights_vec[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sweep_monotone() {
        let universe = two_asset_universe();
        let strategy = base_strategy();
        let result = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 21)
            .unwrap();

        // Tolerance covers interior-point solver noise (~1e-7) at saturated corners
        for pair in result.points.windows(2) {
            let (prev, next) = (&pair[0], &pair[1]);
            assert!(
                next.portfolio_scores["expected_return"]
                    >= prev.portfolio_scores["expected_return"] - 1e-6
            );
            assert!(
                next.portfolio_scores["financial_risk"]
                    >= prev.portfolio_scores["financial_risk"] - 1e-6
            );
        }
    }

    #[test]
    fn test_sweep_all_efficient_and_base_on_sweep() {
        let universe = two_asset_universe();
        let strategy = base_strategy();
        let result = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 11)
            .unwrap();

        // Convex scalarization ⇒ every sweep point is Pareto-efficient
        assert!(result.points.iter().all(|p| p.is_efficient));

        // The base strategy (α = 0.5, point index 5) matches a direct solve
        let direct = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(base_restrictions())
            .solve()
            .unwrap();
        for (wf, wd) in result.points[5].weights_vec.iter().zip(&direct.weights_vec) {
            assert!((wf - wd).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pareto_flags_unit() {
        // Canonical minimization vectors: p1 and p2 trade off, p3 is dominated
        // by p1, p4 ties with p1 (within tolerance).
        let metrics = vec![
            vec![1.0, 5.0],
            vec![5.0, 1.0],
            vec![2.0, 6.0],
            vec![1.0 + 1e-12, 5.0 - 1e-12],
        ];
        let flags = pareto_flags(&metrics);
        assert_eq!(flags, vec![true, true, false, true]);
    }

    #[test]
    fn test_grid_point_count() {
        let universe = two_asset_universe();
        let strategy = Strategy::builder("3dim")
            .minimize_risk(0.4)
            .maximize("expected_return", 0.3)
            .maximize("esg", 0.3)
            .build();
        let result = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .simplex_grid(4)
            .unwrap();

        // C(4+3-1, 3-1) = C(6,2) = 15 compositions
        assert_eq!(result.points.len() + result.n_skipped, 15);
        for p in &result.points {
            let total: f64 = p.dimension_weights.iter().map(|(_, w)| w).sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_error_paths() {
        let universe = two_asset_universe();
        let strategy = base_strategy();
        let explorer = FrontierExplorer::new(&universe, &strategy);

        assert!(matches!(
            explorer.sweep("nope", "financial_risk", 5),
            Err(FrontierError::UnknownDimension(_))
        ));
        assert!(matches!(
            explorer.sweep("financial_risk", "financial_risk", 5),
            Err(FrontierError::SameDimension(_))
        ));
        assert!(matches!(
            explorer.sweep("expected_return", "financial_risk", 1),
            Err(FrontierError::TooFewPoints { .. })
        ));

        let one_dim = Strategy::builder("1dim").minimize_risk(1.0).build();
        let explorer = FrontierExplorer::new(&universe, &one_dim);
        assert!(matches!(
            explorer.simplex_grid(4),
            Err(FrontierError::TooFewDimensions { found: 1 })
        ));
    }

    #[test]
    fn test_no_feasible_points() {
        let universe = two_asset_universe();
        let strategy = Strategy::builder("Impossible")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .score_min("expected_return", 99.0)
            .build();
        let result = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 5);
        assert!(matches!(
            result,
            Err(FrontierError::NoFeasiblePoints { attempted: 5 })
        ));
    }

    #[test]
    fn test_unnormalized_base_weights() {
        let universe = two_asset_universe();
        // Bypass the builder's normalization: weights 2.0 / 2.0
        let raw = Strategy {
            name: "Raw".into(),
            dimensions: vec![
                Dimension::quadratic("financial_risk", quartz_core::Sense::Minimize, 2.0),
                Dimension::linear(
                    "expected_return",
                    "expected_return",
                    quartz_core::Sense::Maximize,
                    2.0,
                ),
            ],
            group_constraints: vec![],
            score_constraints: vec![],
            fully_invested: true,
        };
        let normalized = base_strategy();

        let sweep_raw = FrontierExplorer::new(&universe, &raw)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 7)
            .unwrap();
        let sweep_norm = FrontierExplorer::new(&universe, &normalized)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 7)
            .unwrap();

        for (pr, pn) in sweep_raw.points.iter().zip(&sweep_norm.points) {
            for (wr, wn) in pr.weights_vec.iter().zip(&pn.weights_vec) {
                assert!((wr - wn).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_tactic_weight_override_does_not_distort_sweep() {
        let universe = two_asset_universe();
        let strategy = base_strategy();
        // A tactic that changes the swept pair's balance must not affect the
        // sweep itself (both swept weights are replaced at every point).
        let tactic = Tactic::builder("Tilt")
            .override_weight("expected_return", 0.9)
            .build();

        let with_tactic = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .tactic(&tactic)
            .sweep("expected_return", "financial_risk", 7)
            .unwrap();
        let without = FrontierExplorer::new(&universe, &strategy)
            .restrictions(base_restrictions())
            .sweep("expected_return", "financial_risk", 7)
            .unwrap();

        for (pt, pn) in with_tactic.points.iter().zip(&without.points) {
            for (wt, wn) in pt.weights_vec.iter().zip(&pn.weights_vec) {
                assert!((wt - wn).abs() < 1e-6);
            }
        }
    }
}
