use std::collections::HashMap;

use quartz_core::{DimensionType, Universe};
use quartz_solver::{self, Backend, SolverSettings, WarmStart};

use crate::compiler::{self, CompileError};
use crate::constraints::{TransactionCosts, TurnoverConstraint};
use crate::normalization::{self, DimensionRange};
use crate::restriction::Restrictions;
use crate::solution::{PortfolioSolution, SolveStatus};
use crate::strategy::{Normalization, Strategy};
use crate::tactic::{self, Tactic};

/// Error from the portfolio model.
#[derive(Debug)]
pub enum PortfolioError {
    Compile(CompileError),
    Solver(quartz_solver::SolverError),
    /// Range normalisation: the single-objective solve of `dimension` did not
    /// reach an optimum (typically it is unbounded on its own), so the
    /// dimension has no feasible range.
    Normalization {
        dimension: String,
        status: SolveStatus,
    },
    /// Ranges passed to `PortfolioModel::dimension_ranges` do not match the
    /// strategy's dimensions.
    DimensionRangeMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for PortfolioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortfolioError::Compile(e) => write!(f, "Compilation error: {e}"),
            PortfolioError::Solver(e) => write!(f, "Solver error: {e}"),
            PortfolioError::Normalization { dimension, status } => write!(
                f,
                "Cannot normalise dimension '{dimension}': optimising it alone ended with \
                 status {status:?}, so it has no feasible range. Bound the problem \
                 (e.g. long-only or a maximum weight) or set the strategy's \
                 normalization to raw"
            ),
            PortfolioError::DimensionRangeMismatch { expected, got } => write!(
                f,
                "Got {got} dimension ranges for a strategy with {expected} dimensions"
            ),
        }
    }
}

impl std::error::Error for PortfolioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PortfolioError::Compile(e) => Some(e),
            PortfolioError::Solver(e) => Some(e),
            _ => None,
        }
    }
}

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
    transaction_costs: Option<TransactionCosts>,
    solver_settings: SolverSettings,
    backend: Backend,
    warm_start: Option<WarmStart>,
    dimension_ranges: Option<Vec<DimensionRange>>,
}

impl<'a> PortfolioModel<'a> {
    #[must_use]
    pub fn new(universe: &'a Universe) -> Self {
        Self {
            universe,
            strategy: None,
            tactic: None,
            restrictions: Restrictions::default(),
            turnover: None,
            transaction_costs: None,
            solver_settings: SolverSettings::default(),
            backend: Backend::default(),
            warm_start: None,
            dimension_ranges: None,
        }
    }

    #[must_use]
    pub fn strategy(mut self, strategy: &'a Strategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    #[must_use]
    pub fn tactic(mut self, tactic: &'a Tactic) -> Self {
        self.tactic = Some(tactic);
        self
    }

    #[must_use]
    pub fn restrictions(mut self, restrictions: Restrictions) -> Self {
        self.restrictions = restrictions;
        self
    }

    #[must_use]
    pub fn turnover(mut self, previous_weights: Vec<f64>, max_turnover: f64) -> Self {
        self.turnover = Some(TurnoverConstraint::new(previous_weights, max_turnover));
        self
    }

    /// Transaction costs: `score_key` becomes net of `Σᵢ rates[i]·|wᵢ − wᵢ_prev|`
    /// wherever the strategy wants more of it (see [`TransactionCosts`]).
    /// Needs [`turnover`](Self::turnover) (in any order); checked at solve time.
    #[must_use]
    pub fn transaction_costs(mut self, rates: Vec<f64>, score_key: impl Into<String>) -> Self {
        self.transaction_costs = Some(TransactionCosts::new(rates, score_key));
        self
    }

    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.solver_settings.verbose = verbose;
        self
    }

    #[must_use]
    pub fn solver_settings(mut self, settings: SolverSettings) -> Self {
        self.solver_settings = settings;
        self
    }

    /// Select the QP solver backend. When using OSQP, pair with
    /// `SolverSettings::default_for(Backend::Osqp)` — Clarabel-tuned settings
    /// (max_iter 200, tol 1e-8) usually end in `MaxIterations` under ADMM.
    #[must_use]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Seed the solver with a previous solution of a structurally identical
    /// problem (same universe size, same turnover/factor setup). Only OSQP
    /// can exploit the hint; Clarabel accepts and ignores it.
    #[must_use]
    pub fn warm_start(self, previous: &PortfolioSolution) -> Self {
        self.warm_start_raw(previous.raw_x.clone(), previous.raw_z.clone())
    }

    /// Like [`warm_start`](Self::warm_start), but from raw primal/dual vectors
    /// (e.g. `PortfolioSolution::raw_x` / `raw_z` carried across a language
    /// boundary, where no `PortfolioSolution` exists).
    #[must_use]
    pub fn warm_start_raw(mut self, x: Vec<f64>, y: Vec<f64>) -> Self {
        self.warm_start = Some(WarmStart { x, y });
        self
    }

    /// Use these feasible ranges (one per dimension, strategy order) instead
    /// of computing them. The ranges depend on the universe and the
    /// constraints but not on the dimension weights, so one set serves every
    /// re-solve that only changes weights. Ignored under
    /// [`Normalization::Raw`].
    #[must_use]
    pub fn dimension_ranges(mut self, ranges: Vec<DimensionRange>) -> Self {
        self.dimension_ranges = Some(ranges);
        self
    }

    /// The feasible range of every dimension for this problem, including
    /// dimensions at weight 0, whatever the strategy's normalisation.
    /// `None` when there are fewer than two dimensions or the problem is
    /// infeasible. Costs one solve per dimension.
    pub fn compute_dimension_ranges(&self) -> Result<Option<Vec<DimensionRange>>, PortfolioError> {
        let strategy = self.strategy.ok_or(PortfolioError::Compile(
            CompileError::NoDimensions,
        ))?;
        let base = merged_strategy(strategy, self.tactic)?;
        normalization::dimension_ranges(
            self.universe,
            &base,
            &self.restrictions,
            self.turnover.as_ref(),
            self.transaction_costs.as_ref(),
            &self.solver_settings,
            self.backend,
            true,
        )
    }

    /// Compile to a QP. Under [`Normalization::Range`] this first measures
    /// each dimension's feasible range (one extra solve per weighted
    /// dimension, unless ranges were supplied) and divides by it.
    fn compile(
        &self,
        strategy: &Strategy,
    ) -> Result<(quartz_solver::CompiledProblem, Vec<DimensionRange>), PortfolioError> {
        let raw = |strategy: &Strategy, tactic| {
            compiler::compile_with_costs(
                self.universe,
                strategy,
                tactic,
                &self.restrictions,
                self.turnover.as_ref(),
                self.transaction_costs.as_ref(),
            )
        };
        if strategy.normalization == Normalization::Raw {
            return Ok((raw(strategy, self.tactic)?, Vec::new()));
        }

        // Fail on a malformed universe before any anchor solve does.
        self.universe.validate().map_err(CompileError::from)?;
        if strategy.dimensions.is_empty() {
            return Err(CompileError::NoDimensions.into());
        }

        // Merge the tactic once: the ranges belong to the merged constraints,
        // and the scales are indexed by the merged dimensions.
        let base = merged_strategy(strategy, self.tactic)?;
        // Fail once on a name that matches nothing, not in every anchor solve.
        compiler::validate_references(
            self.universe,
            &base.dimensions,
            &base.group_constraints,
            &base.score_constraints,
            &self.restrictions,
            self.transaction_costs.as_ref(),
        )?;
        let ranges = match &self.dimension_ranges {
            Some(r) => {
                if r.len() != base.dimensions.len() {
                    return Err(PortfolioError::DimensionRangeMismatch {
                        expected: base.dimensions.len(),
                        got: r.len(),
                    });
                }
                Some(r.clone())
            }
            None => normalization::dimension_ranges(
                self.universe,
                &base,
                &self.restrictions,
                self.turnover.as_ref(),
                self.transaction_costs.as_ref(),
                &self.solver_settings,
                self.backend,
                false,
            )?,
        };
        let Some(ranges) = ranges else {
            return Ok((raw(&base, None)?, Vec::new()));
        };
        let scales: Vec<f64> = ranges.iter().map(DimensionRange::scale).collect();
        let problem = compiler::compile_scaled(
            self.universe,
            &base,
            None,
            &self.restrictions,
            self.turnover.as_ref(),
            self.transaction_costs.as_ref(),
            Some(&scales),
        )?;
        Ok((problem, ranges))
    }

    /// Compile, solve, and return the enriched portfolio solution.
    pub fn solve(self) -> Result<PortfolioSolution, PortfolioError> {
        let strategy = self.strategy.ok_or(PortfolioError::Compile(
            CompileError::NoDimensions,
        ))?;

        let (problem, dimension_ranges) = self.compile(strategy)?;

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
            dimension_ranges,
        })
    }
}

/// The strategy with the tactic merged in, as a plain raw-weighted `Strategy`
/// (compile it with tactic `None`: the overrides are already applied).
fn merged_strategy(strategy: &Strategy, tactic: Option<&Tactic>) -> Result<Strategy, PortfolioError> {
    let merged = tactic::merge(strategy, tactic).map_err(CompileError::from)?;
    Ok(Strategy {
        name: strategy.name.clone(),
        dimensions: merged.dimensions,
        group_constraints: merged.group_constraints,
        score_constraints: merged.score_constraints,
        fully_invested: merged.fully_invested,
        tracking_error: merged.tracking_error,
        cvar: merged.cvar,
        normalization: Normalization::Raw,
    })
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
    losses.sort_by(|a, b| b.total_cmp(a));
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
        // (two different formulations, each solved to ~1e-8: the weights agree
        // to solver tolerance, the variance to about one order less)
        assert!((vf - vd).abs() < 1e-7, "variance differs: {vf} vs {vd}");
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

    // ---- transaction costs -------------------------------------------------

    use crate::solution::SolveStatus;

    /// Equal-variance universe: returns 10%, 5%, 3%, today 1/3 each.
    fn cost_universe() -> Universe {
        Universe::builder()
            .assets(assets())
            .covariance_full(CscMatrix::from(&[
                [0.04, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.04],
            ]))
            .build()
            .unwrap()
    }

    const PREV: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

    fn max_return(rates: Option<Vec<f64>>) -> Vec<f64> {
        let universe = cost_universe();
        let strategy = Strategy::builder("Max").maximize("expected_return", 1.0).build();
        let mut model = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build());
        if let Some(r) = rates {
            // costs before turnover: the order does not matter
            model = model.transaction_costs(r, "expected_return");
        }
        let sol = model.turnover(PREV.to_vec(), 2.0).solve().unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
        sol.weights_vec
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn test_costs_trade_only_where_the_gain_pays() {
        assert_close(&max_return(None), &[1.0, 0.0, 0.0]);
        // C→A gains 7% for 2×3% of costs, B→A 5% < 6%: B stays
        assert_close(&max_return(Some(vec![0.03; 3])), &[2.0 / 3.0, 1.0 / 3.0, 0.0]);
        // A is too dear to buy; C→B gains 2% for free
        assert_close(&max_return(Some(vec![0.3, 0.0, 0.0])), &[1.0 / 3.0, 2.0 / 3.0, 0.0]);
        // nothing pays: hold
        assert_close(&max_return(Some(vec![0.04; 3])), &PREV);
    }

    fn min_risk(min_return: Option<f64>, rates: Option<Vec<f64>>) -> Vec<f64> {
        let universe = cost_universe();
        let mut b = Strategy::builder("MinVar").minimize_risk(1.0);
        if let Some(r) = min_return {
            b = b.score_min("expected_return", r);
        }
        let strategy = b.build();
        let mut model = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .turnover(PREV.to_vec(), 2.0);
        if let Some(r) = rates {
            model = model.transaction_costs(r, "expected_return");
        }
        let sol = model.solve().unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
        sol.weights_vec
    }

    #[test]
    fn test_costs_do_not_move_the_minimum_risk_portfolio() {
        assert_close(&min_risk(None, Some(vec![0.05; 3])), &min_risk(None, None));
    }

    #[test]
    fn test_costs_make_a_return_target_net() {
        let rates = vec![0.02; 3];
        let net_return = |w: &[f64]| -> f64 {
            let mu = [0.10, 0.05, 0.03];
            (0..3).map(|i| mu[i] * w[i] - rates[i] * (w[i] - PREV[i]).abs()).sum()
        };
        let gross = min_risk(Some(0.07), None);
        let net = min_risk(Some(0.07), Some(rates.clone()));
        assert!(net_return(&gross) < 0.07 - 1e-4, "the gross answer misses the target net");
        assert!(net_return(&net) >= 0.07 - 1e-6, "net return {}", net_return(&net));
    }

    #[test]
    fn test_costs_without_turnover_fail_cleanly() {
        let universe = cost_universe();
        let strategy = Strategy::builder("Max").maximize("expected_return", 1.0).build();
        let err = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .transaction_costs(vec![0.01; 3], "expected_return")
            .solve()
            .unwrap_err();
        assert!(matches!(
            err,
            PortfolioError::Compile(CompileError::TransactionCostsWithoutTurnover)
        ));
    }

    #[test]
    fn test_warm_start_raw_accepted() {
        let universe = dense_universe();
        let strategy = Strategy::builder("S").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let sol = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(restrictions.clone())
            .solve()
            .unwrap();
        // Clarabel ignores the hint; the solve must still succeed identically.
        let warmed = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(restrictions)
            .warm_start_raw(sol.raw_x.clone(), sol.raw_z.clone())
            .solve()
            .unwrap();
        assert_eq!(warmed.status, sol.status);
        for (a, b) in warmed.weights_vec.iter().zip(&sol.weights_vec) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_error_source_chains() {
        use std::error::Error;
        let err = PortfolioError::Compile(CompileError::NoDimensions);
        assert!(err.source().is_some_and(|s| s.is::<CompileError>()));
        let inner = CompileError::InvalidUniverse(quartz_core::UniverseError::NoAssets);
        assert!(inner.source().is_some_and(|s| s.is::<quartz_core::UniverseError>()));
    }

    // ---- range normalisation -----------------------------------------------

    /// The README universe; `esg_scale` multiplies `environmental_impact` and
    /// `cov_scale` the covariance (a change of units, nothing else).
    fn esg_universe(esg_scale: f64, cov_scale: f64) -> Universe {
        let c = cov_scale;
        Universe::builder()
            .add_asset(
                Asset::new("AAPL")
                    .tag("currency", "USD")
                    .score("expected_return", 0.10)
                    .score("environmental_impact", 6.5 * esg_scale)
                    .score("transition_risk", 3.0),
            )
            .add_asset(
                Asset::new("NESN")
                    .tag("currency", "CHF")
                    .score("expected_return", 0.04)
                    .score("environmental_impact", 8.0 * esg_scale)
                    .score("transition_risk", 1.5),
            )
            .add_asset(
                Asset::new("GREEN_BOND")
                    .tag("currency", "EUR")
                    .score("expected_return", 0.025)
                    .score("environmental_impact", 9.5 * esg_scale)
                    .score("transition_risk", 0.5),
            )
            .covariance_full(CscMatrix::from(&[
                [0.04 * c, 0.002 * c, 0.001 * c],
                [0.002 * c, 0.01 * c, 0.001 * c],
                [0.001 * c, 0.001 * c, 0.003 * c],
            ]))
            .build()
            .unwrap()
    }

    fn esg_strategy(normalization: Normalization) -> Strategy {
        Strategy::builder("ESG Balanced")
            .minimize_risk(0.40)
            .maximize("expected_return", 0.25)
            .minimize("transition_risk", 0.15)
            .maximize("environmental_impact", 0.20)
            .normalization(normalization)
            .build()
    }

    fn esg_solve(universe: &Universe, normalization: Normalization) -> PortfolioSolution {
        let strategy = esg_strategy(normalization);
        let sol = PortfolioModel::new(universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
        sol
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn test_ranges_are_the_payoff_table() {
        let universe = esg_universe(1.0, 1.0);
        let sol = esg_solve(&universe, Normalization::Range);
        let r = &sol.dimension_ranges;
        assert_eq!(r.len(), 4);
        let names: Vec<&str> = r.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(
            names,
            ["financial_risk", "expected_return", "transition_risk", "environmental_impact"]
        );
        // Long-only, fully invested: each score's best is a single asset,
        // its worst the asset the opposite objective picks.
        assert!((r[1].best - 0.10).abs() < 1e-6 && (r[1].worst - 0.025).abs() < 1e-6, "{r:?}");
        assert!((r[2].best - 0.5).abs() < 1e-6 && (r[2].worst - 3.0).abs() < 1e-6, "{r:?}");
        assert!((r[3].best - 9.5).abs() < 1e-6 && (r[3].worst - 6.5).abs() < 1e-6, "{r:?}");
        // Variance: worst is all-AAPL (0.04), best the minimum-variance mix.
        assert!((r[0].worst - 0.04).abs() < 1e-6, "{r:?}");
        assert!(r[0].best > 0.0 && r[0].best < 0.003, "{r:?}");
    }

    #[test]
    fn test_range_normalisation_gives_small_scale_dimensions_a_say() {
        // 70% return against 30% ESG. Worst → best asset is 0.075 of return
        // and 3.0 of ESG score.
        let universe = esg_universe(1.0, 1.0);
        let solve = |n| {
            let strategy = Strategy::builder("S")
                .maximize("expected_return", 0.7)
                .maximize("environmental_impact", 0.3)
                .normalization(n)
                .build();
            PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(Restrictions::builder().long_only().fully_invested().build())
                .solve()
                .unwrap()
        };
        // Raw: 0.7·0.075 = 0.05 against 0.3·3.0 = 0.9, so ESG decides alone
        // and everything goes to the bond.
        let raw = solve(Normalization::Raw);
        assert!(raw.dimension_ranges.is_empty());
        assert!(raw.weights_vec[2] > 0.999, "{:?}", raw.weights_vec);
        // Normalised: 0.7 against 0.3, so the high-return asset wins.
        let ranged = solve(Normalization::Range);
        assert!(ranged.weights_vec[0] > 0.999, "{:?}", ranged.weights_vec);
    }

    #[test]
    fn test_range_normalisation_is_invariant_to_units() {
        let base = esg_universe(1.0, 1.0);
        let reference = esg_solve(&base, Normalization::Range).weights_vec;
        // ESG on a 0–100 scale, and variance in percent² instead of decimals
        for universe in [esg_universe(10.0, 1.0), esg_universe(1.0, 1e4)] {
            let w = esg_solve(&universe, Normalization::Range).weights_vec;
            assert!(max_diff(&w, &reference) < 1e-5, "{w:?} != {reference:?}");
        }
        // Raw weights do depend on the units of the covariance.
        let raw_a = esg_solve(&base, Normalization::Raw).weights_vec;
        let raw_b = esg_solve(&esg_universe(1.0, 1e4), Normalization::Raw).weights_vec;
        assert!(max_diff(&raw_a, &raw_b) > 1e-2, "{raw_a:?} vs {raw_b:?}");
    }

    #[test]
    fn test_objective_is_the_weighted_sum_of_normalised_scores() {
        let universe = esg_universe(1.0, 1.0);
        let strategy = esg_strategy(Normalization::Range);
        let sol = esg_solve(&universe, Normalization::Range);
        let expected: f64 = strategy
            .dimensions
            .iter()
            .zip(&sol.dimension_ranges)
            .map(|(d, r)| {
                let value = sol.portfolio_scores[normalization::metric_key(d)];
                d.weight * d.sense.sign() * value / r.scale()
            })
            .sum();
        assert!((sol.objective_value - expected).abs() < 1e-6);
    }

    #[test]
    fn test_single_dimension_is_not_normalised() {
        let universe = esg_universe(1.0, 1.0);
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let solve = |n| {
            let strategy = Strategy::builder("MinVar").minimize_risk(1.0).normalization(n).build();
            PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(restrictions.clone())
                .solve()
                .unwrap()
        };
        let ranged = solve(Normalization::Range);
        assert!(ranged.dimension_ranges.is_empty());
        assert_eq!(ranged.weights_vec, solve(Normalization::Raw).weights_vec);
    }

    #[test]
    fn test_zero_weight_dimension_is_not_anchored() {
        // An unbounded dimension at weight 0 must not break the solve.
        let universe = esg_universe(1.0, 1.0);
        let strategy = Strategy::builder("S")
            .minimize_risk(0.5)
            .minimize("transition_risk", 0.5)
            .maximize("expected_return", 0.0)
            .build();
        let sol = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
        assert_eq!(sol.dimension_ranges.len(), 3);
        assert_eq!(sol.dimension_ranges[2].scale(), 1.0);
    }

    #[test]
    fn test_infeasible_problem_is_reported_not_an_error() {
        let universe = esg_universe(1.0, 1.0);
        let strategy = Strategy::builder("S")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .score_min("environmental_impact", 20.0)
            .build();
        let sol = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();
        assert_eq!(sol.status, SolveStatus::Infeasible);
        assert!(sol.dimension_ranges.is_empty());
    }

    #[test]
    fn test_unbounded_dimension_is_a_normalisation_error() {
        // Fully invested but shorting allowed: return alone has no optimum.
        let universe = esg_universe(1.0, 1.0);
        let restrictions = Restrictions::builder().fully_invested().build();
        let solve = |n| {
            let strategy = Strategy::builder("S")
                .minimize_risk(0.5)
                .maximize("expected_return", 0.5)
                .normalization(n)
                .build();
            PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(restrictions.clone())
                .solve()
        };
        match solve(Normalization::Range) {
            Err(PortfolioError::Normalization { dimension, .. }) => {
                assert_eq!(dimension, "expected_return")
            }
            other => panic!("expected a normalisation error, got {other:?}"),
        }
        // The combined problem is bounded by the risk term: raw still solves.
        assert_eq!(solve(Normalization::Raw).unwrap().status, SolveStatus::Optimal);
    }

    #[test]
    fn test_constant_dimension_solves() {
        // Identical scores: zero range, divisor 1, no NaN.
        let universe = Universe::builder()
            .add_asset(Asset::new("A").score("expected_return", 0.10).score("esg", 5.0))
            .add_asset(Asset::new("B").score("expected_return", 0.02).score("esg", 5.0))
            .covariance_full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]]))
            .build()
            .unwrap();
        let strategy = Strategy::builder("S")
            .minimize_risk(0.4)
            .maximize("expected_return", 0.3)
            .maximize("esg", 0.3)
            .build();
        let sol = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve()
            .unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
        assert_eq!(sol.dimension_ranges[2].scale(), 1.0);
        assert!(sol.weights_vec.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_supplied_ranges_replace_the_anchor_solves() {
        let universe = esg_universe(1.0, 1.0);
        let strategy = esg_strategy(Normalization::Range);
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let model = || {
            PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(restrictions.clone())
        };
        let ranges = model().compute_dimension_ranges().unwrap().unwrap();
        let direct = model().solve().unwrap();
        assert_eq!(direct.dimension_ranges, ranges);
        let supplied = model().dimension_ranges(ranges.clone()).solve().unwrap();
        assert_eq!(supplied.weights_vec, direct.weights_vec);

        let short = model().dimension_ranges(ranges[..2].to_vec()).solve();
        assert!(matches!(
            short,
            Err(PortfolioError::DimensionRangeMismatch { expected: 4, got: 2 })
        ));
    }

    #[test]
    fn test_range_normalisation_with_tactic_factor_turnover_and_costs() {
        let universe = factor_universe();
        let strategy = Strategy::builder("S")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .build();
        let tactic = Tactic::builder("T").override_weight("financial_risk", 0.8).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let solve = |tactic: Option<&Tactic>| {
            let mut model = PortfolioModel::new(&universe)
                .strategy(&strategy)
                .restrictions(restrictions.clone())
                .turnover(PREV.to_vec(), 0.5)
                .transaction_costs(vec![0.001; 3], "expected_return");
            if let Some(t) = tactic {
                model = model.tactic(t);
            }
            let sol = model.solve().unwrap();
            assert_eq!(sol.status, SolveStatus::Optimal);
            assert_eq!(sol.dimension_ranges.len(), 2);
            sol
        };
        let base = solve(None);
        let defensive = solve(Some(&tactic));
        // More weight on risk: no more variance than before.
        assert!(defensive.variance().unwrap() <= base.variance().unwrap() + 1e-9);
        let turnover: f64 = base.weights_vec.iter().zip(&PREV).map(|(w, p)| (w - p).abs()).sum();
        assert!(turnover <= 0.5 + 1e-6);
    }

    // ---- names that match nothing ------------------------------------------

    #[test]
    fn test_typo_in_score_key_is_a_compile_error_under_range_normalisation() {
        // The error is the compile error, not a Normalization error from an
        // anchor solve of the phantom dimension.
        let universe = esg_universe(1.0, 1.0);
        let strategy = Strategy::builder("S")
            .minimize_risk(0.4)
            .maximize("expected_return", 0.3)
            .maximize("enviromental_impact", 0.3)
            .build();
        let result = PortfolioModel::new(&universe)
            .strategy(&strategy)
            .restrictions(Restrictions::builder().long_only().fully_invested().build())
            .solve();
        assert!(matches!(
            result,
            Err(PortfolioError::Compile(CompileError::UnknownScoreKey { ref score_key, .. }))
                if score_key == "enviromental_impact"
        ));
    }

    #[test]
    fn test_unknown_exclusion_is_an_error_unless_opted_out() {
        let universe = esg_universe(1.0, 1.0);
        let strategy = esg_strategy(Normalization::Range);
        let solve = |r: Restrictions| {
            PortfolioModel::new(&universe).strategy(&strategy).restrictions(r).solve()
        };
        let base = || Restrictions::builder().long_only().fully_invested().exclude_asset("TOBACCO_CO");
        assert!(matches!(
            solve(base().build()),
            Err(PortfolioError::Compile(CompileError::UnknownExcludedAssets { ref ids }))
                if ids == &["TOBACCO_CO"]
        ));
        let sol = solve(base().ignore_unknown_assets().build()).unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
    }
}
