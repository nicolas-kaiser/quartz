//! Parallel batch solving: solve many independent portfolio problems at once
//! (e.g. backtesting one strategy over hundreds of rebalance dates, each with
//! its own covariance and expected returns).
//!
//! With the default `parallel` feature, solves run on the rayon thread pool;
//! without it, they run serially. Either way:
//!
//! - Results come back in **input order**.
//! - Errors are **isolated per item**: one bad problem doesn't affect the
//!   others. Infeasibility is a `SolveStatus` inside `Ok`, not an `Err`.
//! - Turnover *chaining* (date t constrained by date t−1's solution) is
//!   inherently sequential and out of scope; per-item turnover against known
//!   previous weights is supported.
//! - Avoid `SolverSettings { verbose: true, .. }` in parallel runs: Clarabel's
//!   progress output would interleave across threads.

use quartz_core::Universe;
use quartz_solver::SolverSettings;

use crate::model::{PortfolioError, PortfolioModel};
use crate::restriction::Restrictions;
use crate::solution::PortfolioSolution;
use crate::strategy::Strategy;
use crate::tactic::Tactic;

/// One independent problem in a batch.
///
/// Borrows the large inputs (universe, strategy) so a shared strategy across
/// 1000 dates costs nothing; restrictions and turnover are small owned values.
pub struct BatchProblem<'a> {
    pub universe: &'a Universe,
    pub strategy: &'a Strategy,
    pub tactic: Option<&'a Tactic>,
    pub restrictions: Restrictions,
    /// Per-item turnover: (previous weights, max turnover).
    pub turnover: Option<(Vec<f64>, f64)>,
}

impl<'a> BatchProblem<'a> {
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
}

/// Solve every problem in the batch, in parallel when the `parallel` feature
/// is enabled. Results are in the same order as `problems`.
pub fn solve_batch(
    problems: &[BatchProblem<'_>],
    settings: &SolverSettings,
) -> Vec<Result<PortfolioSolution, PortfolioError>> {
    crate::par::par_map(problems, |p| {
        let mut model = PortfolioModel::new(p.universe)
            .strategy(p.strategy)
            .restrictions(p.restrictions.clone())
            .solver_settings(settings.clone());
        if let Some(t) = p.tactic {
            model = model.tactic(t);
        }
        if let Some((prev, max)) = &p.turnover {
            model = model.turnover(prev.clone(), *max);
        }
        model.solve()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution::SolveStatus;
    use clarabel::algebra::CscMatrix;
    use quartz_core::Asset;

    /// 3 assets; `hot` gets a dominant expected return.
    fn universe_favoring(hot: usize) -> Universe {
        let mut builder = Universe::builder();
        for i in 0..3 {
            let ret = if i == hot { 0.50 } else { 0.01 };
            let id = format!("A{i}");
            builder = builder.add_asset(Asset::new(id.as_str()).score("expected_return", ret));
        }
        builder
            .covariance_full(CscMatrix::from(&[
                [0.04, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.04],
            ]))
            .build()
            .unwrap()
    }

    fn strategy() -> Strategy {
        Strategy::builder("Batch")
            .minimize_risk(0.3)
            .maximize("expected_return", 0.7)
            .build()
    }

    fn restrictions() -> Restrictions {
        Restrictions::builder().long_only().fully_invested().build()
    }

    #[test]
    fn test_batch_matches_serial_and_order_preserved() {
        let universes: Vec<Universe> = (0..8).map(|i| universe_favoring(i % 3)).collect();
        let strategy = strategy();
        let problems: Vec<BatchProblem> = universes
            .iter()
            .map(|u| BatchProblem::new(u, &strategy).restrictions(restrictions()))
            .collect();

        let results = solve_batch(&problems, &SolverSettings::default());
        assert_eq!(results.len(), 8);

        for (i, (result, universe)) in results.iter().zip(&universes).enumerate() {
            let batch_sol = result.as_ref().unwrap();
            assert_eq!(batch_sol.status, SolveStatus::Optimal);

            // Order: item i's optimum concentrates in the favored asset i % 3
            let hot = i % 3;
            let max_idx = batch_sol
                .weights_vec
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(max_idx, hot, "item {i} should favor asset {hot}");

            // Serial equivalence (deterministic solver → identical results)
            let serial = PortfolioModel::new(universe)
                .strategy(&strategy)
                .restrictions(restrictions())
                .solve()
                .unwrap();
            assert_eq!(serial.iterations, batch_sol.iterations);
            for (wb, ws) in batch_sol.weights_vec.iter().zip(&serial.weights_vec) {
                assert!((wb - ws).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_batch_error_isolation() {
        let universes: Vec<Universe> = (0..4).map(|_| universe_favoring(0)).collect();
        let strategy = strategy();
        let problems: Vec<BatchProblem> = universes
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let mut p = BatchProblem::new(u, &strategy).restrictions(restrictions());
                if i == 2 {
                    // Wrong-length previous weights → deterministic CompileError
                    p = p.turnover(vec![0.5, 0.5], 0.10);
                }
                p
            })
            .collect();

        let results = solve_batch(&problems, &SolverSettings::default());
        for (i, r) in results.iter().enumerate() {
            if i == 2 {
                assert!(r.is_err(), "item 2 must fail");
            } else {
                assert_eq!(r.as_ref().unwrap().status, SolveStatus::Optimal);
            }
        }
    }

    #[test]
    fn test_batch_infeasible_is_ok_not_err() {
        let universe = universe_favoring(0);
        let impossible = Strategy::builder("Impossible")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .score_min("expected_return", 99.0)
            .build();
        let problems = vec![BatchProblem::new(&universe, &impossible).restrictions(restrictions())];

        let results = solve_batch(&problems, &SolverSettings::default());
        let sol = results[0].as_ref().unwrap();
        assert_eq!(sol.status, SolveStatus::Infeasible);
    }

    #[test]
    fn test_batch_empty() {
        let results = solve_batch(&[], &SolverSettings::default());
        assert!(results.is_empty());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_thread_count_independence() {
        let universes: Vec<Universe> = (0..6).map(|i| universe_favoring(i % 3)).collect();
        let strategy = strategy();
        let problems: Vec<BatchProblem> = universes
            .iter()
            .map(|u| BatchProblem::new(u, &strategy).restrictions(restrictions()))
            .collect();

        let default_pool = solve_batch(&problems, &SolverSettings::default());
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| solve_batch(&problems, &SolverSettings::default()));

        for (a, b) in default_pool.iter().zip(&single_thread) {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            assert_eq!(a.iterations, b.iterations);
            assert_eq!(a.weights_vec, b.weights_vec); // bitwise: deterministic solver
        }
    }
}
