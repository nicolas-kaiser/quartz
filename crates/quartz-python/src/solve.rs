//! Solve entry points and result types.
//!
//! All solving happens with the GIL released (`Python::detach`), so
//! `solve_batch` runs truly parallel on the rayon pool while Python threads
//! keep working. PyRef guards are not Send, so every Rust value is cloned out
//! of its pyclass *before* detaching.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use quartz_portfolio::batch::BatchProblem;
use quartz_portfolio::frontier::FrontierExplorer;
use quartz_portfolio::{PortfolioModel, Restrictions as RRestrictions};
use quartz_solver::SolverSettings;

use crate::convert::SolveStatus;
use crate::qerr;
use crate::types::{Problem, Restrictions, Strategy, Tactic, Universe};

/// The result of one portfolio optimization.
#[pyclass(module = "quartz")]
pub struct Solution {
    #[pyo3(get)]
    pub status: SolveStatus,
    weights: Vec<(String, f64)>,
    #[pyo3(get)]
    pub weights_vec: Vec<f64>,
    #[pyo3(get)]
    pub portfolio_scores: HashMap<String, f64>,
    #[pyo3(get)]
    pub objective_value: f64,
    #[pyo3(get)]
    pub solve_time_s: f64,
    #[pyo3(get)]
    pub iterations: u32,
    /// Full primal vector `[w, t?, y_factor?, cvar_aux?]` including auxiliary
    /// variables. Pass a previous `Solution` as `warm_start` to `solve` to seed
    /// the next solve with it (only the OSQP backend exploits the hint).
    #[pyo3(get)]
    pub raw_x: Vec<f64>,
    /// Full dual vector (same sign convention on all backends).
    #[pyo3(get)]
    pub raw_z: Vec<f64>,
}

impl Solution {
    fn from_rust(sol: quartz_portfolio::PortfolioSolution) -> Self {
        Self {
            status: sol.status.into(),
            weights: sol
                .weights
                .iter()
                .map(|(id, w)| (id.to_string(), *w))
                .collect(),
            weights_vec: sol.weights_vec,
            portfolio_scores: sol.portfolio_scores,
            objective_value: sol.objective_value,
            solve_time_s: sol.solve_time_s,
            iterations: sol.iterations,
            raw_x: sol.raw_x,
            raw_z: sol.raw_z,
        }
    }
}

#[pymethods]
impl Solution {
    /// Weights as an ordered dict {asset_id: weight}.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (id, w) in &self.weights {
            d.set_item(id, w)?;
        }
        Ok(d)
    }

    #[getter]
    fn is_optimal(&self) -> bool {
        self.status == SolveStatus::Optimal
    }

    fn __repr__(&self) -> String {
        format!(
            "Solution(status={:?}, objective={:.6}, {} assets)",
            self.status,
            self.objective_value,
            self.weights.len()
        )
    }
}

fn settings(
    max_iter: u32,
    verbose: bool,
    tol_gap_abs: f64,
    tol_gap_rel: f64,
    tol_feas: f64,
) -> SolverSettings {
    SolverSettings {
        verbose,
        max_iter,
        tol_gap_abs,
        tol_gap_rel,
        tol_feas,
    }
}

/// Solve a single portfolio optimization problem.
///
/// `turnover=(previous_weights, max_turnover)`; `transaction_costs=(rates, score_key)`
/// makes `score_key` net of `Σ rates[i]·|w_i − previous_i|` wherever the strategy
/// wants more of it (needs `turnover`; see `docs/transaction-costs.md`).
/// `warm_start=<previous Solution>` seeds the solver with that solution's
/// `raw_x`/`raw_z` (only the OSQP backend exploits the hint; Clarabel ignores it).
#[pyfunction]
#[pyo3(signature = (universe, strategy, *, tactic=None, restrictions=None, turnover=None,
                    transaction_costs=None, max_iter=200, verbose=false,
                    tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, warm_start=None))]
#[allow(clippy::too_many_arguments)]
pub fn solve(
    py: Python<'_>,
    universe: &Universe,
    strategy: &Strategy,
    tactic: Option<&Tactic>,
    restrictions: Option<&Restrictions>,
    turnover: Option<(Vec<f64>, f64)>,
    transaction_costs: Option<(Vec<f64>, String)>,
    max_iter: u32,
    verbose: bool,
    tol_gap_abs: f64,
    tol_gap_rel: f64,
    tol_feas: f64,
    warm_start: Option<&Solution>,
) -> PyResult<Solution> {
    // Extract owned Rust values before detaching (PyRef is not Send).
    let u = universe.inner.clone();
    let s = strategy.to_strategy();
    let t = tactic.map(|t| t.to_tactic());
    let r = restrictions.map(|r| r.inner.clone()).unwrap_or_default();
    let solver_settings = settings(max_iter, verbose, tol_gap_abs, tol_gap_rel, tol_feas);
    let ws = warm_start.map(|sol| (sol.raw_x.clone(), sol.raw_z.clone()));

    let result = py.detach(move || {
        let mut model = PortfolioModel::new(&u)
            .strategy(&s)
            .restrictions(r)
            .solver_settings(solver_settings);
        if let Some(t) = &t {
            model = model.tactic(t);
        }
        if let Some((prev, max)) = turnover {
            model = model.turnover(prev, max);
        }
        if let Some((rates, key)) = transaction_costs {
            model = model.transaction_costs(rates, key);
        }
        if let Some((x, z)) = ws {
            model = model.warm_start_raw(x, z);
        }
        model.solve()
    });
    result.map(Solution::from_rust).map_err(qerr)
}

/// A batch item: a `Problem` or a bare `(universe, strategy)` tuple.
#[derive(FromPyObject)]
pub enum BatchInput<'py> {
    Problem(PyRef<'py, Problem>),
    Pair(PyRef<'py, Universe>, PyRef<'py, Strategy>),
}

struct OwnedItem {
    universe: quartz_core::Universe,
    strategy: quartz_portfolio::Strategy,
    tactic: Option<quartz_portfolio::Tactic>,
    restrictions: RRestrictions,
    turnover: Option<(Vec<f64>, f64)>,
    transaction_costs: Option<(Vec<f64>, String)>,
}

/// Solve many independent problems in parallel (rayon; the GIL is released).
///
/// Returns a list in input order, each element either a `Solution` or a
/// `QuartzError` *instance* (never raised) — per-item error isolation.
/// Infeasible problems return a `Solution` with `SolveStatus.Infeasible`.
/// `restrictions`/`tactic` kwargs are defaults for items that don't carry
/// their own.
#[pyfunction]
#[pyo3(signature = (problems, *, restrictions=None, tactic=None, max_iter=200,
                    tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8))]
pub fn solve_batch(
    py: Python<'_>,
    problems: Vec<BatchInput<'_>>,
    restrictions: Option<&Restrictions>,
    tactic: Option<&Tactic>,
    max_iter: u32,
    tol_gap_abs: f64,
    tol_gap_rel: f64,
    tol_feas: f64,
) -> PyResult<Vec<Py<pyo3::types::PyAny>>> {
    let default_r = restrictions.map(|r| r.inner.clone()).unwrap_or_default();
    let default_t = tactic.map(|t| t.to_tactic());

    // Clone everything into owned items before detaching.
    let items: Vec<OwnedItem> = problems
        .iter()
        .map(|input| match input {
            BatchInput::Problem(p) => OwnedItem {
                universe: p.universe.clone(),
                strategy: p.strategy.clone(),
                tactic: p.tactic.clone().or_else(|| default_t.clone()),
                restrictions: p
                    .restrictions
                    .clone()
                    .unwrap_or_else(|| default_r.clone()),
                turnover: p.turnover.clone(),
                transaction_costs: p.transaction_costs.clone(),
            },
            BatchInput::Pair(u, s) => OwnedItem {
                universe: u.inner.clone(),
                strategy: s.to_strategy(),
                tactic: default_t.clone(),
                restrictions: default_r.clone(),
                turnover: None,
                transaction_costs: None,
            },
        })
        .collect();
    let solver_settings = settings(max_iter, false, tol_gap_abs, tol_gap_rel, tol_feas);

    let results = py.detach(move || {
        let batch: Vec<BatchProblem> = items
            .iter()
            .map(|item| {
                let mut p = BatchProblem::new(&item.universe, &item.strategy)
                    .restrictions(item.restrictions.clone());
                if let Some(t) = &item.tactic {
                    p = p.tactic(t);
                }
                if let Some((prev, max)) = &item.turnover {
                    p = p.turnover(prev.clone(), *max);
                }
                if let Some((rates, key)) = &item.transaction_costs {
                    p = p.transaction_costs(rates.clone(), key.clone());
                }
                p
            })
            .collect();
        quartz_portfolio::solve_batch(&batch, &solver_settings)
    });

    results
        .into_iter()
        .map(|r| match r {
            Ok(sol) => Ok(Solution::from_rust(sol)
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            Err(e) => Ok(qerr(e).into_value(py).into_any()),
        })
        .collect()
}

/// One explored point on a Pareto frontier.
#[pyclass(module = "quartz")]
pub struct FrontierPoint {
    dimension_weights: Vec<(String, f64)>,
    weights: Vec<(String, f64)>,
    #[pyo3(get)]
    pub weights_vec: Vec<f64>,
    #[pyo3(get)]
    pub portfolio_scores: HashMap<String, f64>,
    #[pyo3(get)]
    pub objective_value: f64,
    #[pyo3(get)]
    pub is_efficient: bool,
}

#[pymethods]
impl FrontierPoint {
    #[getter]
    fn dimension_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, w) in &self.dimension_weights {
            d.set_item(name, w)?;
        }
        Ok(d)
    }

    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (id, w) in &self.weights {
            d.set_item(id, w)?;
        }
        Ok(d)
    }
}

/// The result of a frontier exploration.
#[pyclass(module = "quartz")]
pub struct Frontier {
    #[pyo3(get)]
    pub objective_dims: Vec<String>,
    #[pyo3(get)]
    pub n_skipped: usize,
    points: Vec<Py<FrontierPoint>>,
}

#[pymethods]
impl Frontier {
    #[getter]
    fn points(&self, py: Python<'_>) -> Vec<Py<FrontierPoint>> {
        self.points.iter().map(|p| p.clone_ref(py)).collect()
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }
}

fn frontier_from_rust(
    py: Python<'_>,
    result: quartz_portfolio::FrontierResult,
) -> PyResult<Frontier> {
    let points = result
        .points
        .into_iter()
        .map(|p| {
            Py::new(
                py,
                FrontierPoint {
                    dimension_weights: p.dimension_weights,
                    weights: p
                        .weights
                        .iter()
                        .map(|(id, w)| (id.to_string(), *w))
                        .collect(),
                    weights_vec: p.weights_vec,
                    portfolio_scores: p.portfolio_scores,
                    objective_value: p.objective_value,
                    is_efficient: p.is_efficient,
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Frontier {
        objective_dims: result.objective_dims,
        n_skipped: result.n_skipped,
        points,
    })
}

/// Trade two dimensions against each other over `n_points` steps.
/// Dimension names are the score key, or "financial_risk" for the risk dim.
/// At most 10,000 points are explored; `n_points` above that raises `QuartzError`.
#[pyfunction]
#[pyo3(signature = (universe, strategy, dim_a, dim_b, n_points=25, *,
                    tactic=None, restrictions=None, turnover=None,
                    transaction_costs=None))]
#[allow(clippy::too_many_arguments)]
pub fn sweep(
    py: Python<'_>,
    universe: &Universe,
    strategy: &Strategy,
    dim_a: &str,
    dim_b: &str,
    n_points: usize,
    tactic: Option<&Tactic>,
    restrictions: Option<&Restrictions>,
    turnover: Option<(Vec<f64>, f64)>,
    transaction_costs: Option<(Vec<f64>, String)>,
) -> PyResult<Frontier> {
    let u = universe.inner.clone();
    let s = strategy.to_strategy();
    let t = tactic.map(|t| t.to_tactic());
    let r = restrictions.map(|r| r.inner.clone()).unwrap_or_default();
    let (dim_a, dim_b) = (dim_a.to_string(), dim_b.to_string());

    let result = py.detach(move || {
        let mut explorer = FrontierExplorer::new(&u, &s).restrictions(r);
        if let Some(t) = &t {
            explorer = explorer.tactic(t);
        }
        if let Some((prev, max)) = turnover {
            explorer = explorer.turnover(prev, max);
        }
        if let Some((rates, key)) = transaction_costs {
            explorer = explorer.transaction_costs(rates, key);
        }
        explorer.sweep(&dim_a, &dim_b, n_points)
    });
    frontier_from_rust(py, result.map_err(qerr)?)
}

/// Enumerate a full simplex lattice over every strategy dimension.
/// At most 10,000 points are explored; a grid larger than that raises `QuartzError`.
#[pyfunction]
#[pyo3(signature = (universe, strategy, resolution=5, *,
                    tactic=None, restrictions=None, turnover=None,
                    transaction_costs=None))]
#[allow(clippy::too_many_arguments)]
pub fn simplex_grid(
    py: Python<'_>,
    universe: &Universe,
    strategy: &Strategy,
    resolution: usize,
    tactic: Option<&Tactic>,
    restrictions: Option<&Restrictions>,
    turnover: Option<(Vec<f64>, f64)>,
    transaction_costs: Option<(Vec<f64>, String)>,
) -> PyResult<Frontier> {
    let u = universe.inner.clone();
    let s = strategy.to_strategy();
    let t = tactic.map(|t| t.to_tactic());
    let r = restrictions.map(|r| r.inner.clone()).unwrap_or_default();

    let result = py.detach(move || {
        let mut explorer = FrontierExplorer::new(&u, &s).restrictions(r);
        if let Some(t) = &t {
            explorer = explorer.tactic(t);
        }
        if let Some((prev, max)) = turnover {
            explorer = explorer.turnover(prev, max);
        }
        if let Some((rates, key)) = transaction_costs {
            explorer = explorer.transaction_costs(rates, key);
        }
        explorer.simplex_grid(resolution)
    });
    frontier_from_rust(py, result.map_err(qerr)?)
}
