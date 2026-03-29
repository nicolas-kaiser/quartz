use clarabel::algebra::CscMatrix;
use clarabel::solver::{self as cl, DefaultSettings, DefaultSettingsBuilder, IPSolver};

/// A compiled QP/conic problem ready to be solved.
///
/// Represents: min (1/2)xᵀPx + qᵀx  s.t. Ax + s = b, s ∈ K
#[derive(Debug, Clone)]
pub struct CompiledProblem {
    /// Quadratic objective matrix (n × n, upper triangle, symmetric positive semidefinite).
    pub p: CscMatrix<f64>,
    /// Linear objective vector (length n).
    pub q: Vec<f64>,
    /// Constraint matrix (m × n).
    pub a: CscMatrix<f64>,
    /// Constraint right-hand side (length m).
    pub b: Vec<f64>,
    /// Cone specification for slack variables.
    pub cones: Vec<cl::SupportedConeT<f64>>,
    /// Number of asset weight variables (first n_assets entries of x).
    pub n_assets: usize,
    /// Number of auxiliary variables (e.g. turnover vars).
    pub n_aux: usize,
}

impl CompiledProblem {
    /// Total number of decision variables.
    pub fn n_vars(&self) -> usize {
        self.n_assets + self.n_aux
    }

    /// Total number of constraints.
    pub fn n_constraints(&self) -> usize {
        self.b.len()
    }
}

/// Status of the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    Infeasible,
    DualInfeasible,
    AlmostOptimal,
    MaxIterations,
    NumericalError,
}

/// Raw solution from the solver.
#[derive(Debug, Clone)]
pub struct RawSolution {
    pub status: SolveStatus,
    /// Primal solution vector (length n = n_assets + n_aux).
    pub x: Vec<f64>,
    /// Dual variables (length m).
    pub z: Vec<f64>,
    /// Objective value.
    pub obj_val: f64,
    /// Solve time in seconds.
    pub solve_time_s: f64,
    /// Number of iterations.
    pub iterations: u32,
}

/// Solver settings.
#[derive(Debug, Clone)]
pub struct SolverSettings {
    pub verbose: bool,
    pub max_iter: u32,
    pub tol_gap_abs: f64,
    pub tol_gap_rel: f64,
    pub tol_feas: f64,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            verbose: false,
            max_iter: 200,
            tol_gap_abs: 1e-8,
            tol_gap_rel: 1e-8,
            tol_feas: 1e-8,
        }
    }
}

/// Solve a compiled QP/conic problem using Clarabel.
pub fn solve_qp(
    problem: &CompiledProblem,
    settings: &SolverSettings,
) -> Result<RawSolution, SolverError> {
    let cl_settings: DefaultSettings<f64> = DefaultSettingsBuilder::default()
        .verbose(settings.verbose)
        .max_iter(settings.max_iter)
        .tol_gap_abs(settings.tol_gap_abs)
        .tol_gap_rel(settings.tol_gap_rel)
        .tol_feas(settings.tol_feas)
        .build()
        .map_err(|e| SolverError::Settings(e.to_string()))?;

    let mut solver =
        cl::DefaultSolver::new(&problem.p, &problem.q, &problem.a, &problem.b, &problem.cones, cl_settings)
            .map_err(|e| SolverError::Settings(format!("{:?}", e)))?;

    solver.solve();

    let sol = &solver.solution;
    let status = match sol.status {
        cl::SolverStatus::Solved => SolveStatus::Optimal,
        cl::SolverStatus::PrimalInfeasible => SolveStatus::Infeasible,
        cl::SolverStatus::DualInfeasible => SolveStatus::DualInfeasible,
        cl::SolverStatus::AlmostSolved => SolveStatus::AlmostOptimal,
        cl::SolverStatus::MaxIterations => SolveStatus::MaxIterations,
        _ => SolveStatus::NumericalError,
    };

    Ok(RawSolution {
        status,
        x: sol.x.clone(),
        z: sol.z.clone(),
        obj_val: sol.obj_val,
        solve_time_s: sol.solve_time,
        iterations: sol.iterations,
    })
}

/// Errors from the solver layer.
#[derive(Debug, Clone)]
pub enum SolverError {
    Settings(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Settings(msg) => write!(f, "Solver settings error: {msg}"),
        }
    }
}

impl std::error::Error for SolverError {}
