use clarabel::algebra::CscMatrix;
use clarabel::solver::{self as cl, DefaultSettings, DefaultSettingsBuilder, IPSolver};

#[cfg(feature = "osqp")]
mod osqp_backend;

/// Which QP solver to use.
///
/// - `Clarabel` (default): interior-point, high accuracy, pure Rust. Cannot
///   exploit warm starts — every solve starts from scratch.
/// - `Osqp` (cargo feature `osqp`): ADMM first-order method wrapping the OSQP
///   C library. Supports warm starting: seeding a solve with the previous
///   primal/dual pair makes near-identical consecutive problems (sequential
///   backtest dates, adjacent frontier points) converge in far fewer
///   iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Backend {
    #[default]
    Clarabel,
    #[cfg(feature = "osqp")]
    Osqp,
}

/// A warm-start hint: the primal/dual solution of a previous, similar problem.
///
/// `x` must have length n_vars of the new problem, `y` length m (or empty to
/// seed only the primal). Both backends accept the hint; Clarabel ignores it
/// (interior-point methods cannot use one), OSQP errors on a length mismatch —
/// that means the caller chained solutions across structurally different
/// problems (e.g. turnover or factor aux variables added/removed).
#[derive(Debug, Clone)]
pub struct WarmStart {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

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

impl SolverSettings {
    /// Backend-appropriate defaults.
    ///
    /// Clarabel (interior-point) converges to 1e-8 in ~10 iterations; OSQP
    /// (ADMM) needs thousands of cheap iterations for tight tolerances, so it
    /// gets 1e-5 tolerances with generous iteration headroom — solution
    /// polishing then recovers high accuracy. Passing Clarabel-tuned settings
    /// (max_iter 200, tol 1e-8) to OSQP is legal but usually ends in
    /// `MaxIterations`.
    pub fn default_for(backend: Backend) -> Self {
        match backend {
            Backend::Clarabel => Self::default(),
            #[cfg(feature = "osqp")]
            Backend::Osqp => Self {
                verbose: false,
                max_iter: 20_000,
                tol_gap_abs: 1e-6,
                tol_gap_rel: 1e-6,
                tol_feas: 1e-7,
            },
        }
    }
}

/// Solve a compiled QP with the default backend (Clarabel), no warm start.
pub fn solve_qp(
    problem: &CompiledProblem,
    settings: &SolverSettings,
) -> Result<RawSolution, SolverError> {
    solve_qp_with(problem, settings, Backend::Clarabel, None)
}

/// Solve a compiled QP with an explicit backend and optional warm start.
///
/// The warm-start hint is ignored by Clarabel (with no error) so callers can
/// stay backend-generic; under OSQP a dimension mismatch is a hard error.
pub fn solve_qp_with(
    problem: &CompiledProblem,
    settings: &SolverSettings,
    backend: Backend,
    warm_start: Option<&WarmStart>,
) -> Result<RawSolution, SolverError> {
    match backend {
        Backend::Clarabel => {
            let _ = warm_start; // interior-point: cannot use a warm start
            solve_clarabel(problem, settings)
        }
        #[cfg(feature = "osqp")]
        Backend::Osqp => osqp_backend::solve(problem, settings, warm_start),
    }
}

fn solve_clarabel(
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
    /// Warm-start vectors don't match the problem dimensions.
    WarmStart(String),
    /// The problem uses cones the selected backend cannot express.
    Unsupported(String),
    /// Backend problem setup failed.
    Setup(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Settings(msg) => write!(f, "Solver settings error: {msg}"),
            SolverError::WarmStart(msg) => write!(f, "Warm start error: {msg}"),
            SolverError::Unsupported(msg) => write!(f, "Unsupported problem: {msg}"),
            SolverError::Setup(msg) => write!(f, "Solver setup error: {msg}"),
        }
    }
}

impl std::error::Error for SolverError {}
