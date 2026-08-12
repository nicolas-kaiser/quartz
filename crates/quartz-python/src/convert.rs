//! Input/output conversions: dense matrices from Python, solver status mirror.

use clarabel::algebra::CscMatrix;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use crate::qerr;

/// Solver status, mirroring `quartz_solver::SolveStatus`.
#[pyclass(eq, eq_int, frozen, skip_from_py_object, module = "quartz")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolveStatus {
    Optimal,
    Infeasible,
    DualInfeasible,
    AlmostOptimal,
    MaxIterations,
    NumericalError,
}

impl From<quartz_solver::SolveStatus> for SolveStatus {
    fn from(s: quartz_solver::SolveStatus) -> Self {
        match s {
            quartz_solver::SolveStatus::Optimal => SolveStatus::Optimal,
            quartz_solver::SolveStatus::Infeasible => SolveStatus::Infeasible,
            quartz_solver::SolveStatus::DualInfeasible => SolveStatus::DualInfeasible,
            quartz_solver::SolveStatus::AlmostOptimal => SolveStatus::AlmostOptimal,
            quartz_solver::SolveStatus::MaxIterations => SolveStatus::MaxIterations,
            quartz_solver::SolveStatus::NumericalError => SolveStatus::NumericalError,
        }
    }
}

/// A dense matrix from Python: numpy float64 array (fast path) or any nested
/// sequence of floats. numpy is an optional runtime dependency — the Rows
/// variant covers plain lists and non-f64 numpy arrays via element extraction.
#[derive(FromPyObject)]
pub enum Matrix<'py> {
    Array(PyReadonlyArray2<'py, f64>),
    Rows(Vec<Vec<f64>>),
}

impl Matrix<'_> {
    pub fn to_csc(&self) -> PyResult<CscMatrix<f64>> {
        match self {
            Matrix::Array(a) => {
                let view = a.as_array();
                let (m, n) = (view.nrows(), view.ncols());
                if m == 0 || n == 0 {
                    return Err(qerr("matrix is empty"));
                }
                let mut colptr = Vec::with_capacity(n + 1);
                let mut rowval = Vec::new();
                let mut nzval = Vec::new();
                colptr.push(0);
                for j in 0..n {
                    for i in 0..m {
                        let v = view[(i, j)];
                        if v != 0.0 {
                            rowval.push(i);
                            nzval.push(v);
                        }
                    }
                    colptr.push(rowval.len());
                }
                Ok(CscMatrix::new(m, n, colptr, rowval, nzval))
            }
            Matrix::Rows(rows) => dense_to_csc(rows).map_err(qerr),
        }
    }
}

/// Dense row-major matrix to CSC (same helper as the quartz-demo bridge).
fn dense_to_csc(rows: &[Vec<f64>]) -> Result<CscMatrix<f64>, String> {
    let m = rows.len();
    if m == 0 {
        return Err("matrix is empty".into());
    }
    let n = rows[0].len();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(format!("matrix row {i} has {} entries, expected {n}", row.len()));
            }
            let v = row[j];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    Ok(CscMatrix::new(m, n, colptr, rowval, nzval))
}
