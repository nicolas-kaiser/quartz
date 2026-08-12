//! Python bindings for Quartz.
//!
//! Built with maturin (`maturin develop --release -m crates/quartz-python/Cargo.toml`).
//! The `pyo3/extension-module` feature is enabled only by maturin via
//! pyproject.toml so that plain `cargo build`/`cargo test` keep working.

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod convert;
mod solve;
mod types;

create_exception!(
    quartz,
    QuartzError,
    PyValueError,
    "Quartz modeling or solver error."
);

/// Map any displayable Rust error to the single Python exception type.
pub(crate) fn qerr(e: impl std::fmt::Display) -> PyErr {
    QuartzError::new_err(e.to_string())
}

#[pymodule(gil_used = true)]
fn quartz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("QuartzError", m.py().get_type::<QuartzError>())?;
    m.add_class::<convert::SolveStatus>()?;
    m.add_class::<types::Asset>()?;
    m.add_class::<types::Universe>()?;
    m.add_class::<types::Strategy>()?;
    m.add_class::<types::Tactic>()?;
    m.add_class::<types::Restrictions>()?;
    m.add_class::<types::Problem>()?;
    m.add_class::<solve::Solution>()?;
    m.add_class::<solve::Frontier>()?;
    m.add_class::<solve::FrontierPoint>()?;
    m.add_function(wrap_pyfunction!(solve::solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve::solve_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve::sweep, m)?)?;
    m.add_function(wrap_pyfunction!(solve::simplex_grid, m)?)?;
    Ok(())
}
