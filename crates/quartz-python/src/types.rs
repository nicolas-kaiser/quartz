//! Python-facing data types: Asset, Universe, Strategy, Tactic, Restrictions,
//! and Problem (a batch item).

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::convert::Matrix;
use crate::qerr;

/// An investable asset with categorical tags and numerical scores.
// from_py_object: Universe's constructor extracts Vec<Asset> by clone.
#[pyclass(from_py_object, module = "quartz")]
#[derive(Clone)]
pub struct Asset {
    pub(crate) inner: quartz_core::Asset,
}

#[pymethods]
impl Asset {
    #[new]
    #[pyo3(signature = (id, tags=None, scores=None))]
    fn new(
        id: &str,
        tags: Option<HashMap<String, String>>,
        scores: Option<HashMap<String, f64>>,
    ) -> Self {
        let mut inner = quartz_core::Asset::new(id);
        for (k, v) in tags.unwrap_or_default() {
            inner = inner.tag(k, v);
        }
        for (k, v) in scores.unwrap_or_default() {
            inner = inner.score(k, v);
        }
        Self { inner }
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Asset('{}')", self.inner.id)
    }
}

/// The investment universe: assets + covariance structure.
///
/// Provide exactly one of:
/// - `covariance`: full n×n matrix (numpy 2D float64 array or nested lists),
///   stored full-symmetric;
/// - `factor_model`: `(loadings n×k, factor_cov k×k, specific_variance len n)`
///   for Σ = BFBᵀ + D (factor_cov full-symmetric, specific variances ≥ 0).
#[pyclass(skip_from_py_object, module = "quartz")]
#[derive(Clone)]
pub struct Universe {
    pub(crate) inner: quartz_core::Universe,
}

#[pymethods]
impl Universe {
    #[new]
    #[pyo3(signature = (assets, covariance=None, factor_model=None))]
    fn new(
        assets: Vec<Asset>,
        covariance: Option<Matrix<'_>>,
        factor_model: Option<(Matrix<'_>, Matrix<'_>, Vec<f64>)>,
    ) -> PyResult<Self> {
        let mut builder = quartz_core::Universe::builder();
        for a in assets {
            builder = builder.add_asset(a.inner);
        }
        let builder = match (&covariance, &factor_model) {
            (Some(c), None) => builder.covariance_full(c.to_csc()?),
            (None, Some((loadings, factor_cov, specific))) => builder.covariance_factor(
                loadings.to_csc()?,
                factor_cov.to_csc()?,
                specific.clone(),
            ),
            _ => return Err(qerr("provide exactly one of covariance or factor_model")),
        };
        Ok(Self {
            inner: builder.build().map_err(qerr)?,
        })
    }

    #[getter]
    fn n_assets(&self) -> usize {
        self.inner.n_assets()
    }

    #[getter]
    fn asset_ids(&self) -> Vec<String> {
        self.inner.assets.iter().map(|a| a.id.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        format!("Universe({} assets)", self.inner.n_assets())
    }
}

/// A portfolio strategy, built by method chaining:
///
/// ```python
/// s = (quartz.Strategy("ESG")
///      .minimize_risk(0.4)
///      .maximize("expected_return", 0.3)
///      .group("currency", "EUR", 0.3, 0.6)
///      .score_min("environmental_impact", 7.0))
/// ```
#[pyclass(module = "quartz")]
pub struct Strategy {
    builder: quartz_portfolio::strategy::StrategyBuilder,
}

impl Strategy {
    pub(crate) fn to_strategy(&self) -> quartz_portfolio::Strategy {
        self.builder.clone().build()
    }
}

#[pymethods]
impl Strategy {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            builder: quartz_portfolio::Strategy::builder(name),
        }
    }

    /// Add the quadratic risk dimension (minimize portfolio variance).
    fn minimize_risk(mut slf: PyRefMut<'_, Self>, weight: f64) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().minimize_risk(weight);
        slf
    }

    /// Add a linear dimension to maximize (e.g. expected return, ESG score).
    fn maximize<'py>(
        mut slf: PyRefMut<'py, Self>,
        score_key: &str,
        weight: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf.builder.clone().maximize(score_key, weight);
        slf
    }

    /// Add a linear dimension to minimize (e.g. transition risk).
    fn minimize<'py>(
        mut slf: PyRefMut<'py, Self>,
        score_key: &str,
        weight: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf.builder.clone().minimize(score_key, weight);
        slf
    }

    /// Constrain total weight of assets with a tag value to [lower, upper].
    fn group<'py>(
        mut slf: PyRefMut<'py, Self>,
        tag_key: &str,
        tag_value: &str,
        lower: f64,
        upper: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf.builder.clone().group(tag_key, tag_value, lower, upper);
        slf
    }

    /// Require the weighted portfolio score to be at least `threshold`.
    fn score_min<'py>(
        mut slf: PyRefMut<'py, Self>,
        score_key: &str,
        threshold: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf.builder.clone().score_min(score_key, threshold);
        slf
    }

    /// Require the weighted portfolio score to be at most `threshold`.
    fn score_max<'py>(
        mut slf: PyRefMut<'py, Self>,
        score_key: &str,
        threshold: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf.builder.clone().score_max(score_key, threshold);
        slf
    }

    /// Set whether the portfolio must be fully invested (default True).
    fn fully_invested(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.builder = slf.builder.clone().fully_invested(value);
        slf
    }
}

/// A tactical overlay: tightens a strategy's bounds and overrides weights.
#[pyclass(module = "quartz")]
pub struct Tactic {
    builder: quartz_portfolio::tactic::TacticBuilder,
}

impl Tactic {
    pub(crate) fn to_tactic(&self) -> quartz_portfolio::Tactic {
        self.builder.clone().build()
    }
}

#[pymethods]
impl Tactic {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            builder: quartz_portfolio::Tactic::builder(name),
        }
    }

    /// Override a group bound; merged with the strategy by intersection.
    fn override_group<'py>(
        mut slf: PyRefMut<'py, Self>,
        tag_key: &str,
        tag_value: &str,
        lower: f64,
        upper: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf
            .builder
            .clone()
            .override_group(tag_key, tag_value, lower, upper);
        slf
    }

    /// Override a dimension's weight by name.
    fn override_weight<'py>(
        mut slf: PyRefMut<'py, Self>,
        dimension_name: &str,
        weight: f64,
    ) -> PyRefMut<'py, Self> {
        slf.builder = slf.builder.clone().override_weight(dimension_name, weight);
        slf
    }
}

/// Hard compliance constraints.
#[pyclass(skip_from_py_object, module = "quartz")]
#[derive(Clone)]
pub struct Restrictions {
    pub(crate) inner: quartz_portfolio::Restrictions,
}

#[pymethods]
impl Restrictions {
    #[new]
    #[pyo3(signature = (long_only=false, fully_invested=false, max_single_weight=None,
                        exclude_assets=None, exclude_tags=None))]
    fn new(
        long_only: bool,
        fully_invested: bool,
        max_single_weight: Option<f64>,
        exclude_assets: Option<Vec<String>>,
        exclude_tags: Option<Vec<(String, String)>>,
    ) -> Self {
        let mut builder = quartz_portfolio::Restrictions::builder();
        if long_only {
            builder = builder.long_only();
        }
        if fully_invested {
            builder = builder.fully_invested();
        }
        if let Some(max_w) = max_single_weight {
            builder = builder.max_single_weight(max_w);
        }
        for id in exclude_assets.unwrap_or_default() {
            builder = builder.exclude_asset(id.as_str());
        }
        for (k, v) in exclude_tags.unwrap_or_default() {
            builder = builder.exclude_tag(k, v);
        }
        Self {
            inner: builder.build(),
        }
    }
}

/// One item of a `solve_batch` call: a universe/strategy pair with optional
/// per-item tactic, restrictions, and turnover.
#[pyclass(module = "quartz")]
pub struct Problem {
    pub(crate) universe: quartz_core::Universe,
    pub(crate) strategy: quartz_portfolio::Strategy,
    pub(crate) tactic: Option<quartz_portfolio::Tactic>,
    pub(crate) restrictions: Option<quartz_portfolio::Restrictions>,
    pub(crate) turnover: Option<(Vec<f64>, f64)>,
}

#[pymethods]
impl Problem {
    #[new]
    #[pyo3(signature = (universe, strategy, *, tactic=None, restrictions=None, turnover=None))]
    fn new(
        universe: &Universe,
        strategy: &Strategy,
        tactic: Option<&Tactic>,
        restrictions: Option<&Restrictions>,
        turnover: Option<(Vec<f64>, f64)>,
    ) -> Self {
        Self {
            universe: universe.inner.clone(),
            strategy: strategy.to_strategy(),
            tactic: tactic.map(|t| t.to_tactic()),
            restrictions: restrictions.map(|r| r.inner.clone()),
            turnover,
        }
    }
}
