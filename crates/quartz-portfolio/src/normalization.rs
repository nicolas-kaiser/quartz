//! Range normalisation of dimension weights.
//!
//! Dimension values come in unrelated units: a variance around 0.01, a return
//! around 0.1, an ESG score between 0 and 10. Weighting the raw values lets
//! the dimension with the largest numbers decide the portfolio, whatever its
//! weight. With [`Normalization::Range`](crate::strategy::Normalization) each
//! dimension is first divided by its **feasible range**, so a weight is a
//! share of a 0–1 trade-off.
//!
//! The range comes from a payoff table: one single-objective solve per
//! dimension over the problem's own feasible set (the *anchor* solves).
//! `best` is the dimension's value at its own anchor, `worst` its worst value
//! over all anchors. See `docs/dimension-normalization.md`.

use serde::{Deserialize, Serialize};

use quartz_core::{Dimension, DimensionType, Universe};
use quartz_solver::{Backend, SolverSettings};

use crate::constraints::{TransactionCosts, TurnoverConstraint};
use crate::model::{PortfolioError, PortfolioModel};
use crate::restriction::Restrictions;
use crate::solution::SolveStatus;
use crate::strategy::{Normalization, Strategy};

/// The feasible range of one dimension, in the units of its reported score
/// (`financial_risk` is a variance).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionRange {
    /// Name of the dimension.
    pub name: String,
    /// Value when this dimension alone is optimised.
    pub best: f64,
    /// Worst value over the single-objective optima of all dimensions.
    pub worst: f64,
}

impl DimensionRange {
    /// The divisor applied to the dimension: `|worst − best|`.
    ///
    /// A range of (numerically) zero means the dimension does not trade off
    /// against the others; it keeps its raw coefficient (divisor 1).
    pub fn scale(&self) -> f64 {
        let range = (self.worst - self.best).abs();
        let tol = 1e-12 * self.best.abs().max(self.worst.abs()).max(1.0);
        if range.is_finite() && range > tol {
            range
        } else {
            1.0
        }
    }
}

/// The `portfolio_scores` key a dimension is measured by.
pub(crate) fn metric_key(dim: &Dimension) -> &str {
    match &dim.dim_type {
        DimensionType::Quadratic => "financial_risk",
        DimensionType::Linear { score_key } => score_key,
    }
}

/// Compute the feasible range of every dimension of `base` (a strategy with
/// the tactic already merged in).
///
/// Dimensions with weight 0 get no anchor solve unless `include_unweighted`
/// is set (the frontier sweeps weights, so it needs them all); their entry is
/// `best = worst = 0`, i.e. divisor 1.
///
/// Returns `Ok(None)` when there is nothing to normalise: fewer than two
/// anchored dimensions, or an infeasible problem (the caller's own solve then
/// reports the infeasibility).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dimension_ranges(
    universe: &Universe,
    base: &Strategy,
    restrictions: &Restrictions,
    turnover: Option<&TurnoverConstraint>,
    costs: Option<&TransactionCosts>,
    settings: &SolverSettings,
    backend: Backend,
    include_unweighted: bool,
) -> Result<Option<Vec<DimensionRange>>, PortfolioError> {
    let anchored: Vec<usize> = base
        .dimensions
        .iter()
        .enumerate()
        .filter(|(_, d)| include_unweighted || d.weight != 0.0)
        .map(|(i, _)| i)
        .collect();
    if anchored.len() < 2 {
        return Ok(None);
    }

    // One solve per anchored dimension: that dimension at weight 1, the
    // others at 0 — kept, not removed, so the quadratic dimension still
    // allocates and links the factor-model variables.
    let solved = crate::par::par_map(&anchored, |&k| {
        let mut anchor = base.clone();
        anchor.normalization = Normalization::Raw;
        for (i, d) in anchor.dimensions.iter_mut().enumerate() {
            d.weight = if i == k { 1.0 } else { 0.0 };
        }
        let mut model = PortfolioModel::new(universe)
            .strategy(&anchor)
            .restrictions(restrictions.clone())
            .solver_settings(settings.clone())
            .backend(backend);
        if let Some(t) = turnover {
            model = model.turnover(t.previous_weights.clone(), t.max_turnover);
        }
        if let Some(c) = costs {
            model = model.transaction_costs(c.rates.clone(), c.score_key.clone());
        }
        model.solve()
    });

    let mut anchors = Vec::with_capacity(anchored.len());
    for (&k, result) in anchored.iter().zip(solved) {
        let sol = result?;
        match sol.status {
            SolveStatus::Optimal | SolveStatus::AlmostOptimal => anchors.push(sol),
            // Same feasible set as the real problem: let its solve report it.
            SolveStatus::Infeasible => return Ok(None),
            status => {
                return Err(PortfolioError::Normalization {
                    dimension: base.dimensions[k].name.clone(),
                    status,
                })
            }
        }
    }

    let ranges = base
        .dimensions
        .iter()
        .enumerate()
        .map(|(i, dim)| {
            let Some(own) = anchored.iter().position(|&k| k == i) else {
                return DimensionRange {
                    name: dim.name.clone(),
                    best: 0.0,
                    worst: 0.0,
                };
            };
            let value = |a: &crate::solution::PortfolioSolution| {
                a.portfolio_scores.get(metric_key(dim)).copied().unwrap_or(0.0)
            };
            let sign = dim.sense.sign();
            let worst = anchors
                .iter()
                .map(value)
                .fold(f64::NEG_INFINITY, |acc, v| acc.max(sign * v))
                * sign;
            DimensionRange {
                name: dim.name.clone(),
                best: value(&anchors[own]),
                worst,
            }
        })
        .collect();
    Ok(Some(ranges))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_is_the_absolute_range() {
        let r = DimensionRange { name: "x".into(), best: 9.5, worst: 6.5 };
        assert!((r.scale() - 3.0).abs() < 1e-12);
        let r = DimensionRange { name: "x".into(), best: 0.003, worst: 0.01 };
        assert!((r.scale() - 0.007).abs() < 1e-12);
    }

    #[test]
    fn test_zero_range_keeps_the_raw_coefficient() {
        let r = DimensionRange { name: "x".into(), best: 5.0, worst: 5.0 };
        assert_eq!(r.scale(), 1.0);
        let r = DimensionRange { name: "x".into(), best: 0.0, worst: 0.0 };
        assert_eq!(r.scale(), 1.0);
        let r = DimensionRange { name: "x".into(), best: f64::NAN, worst: 1.0 };
        assert_eq!(r.scale(), 1.0);
    }
}
