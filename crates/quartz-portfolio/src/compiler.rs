use std::collections::HashSet;

use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT;

use quartz_core::{DimensionType, Universe};
use quartz_solver::CompiledProblem;

use crate::constraints::allocation::FullyInvested;
use crate::constraints::bounds::WeightBounds;
use crate::constraints::factor::FactorLink;
use crate::constraints::turnover::TurnoverConstraint;
use crate::constraints::ConstraintContribution;
use crate::restriction::Restrictions;
use crate::strategy::Strategy;
use crate::tactic::{self, MergedStrategy, Tactic};

/// Errors during compilation.
#[derive(Debug)]
pub enum CompileError {
    /// Tactic merge produced empty intervals.
    MergeError(tactic::MergeError),
    /// No dimensions specified in strategy.
    NoDimensions,
    /// Quadratic dimension requested but no covariance available.
    NoCovarianceForQuadratic,
    /// Turnover constraint dimension mismatch.
    TurnoverDimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::MergeError(e) => write!(f, "Merge error: {e}"),
            CompileError::NoDimensions => write!(f, "Strategy has no optimization dimensions"),
            CompileError::NoCovarianceForQuadratic => {
                write!(f, "Quadratic dimension requires a covariance model")
            }
            CompileError::TurnoverDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Turnover previous_weights length ({got}) != n_assets ({expected})"
                )
            }
        }
    }
}

impl std::error::Error for CompileError {}

impl From<tactic::MergeError> for CompileError {
    fn from(e: tactic::MergeError) -> Self {
        CompileError::MergeError(e)
    }
}

/// Compile a portfolio optimization problem into a QP for the solver.
///
/// This is the central function that translates high-level portfolio concepts
/// (strategy, tactic, restrictions) into the mathematical form:
///
///   min (1/2)xᵀPx + qᵀx  s.t. Ax + s = b, s ∈ K
///
/// where K is a product of zero cones (equalities) and nonnegative cones (inequalities).
pub fn compile(
    universe: &Universe,
    strategy: &Strategy,
    tactic: Option<&Tactic>,
    restrictions: &Restrictions,
    turnover: Option<&TurnoverConstraint>,
) -> Result<CompiledProblem, CompileError> {
    let n = universe.n_assets();

    if strategy.dimensions.is_empty() {
        return Err(CompileError::NoDimensions);
    }

    // Variable layout: [w (n), t (n, if turnover), y (k, if factor covariance)].
    // The turnover constraint hardcodes t at columns n..2n, so y must come after t.
    // y vars are only allocated when a quadratic dimension is present: a free
    // variable appearing in no constraint and no objective can make the solver's
    // KKT system singular. (Tactic merge can change dimension weights but never
    // dimension types, so checking the strategy is sufficient.)
    let has_quadratic = strategy
        .dimensions
        .iter()
        .any(|d| matches!(d.dim_type, DimensionType::Quadratic));
    let n_turnover = if turnover.is_some() { n } else { 0 };
    let k_factors = match &universe.covariance {
        quartz_core::CovarianceModel::Factor { loadings, .. } if has_quadratic => loadings.n,
        _ => 0,
    };
    let y_offset = n + n_turnover;
    let n_aux = n_turnover + k_factors;
    let n_vars = n + n_aux;

    // Validate turnover dimensions
    if let Some(tc) = turnover {
        if tc.previous_weights.len() != n {
            return Err(CompileError::TurnoverDimensionMismatch {
                expected: n,
                got: tc.previous_weights.len(),
            });
        }
    }

    // Step 1: Merge strategy + tactic
    let merged = tactic::merge(strategy, tactic)?;

    // Step 2: Compute excluded asset indices
    let excluded: HashSet<usize> = restrictions
        .exclusions
        .iter()
        .flat_map(|e| e.excluded_indices(universe))
        .collect();

    // Step 3: Build objective (P, q)
    let (p, q) = build_objective(universe, &merged, n_vars, y_offset)?;

    // Step 4: Collect all constraints
    let mut eq_contributions: Vec<ConstraintContribution> = Vec::new();
    let mut ineq_contributions: Vec<ConstraintContribution> = Vec::new();

    // 4a: Fully invested (equality)
    if merged.fully_invested || restrictions.fully_invested {
        eq_contributions.push(FullyInvested.compile(n));
    }

    // 4b: Group constraints (inequalities)
    for gc in &merged.group_constraints {
        ineq_contributions.push(gc.compile(universe));
    }

    // 4c: Score constraints (inequalities)
    for sc in &merged.score_constraints {
        ineq_contributions.push(sc.compile(universe));
    }

    // 4d: Weight bounds (long-only and/or max single weight)
    if restrictions.long_only {
        ineq_contributions.push(WeightBounds::long_only().compile(n));
    }
    if let Some(max_w) = restrictions.max_single_weight {
        ineq_contributions.push(WeightBounds::max_weight(max_w).compile(n));
    }

    // 4e: Exclusions (force weight to 0 via equality constraint)
    if !excluded.is_empty() {
        let mut contrib = ConstraintContribution::new();
        for &i in &excluded {
            contrib.triplets.push(crate::constraints::Triplet {
                row: contrib.n_equality,
                col: i,
                val: 1.0,
            });
            contrib.b_entries.push(0.0);
            contrib.n_equality += 1;
        }
        eq_contributions.push(contrib);
    }

    // 4f: Turnover constraints (inequalities, using auxiliary columns n..2n)
    if let Some(tc) = turnover {
        ineq_contributions.push(tc.compile(n));
    }

    // 4g: Factor link equalities y = Bᵀw (only when factor covariance + quadratic dim)
    if k_factors > 0 {
        if let quartz_core::CovarianceModel::Factor { loadings, .. } = &universe.covariance {
            eq_contributions.push(FactorLink.compile(loadings, y_offset));
        }
    }

    // Step 5: Assemble A and b matrices
    //
    // Row ordering: all equalities first (ZeroConeT), then all inequalities (NonnegativeConeT).
    let total_eq: usize = eq_contributions.iter().map(|c| c.n_equality).sum();
    let total_ineq: usize = ineq_contributions.iter().map(|c| c.n_inequality).sum();
    let m = total_eq + total_ineq;

    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();
    let mut b = Vec::with_capacity(m);

    // Equalities first
    let mut row_offset = 0;
    for contrib in &eq_contributions {
        for t in &contrib.triplets {
            a_rows.push(row_offset + t.row);
            a_cols.push(t.col);
            a_vals.push(t.val);
        }
        b.extend_from_slice(&contrib.b_entries);
        row_offset += contrib.n_rows();
    }

    // Then inequalities
    for contrib in &ineq_contributions {
        for t in &contrib.triplets {
            a_rows.push(row_offset + t.row);
            a_cols.push(t.col);
            a_vals.push(t.val);
        }
        b.extend_from_slice(&contrib.b_entries);
        row_offset += contrib.n_rows();
    }

    let a = CscMatrix::new_from_triplets(m, n_vars, a_rows, a_cols, a_vals);

    // Step 6: Build cones
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if total_eq > 0 {
        cones.push(SupportedConeT::ZeroConeT(total_eq));
    }
    if total_ineq > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(total_ineq));
    }

    Ok(CompiledProblem {
        p,
        q,
        a,
        b,
        cones,
        n_assets: n,
        n_aux,
    })
}

/// Build the objective P (quadratic) and q (linear) from merged dimensions.
fn build_objective(
    universe: &Universe,
    merged: &MergedStrategy,
    n_vars: usize,
    y_offset: usize,
) -> Result<(CscMatrix<f64>, Vec<f64>), CompileError> {
    let mut q = vec![0.0; n_vars];

    // Collect P contributions
    let mut p_triplet_rows = Vec::new();
    let mut p_triplet_cols = Vec::new();
    let mut p_triplet_vals = Vec::new();

    // Quadratic dimensions all share one covariance term, so their lambdas sum.
    let mut lambda_quad = 0.0;
    let mut has_quadratic = false;

    for dim in &merged.dimensions {
        match &dim.dim_type {
            DimensionType::Quadratic => {
                has_quadratic = true;
                lambda_quad += dim.weight * dim.sense.sign();
            }
            DimensionType::Linear { score_key } => {
                // q += λ * sense * score_vector
                let scores = universe.score_vector(score_key);
                let factor = dim.weight * dim.sense.sign();
                for (i, &s) in scores.iter().enumerate() {
                    q[i] += factor * s;
                }
            }
        }
    }

    if has_quadratic {
        // P += λ * Σ (only upper triangle for Clarabel)
        add_covariance_to_p(
            universe,
            lambda_quad,
            y_offset,
            &mut p_triplet_rows,
            &mut p_triplet_cols,
            &mut p_triplet_vals,
        );
    }

    let p = if p_triplet_rows.is_empty() {
        // No quadratic term — create a zero matrix
        CscMatrix::zeros((n_vars, n_vars))
    } else {
        // Pad to n_vars x n_vars (aux variables have no quadratic cost)
        CscMatrix::new_from_triplets(n_vars, n_vars, p_triplet_rows, p_triplet_cols, p_triplet_vals)
    };

    Ok((p, q))
}

/// Extract the covariance and add λ*Σ to the P matrix triplets (upper triangle only).
///
/// For the factor model Σ = BFBᵀ + D, the quadratic form is expressed through
/// auxiliary variables y = Bᵀw (linked by equality rows added in `compile`):
/// wᵀΣw = yᵀFy + wᵀDw, so P gets λ·diag(D) on the w block and λ·F on the
/// y block at `y_offset` — n + k(k+1)/2 nonzeros instead of n².
fn add_covariance_to_p(
    universe: &Universe,
    lambda: f64,
    y_offset: usize,
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<f64>,
) {
    match &universe.covariance {
        quartz_core::CovarianceModel::Full(cov) => {
            // Extract triplets from CscMatrix, keeping only upper triangle
            for j in 0..cov.n {
                let col_start = cov.colptr[j];
                let col_end = cov.colptr[j + 1];
                for idx in col_start..col_end {
                    let i = cov.rowval[idx];
                    let v = cov.nzval[idx];
                    if i <= j {
                        // Upper triangle only
                        rows.push(i);
                        cols.push(j);
                        vals.push(lambda * v);
                    }
                }
            }
        }
        quartz_core::CovarianceModel::Factor {
            factor_cov,
            specific_variance,
            ..
        } => {
            // P += λ·D on the w block (diagonal)
            for (i, &d) in specific_variance.iter().enumerate() {
                if d != 0.0 {
                    rows.push(i);
                    cols.push(i);
                    vals.push(lambda * d);
                }
            }
            // P += λ·F on the y block (upper triangle; F stored full-symmetric)
            for j in 0..factor_cov.n {
                for idx in factor_cov.colptr[j]..factor_cov.colptr[j + 1] {
                    let i = factor_cov.rowval[idx];
                    if i <= j {
                        rows.push(y_offset + i);
                        cols.push(y_offset + j);
                        vals.push(lambda * factor_cov.nzval[idx]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quartz_core::Asset;

    fn test_universe() -> Universe {
        Universe::builder()
            .add_asset(
                Asset::new("A")
                    .tag("currency", "USD")
                    .score("expected_return", 0.10)
                    .score("esg", 8.0),
            )
            .add_asset(
                Asset::new("B")
                    .tag("currency", "EUR")
                    .score("expected_return", 0.05)
                    .score("esg", 5.0),
            )
            .add_asset(
                Asset::new("C")
                    .tag("currency", "EUR")
                    .score("expected_return", 0.03)
                    .score("esg", 3.0),
            )
            .covariance_full(CscMatrix::from(&[
                [0.04, 0.006, 0.002],
                [0.006, 0.09, 0.009],
                [0.002, 0.009, 0.01],
            ]))
            .build()
            .unwrap()
    }

    #[test]
    fn test_compile_basic_markowitz() {
        let universe = test_universe();
        let strategy = Strategy::builder("MinVar")
            .minimize_risk(1.0)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        assert_eq!(problem.n_assets, 3);
        assert_eq!(problem.n_aux, 0);
        assert_eq!(problem.n_vars(), 3);
        // 1 equality (fully invested) + 3 inequalities (long only)
        assert_eq!(problem.b.len(), 4);
    }

    #[test]
    fn test_compile_with_groups() {
        let universe = test_universe();
        let strategy = Strategy::builder("Balanced")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .group("currency", "USD", 0.10, 0.40)
            .group("currency", "EUR", 0.60, 0.90)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // 1 eq (fully_invested) + 2*2 ineq (groups) + 3 ineq (long_only)
        assert_eq!(problem.b.len(), 1 + 4 + 3);
    }

    #[test]
    fn test_compile_with_exclusion() {
        let universe = test_universe();
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder()
            .long_only()
            .fully_invested()
            .exclude_asset("B")
            .build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // 1 eq (fully_invested) + 1 eq (exclusion B) + 3 ineq (long_only)
        assert_eq!(problem.b.len(), 2 + 3);
    }

    #[test]
    fn test_compile_with_turnover() {
        let universe = test_universe();
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 0.20);

        let problem =
            compile(&universe, &strategy, None, &restrictions, Some(&turnover)).unwrap();

        assert_eq!(problem.n_aux, 3);
        assert_eq!(problem.n_vars(), 6);
        // 1 eq + 3 ineq (long only) + 10 ineq (turnover: 3*3+1)
        assert_eq!(problem.b.len(), 1 + 3 + 10);
    }

    fn factor_universe() -> Universe {
        // 3 assets, 2 factors
        Universe::builder()
            .add_asset(Asset::new("A").score("expected_return", 0.10))
            .add_asset(Asset::new("B").score("expected_return", 0.05))
            .add_asset(Asset::new("C").score("expected_return", 0.03))
            .covariance_factor(
                CscMatrix::from(&[[1.0, 0.2], [0.8, -0.1], [0.5, 0.7]]),
                CscMatrix::from(&[[0.04, 0.01], [0.01, 0.02]]),
                vec![0.01, 0.02, 0.015],
            )
            .build()
            .unwrap()
    }

    #[test]
    fn test_compile_factor_model() {
        let universe = factor_universe();
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // 2 factor aux vars (y), no turnover
        assert_eq!(problem.n_assets, 3);
        assert_eq!(problem.n_aux, 2);
        assert_eq!(problem.n_vars(), 5);
        // 1 eq (fully invested) + 2 eq (factor links) + 3 ineq (long only)
        assert_eq!(problem.b.len(), 6);
        assert!(matches!(problem.cones[0], SupportedConeT::ZeroConeT(3)));
        assert!(matches!(problem.cones[1], SupportedConeT::NonnegativeConeT(3)));
        // P is block diagonal: 3 diagonal D entries + 3 upper-triangle F entries
        assert_eq!(problem.p.nzval.len(), 6);
    }

    #[test]
    fn test_compile_factor_with_turnover() {
        let universe = factor_universe();
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 0.20);

        let problem =
            compile(&universe, &strategy, None, &restrictions, Some(&turnover)).unwrap();

        // aux = 3 turnover t vars + 2 factor y vars; layout [w, t, y]
        assert_eq!(problem.n_aux, 5);
        assert_eq!(problem.n_vars(), 8);
        // 1 eq + 2 eq (factor) + 3 ineq (long only) + 10 ineq (turnover)
        assert_eq!(problem.b.len(), 16);
    }

    #[test]
    fn test_compile_factor_linear_only_no_aux() {
        // Without a quadratic dimension, no y vars or link rows must be created
        // (dead free variables can make the KKT system singular).
        let universe = factor_universe();
        let strategy = Strategy::builder("ReturnOnly")
            .maximize("expected_return", 1.0)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        assert_eq!(problem.n_aux, 0);
        assert_eq!(problem.n_vars(), 3);
        // 1 eq (fully invested) + 3 ineq (long only)
        assert_eq!(problem.b.len(), 4);
    }

    #[test]
    fn test_compile_no_dimensions_error() {
        let universe = test_universe();
        let strategy = Strategy {
            name: "Empty".into(),
            dimensions: vec![],
            group_constraints: vec![],
            score_constraints: vec![],
            fully_invested: true,
        };
        let restrictions = Restrictions::default();

        let result = compile(&universe, &strategy, None, &restrictions, None);
        assert!(result.is_err());
    }
}
