use std::collections::BTreeSet;

use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT;

use quartz_core::{DimensionType, Sense, Universe};
use quartz_solver::CompiledProblem;

use crate::constraints::allocation::FullyInvested;
use crate::constraints::bounds::WeightBounds;
use crate::constraints::factor::FactorLink;
use crate::constraints::turnover::TurnoverConstraint;
use crate::constraints::TransactionCosts;
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
    /// A group constraint matches no asset and its bounds exclude 0, so it
    /// can never be satisfied.
    EmptyGroup {
        tag_key: String,
        tag_value: String,
        lower: f64,
        upper: f64,
    },
    /// Quadratic dimension requested but no covariance available.
    NoCovarianceForQuadratic,
    /// Turnover constraint dimension mismatch.
    TurnoverDimensionMismatch { expected: usize, got: usize },
    /// Transaction costs need a turnover constraint (they price its t variables).
    TransactionCostsWithoutTurnover,
    /// Transaction cost rates length mismatch.
    TransactionCostDimensionMismatch { expected: usize, got: usize },
    /// A transaction cost rate is negative or not finite.
    InvalidTransactionCost { index: usize, value: f64 },
    /// A CVaR constraint requires scenarios on the universe.
    CvarRequiresScenarios,
    /// CVaR alpha must be in (0, 1) with a finite limit.
    InvalidCvarAlpha(f64),
    /// Tracking-error benchmark weights length mismatch.
    BenchmarkDimensionMismatch { expected: usize, got: usize },
    /// Tracking-error limit must be finite and positive.
    InvalidTrackingErrorLimit(f64),
    /// Covariance is not positive semidefinite (tracking error needs a Cholesky factor).
    CholeskyFailure(String),
    /// The universe failed validation (e.g. constructed via its public fields,
    /// bypassing `UniverseBuilder`).
    InvalidUniverse(quartz_core::UniverseError),
    /// A score key that no asset of the universe carries.
    UnknownScoreKey {
        score_key: String,
        /// What refers to it, e.g. "dimension 'esg'".
        used_by: String,
        /// The score keys the universe does have.
        available: Vec<String>,
    },
    /// A score that is optimised or bounded is missing on assets that can be
    /// held (a missing value would count as 0).
    MissingScores {
        score_key: String,
        used_by: String,
        /// Ids of the non-excluded assets without the score.
        assets: Vec<String>,
    },
    /// A group constraint or tag exclusion uses a tag key no asset carries.
    UnknownTagKey {
        tag_key: String,
        tag_value: String,
        used_by: String,
        available: Vec<String>,
    },
    /// A tag value matches no asset but differs from an existing value only
    /// by case or surrounding spaces.
    TagValueNearMiss {
        tag_key: String,
        tag_value: String,
        existing: String,
        used_by: String,
    },
    /// Exclusions name asset ids that are not in the universe.
    UnknownExcludedAssets { ids: Vec<String> },
    /// An excluded asset id is not in the universe but differs from an
    /// existing id only by case or surrounding spaces.
    ExcludedAssetNearMiss { id: String, existing: String },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::MergeError(e) => write!(f, "Merge error: {e}"),
            CompileError::NoDimensions => write!(f, "Strategy has no optimization dimensions"),
            CompileError::EmptyGroup { tag_key, tag_value, lower, upper } => {
                write!(
                    f,
                    "Group ({tag_key}, {tag_value}) matches no asset, so its weight is 0, \
                     outside the required [{lower}, {upper}]"
                )
            }
            CompileError::NoCovarianceForQuadratic => {
                write!(f, "Quadratic dimension requires a covariance model")
            }
            CompileError::TurnoverDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Turnover previous_weights length ({got}) != n_assets ({expected})"
                )
            }
            CompileError::TransactionCostsWithoutTurnover => {
                write!(f, "Transaction costs require a turnover constraint")
            }
            CompileError::InvalidTransactionCost { index, value } => {
                write!(
                    f,
                    "Transaction cost rate {index} must be finite and >= 0 (got {value})"
                )
            }
            CompileError::TransactionCostDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Transaction cost rates length ({got}) != n_assets ({expected})"
                )
            }
            CompileError::CvarRequiresScenarios => {
                write!(f, "CVaR constraint requires return scenarios on the universe")
            }
            CompileError::InvalidCvarAlpha(a) => {
                write!(f, "CVaR alpha must be in (0, 1) with a finite limit, got {a}")
            }
            CompileError::BenchmarkDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Tracking-error benchmark has {got} weights, expected {expected} (one per asset)"
                )
            }
            CompileError::InvalidTrackingErrorLimit(t) => {
                write!(f, "Tracking-error limit must be finite and positive, got {t}")
            }
            CompileError::CholeskyFailure(msg) => {
                write!(f, "Tracking error needs a PSD covariance: {msg}")
            }
            CompileError::InvalidUniverse(e) => write!(f, "Invalid universe: {e}"),
            CompileError::UnknownScoreKey { score_key, used_by, available } => {
                write!(
                    f,
                    "No asset has a score '{score_key}' (used by {used_by}); scores in \
                     the universe: {}",
                    list(available)
                )?;
                match score_key.as_str() {
                    "financial_risk" => write!(
                        f,
                        ". Portfolio variance is not a score: use minimize_risk \
                         (a `type: quadratic` dimension)"
                    ),
                    "tracking_error" | "cvar" => write!(
                        f,
                        ". '{score_key}' is not a score: use the strategy's \
                         {score_key} constraint"
                    ),
                    _ => Ok(()),
                }
            }
            CompileError::MissingScores { score_key, used_by, assets } => write!(
                f,
                "Score '{score_key}' (used by {used_by}) is missing on {} asset(s) that \
                 can be held: {}. A missing score would count as 0; set the score or \
                 exclude the asset",
                assets.len(),
                list(assets)
            ),
            CompileError::UnknownTagKey { tag_key, tag_value, used_by, available } => write!(
                f,
                "No asset has a tag '{tag_key}' (used by {used_by} on \
                 {tag_key}={tag_value}); tags in the universe: {}",
                list(available)
            ),
            CompileError::TagValueNearMiss { tag_key, tag_value, existing, used_by } => write!(
                f,
                "No asset has {tag_key}='{tag_value}' (used by {used_by}), but \
                 '{existing}' exists; tag values are case-sensitive"
            ),
            CompileError::UnknownExcludedAssets { ids } => write!(
                f,
                "{} excluded asset id(s) are not in the universe: {}. Correct them, or \
                 set ignore_unknown_assets on the restrictions if the list is meant to \
                 cover more than this universe",
                ids.len(),
                list(ids)
            ),
            CompileError::ExcludedAssetNearMiss { id, existing } => write!(
                f,
                "Excluded asset id '{id}' is not in the universe, but '{existing}' is; \
                 asset ids are case-sensitive"
            ),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompileError::MergeError(e) => Some(e),
            CompileError::InvalidUniverse(e) => Some(e),
            _ => None,
        }
    }
}

/// A list of names for an error message, shortened after ten entries.
fn list(names: &[String]) -> String {
    const MAX: usize = 10;
    if names.is_empty() {
        return "(none)".to_string();
    }
    let shown = names.iter().take(MAX).cloned().collect::<Vec<_>>().join(", ");
    if names.len() > MAX {
        format!("{shown}, … ({} more)", names.len() - MAX)
    } else {
        shown
    }
}

/// Case- and whitespace-insensitive form of a name, to detect near-misses.
fn folded(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Reject names that match nothing, before they silently do nothing: score
/// keys, tag keys and excluded asset ids. See `docs/reference-validation.md`.
///
/// Takes the merged strategy's parts, so tactic groups and score bounds are
/// covered. Dimension weights are ignored on purpose: a dimension at weight 0
/// (anchor solves, frontier end points) is held to the same rules.
pub(crate) fn validate_references(
    universe: &Universe,
    dimensions: &[quartz_core::Dimension],
    group_constraints: &[crate::constraints::GroupConstraint],
    score_constraints: &[crate::constraints::ScoreConstraint],
    restrictions: &Restrictions,
    costs: Option<&TransactionCosts>,
) -> Result<(), CompileError> {
    use crate::constraints::exclusion::Exclusion;

    let tag_keys = universe.tag_keys();
    let check_tag = |tag_key: &str, tag_value: &str, used_by: &str| {
        if !tag_keys.iter().any(|k| k == tag_key) {
            return Err(CompileError::UnknownTagKey {
                tag_key: tag_key.to_string(),
                tag_value: tag_value.to_string(),
                used_by: used_by.to_string(),
                available: tag_keys.clone(),
            });
        }
        // A value absent from this universe is legitimate ("no Tobacco" where
        // there is none) unless it is an existing value misspelt in case.
        let values = universe.tag_values(tag_key);
        if !values.iter().any(|v| v == tag_value) {
            if let Some(existing) = values.iter().find(|v| folded(v) == folded(tag_value)) {
                return Err(CompileError::TagValueNearMiss {
                    tag_key: tag_key.to_string(),
                    tag_value: tag_value.to_string(),
                    existing: existing.clone(),
                    used_by: used_by.to_string(),
                });
            }
        }
        Ok(())
    };

    // Exclusions
    let mut unknown_ids: Vec<String> = Vec::new();
    for exclusion in &restrictions.exclusions {
        match exclusion {
            Exclusion::ByTag { tag_key, tag_value } => check_tag(tag_key, tag_value, "an exclusion")?,
            Exclusion::ByAsset(id) => {
                if universe.asset_index(id).is_some() {
                    continue;
                }
                if let Some(existing) = universe
                    .assets
                    .iter()
                    .find(|a| folded(a.id.as_str()) == folded(id.as_str()))
                {
                    return Err(CompileError::ExcludedAssetNearMiss {
                        id: id.to_string(),
                        existing: existing.id.to_string(),
                    });
                }
                if !unknown_ids.contains(&id.0) {
                    unknown_ids.push(id.0.clone());
                }
            }
        }
    }
    if !unknown_ids.is_empty() && !restrictions.ignore_unknown_assets {
        return Err(CompileError::UnknownExcludedAssets { ids: unknown_ids });
    }

    // Group constraints
    for gc in group_constraints {
        check_tag(&gc.tag_key, &gc.tag_value, "a group constraint")?;
    }

    // Score keys. An excluded asset is pinned to weight 0 and needs no score.
    let excluded: BTreeSet<usize> = restrictions
        .exclusions
        .iter()
        .flat_map(|e| e.excluded_indices(universe))
        .collect();
    let score_keys = universe.score_keys();
    let check_score = |score_key: &str, used_by: String, needs_every_asset: bool| {
        if !score_keys.iter().any(|k| k == score_key) {
            return Err(CompileError::UnknownScoreKey {
                score_key: score_key.to_string(),
                used_by,
                available: score_keys.clone(),
            });
        }
        if needs_every_asset {
            let missing: Vec<String> = universe
                .assets_missing_score(score_key)
                .into_iter()
                .filter(|i| !excluded.contains(i))
                .map(|i| universe.assets[i].id.to_string())
                .collect();
            if !missing.is_empty() {
                return Err(CompileError::MissingScores {
                    score_key: score_key.to_string(),
                    used_by,
                    assets: missing,
                });
            }
        }
        Ok(())
    };
    for dim in dimensions {
        if let DimensionType::Linear { score_key } = &dim.dim_type {
            check_score(score_key, format!("dimension '{}'", dim.name), true)?;
        }
    }
    for sc in score_constraints {
        check_score(&sc.score_key, "a score constraint".to_string(), true)?;
    }
    if let Some(tc) = costs {
        // Only the name matters here: the costs themselves are `rates`.
        check_score(&tc.score_key, "the transaction costs".to_string(), false)?;
    }
    Ok(())
}

impl From<tactic::MergeError> for CompileError {
    fn from(e: tactic::MergeError) -> Self {
        CompileError::MergeError(e)
    }
}

impl From<quartz_core::UniverseError> for CompileError {
    fn from(e: quartz_core::UniverseError) -> Self {
        CompileError::InvalidUniverse(e)
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
    compile_with_costs(universe, strategy, tactic, restrictions, turnover, None)
}

/// [`compile`] with transaction costs: the score `costs.score_key` becomes net
/// of costs wherever more of it is wanted (see [`TransactionCosts`]).
///
/// Like [`compile`], this applies the dimension weights to the raw values,
/// whatever `strategy.normalization` says: range normalisation needs solves
/// and is done by `PortfolioModel::solve`.
pub fn compile_with_costs(
    universe: &Universe,
    strategy: &Strategy,
    tactic: Option<&Tactic>,
    restrictions: &Restrictions,
    turnover: Option<&TurnoverConstraint>,
    costs: Option<&TransactionCosts>,
) -> Result<CompiledProblem, CompileError> {
    compile_scaled(universe, strategy, tactic, restrictions, turnover, costs, None)
}

/// [`compile_with_costs`] with one divisor per dimension (in the order of
/// `strategy.dimensions`): dimension k enters the objective as
/// `weight · sign · value / scales[k]`.
pub(crate) fn compile_scaled(
    universe: &Universe,
    strategy: &Strategy,
    tactic: Option<&Tactic>,
    restrictions: &Restrictions,
    turnover: Option<&TurnoverConstraint>,
    costs: Option<&TransactionCosts>,
    scales: Option<&[f64]>,
) -> Result<CompiledProblem, CompileError> {
    // A Universe built through its public fields bypasses UniverseBuilder's
    // checks; re-validate so a malformed universe fails here with a clear
    // error instead of an opaque solver failure.
    universe.validate()?;

    let n = universe.n_assets();

    if strategy.dimensions.is_empty() {
        return Err(CompileError::NoDimensions);
    }

    // Variable layout: [w (n), t (n, if turnover), y (k, if factor covariance),
    // ζ+u (1+S, if CVaR)]. The turnover constraint hardcodes t at columns
    // n..2n, so later blocks always come after it. y vars are only allocated
    // when a quadratic dimension is present: a free variable appearing in no
    // constraint and no objective can make the solver's KKT system singular.
    // (Tactic merge can change dimension weights but never dimension types or
    // the risk constraints, so checking the strategy is sufficient.)
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

    // Validate risk constraints and size the CVaR aux block
    if let Some(te) = &strategy.tracking_error {
        if te.benchmark_weights.len() != n {
            return Err(CompileError::BenchmarkDimensionMismatch {
                expected: n,
                got: te.benchmark_weights.len(),
            });
        }
        if !te.max_te.is_finite() || te.max_te <= 0.0 {
            return Err(CompileError::InvalidTrackingErrorLimit(te.max_te));
        }
    }
    let n_cvar = match &strategy.cvar {
        Some(c) => {
            if !(c.alpha > 0.0 && c.alpha < 1.0) || !c.max_cvar.is_finite() {
                return Err(CompileError::InvalidCvarAlpha(c.alpha));
            }
            let scenarios = universe
                .scenarios
                .as_ref()
                .ok_or(CompileError::CvarRequiresScenarios)?;
            crate::constraints::CvarConstraint::n_aux(scenarios.len())
        }
        None => 0,
    };
    let cvar_offset = n + n_turnover + k_factors;
    let n_aux = n_turnover + k_factors + n_cvar;
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
    if let Some(tc) = costs {
        if turnover.is_none() {
            return Err(CompileError::TransactionCostsWithoutTurnover);
        }
        if tc.rates.len() != n {
            return Err(CompileError::TransactionCostDimensionMismatch {
                expected: n,
                got: tc.rates.len(),
            });
        }
        if let Some((index, &value)) = tc
            .rates
            .iter()
            .enumerate()
            .find(|(_, c)| !c.is_finite() || **c < 0.0)
        {
            return Err(CompileError::InvalidTransactionCost { index, value });
        }
    }

    // Step 1: Merge strategy + tactic
    let merged = tactic::merge(strategy, tactic)?;

    // Names that match nothing are errors, not silent no-ops
    validate_references(
        universe,
        &merged.dimensions,
        &merged.group_constraints,
        &merged.score_constraints,
        restrictions,
        costs,
    )?;

    // Step 2: Compute excluded asset indices (sorted, so the exclusion rows —
    // and their duals — are always in asset order)
    let excluded: BTreeSet<usize> = restrictions
        .exclusions
        .iter()
        .flat_map(|e| e.excluded_indices(universe))
        .collect();

    // Step 3: Build objective (P, q)
    let (p, q) = build_objective(universe, &merged, n_vars, y_offset, costs, scales)?;

    // Step 4: Collect all constraints
    let mut eq_contributions: Vec<ConstraintContribution> = Vec::new();
    let mut ineq_contributions: Vec<ConstraintContribution> = Vec::new();

    // 4a: Fully invested (equality)
    if merged.fully_invested || restrictions.fully_invested {
        eq_contributions.push(FullyInvested.compile(n));
    }

    // 4b: Group constraints (inequalities). A group matching no asset has
    // weight 0: it adds no rows when its bounds allow 0, and is an error
    // otherwise (it could never be satisfied).
    for gc in &merged.group_constraints {
        if universe.asset_indices(&gc.tag_key, &gc.tag_value).is_empty()
            && (gc.lower > 0.0 || gc.upper < 0.0)
        {
            return Err(CompileError::EmptyGroup {
                tag_key: gc.tag_key.clone(),
                tag_value: gc.tag_value.clone(),
                lower: gc.lower,
                upper: gc.upper,
            });
        }
        ineq_contributions.push(gc.compile(universe));
    }

    // 4c: Score constraints (inequalities)
    for sc in &merged.score_constraints {
        let mut contrib = sc.compile(universe);
        if let Some(tc) = costs {
            tc.net_score_rows(sc, &mut contrib, n);
        }
        ineq_contributions.push(contrib);
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

    // 4h: CVaR rows (pure inequalities, using aux columns at cvar_offset)
    if let Some(cvar) = &merged.cvar {
        let scenarios = universe
            .scenarios
            .as_ref()
            .expect("validated above: CVaR requires scenarios");
        ineq_contributions.push(cvar.compile(scenarios, cvar_offset));
    }

    // 4i: Tracking error (second-order cone rows, placed after all inequalities)
    let mut soc_contributions: Vec<ConstraintContribution> = Vec::new();
    if let Some(te) = &merged.tracking_error {
        soc_contributions.push(
            te.compile(&universe.covariance)
                .map_err(|e| CompileError::CholeskyFailure(e.to_string()))?,
        );
    }

    // Step 5: Assemble A and b matrices
    //
    // Row ordering: all equalities first (ZeroConeT), then all inequalities
    // (NonnegativeConeT), then second-order cone blocks in order.
    let total_eq: usize = eq_contributions.iter().map(|c| c.n_equality).sum();
    let total_ineq: usize = ineq_contributions.iter().map(|c| c.n_inequality).sum();
    let total_soc: usize = soc_contributions.iter().map(|c| c.n_rows()).sum();
    let m = total_eq + total_ineq + total_soc;

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

    // Then SOC blocks (rows and cones in the same iteration order)
    for contrib in &soc_contributions {
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
    for contrib in &soc_contributions {
        for &dim in &contrib.soc_blocks {
            cones.push(SupportedConeT::SecondOrderConeT(dim));
        }
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
///
/// With `costs`, a linear dimension that maximises `costs.score_key` maximises
/// it net of costs: `q[n + i] −= factor·cᵢ` on the turnover variables t
/// (columns n..2n), in the dimension's own (normalised) weight.
///
/// With `scales`, dimension k's coefficient is divided by `scales[k]`
/// (a missing entry counts as 1).
fn build_objective(
    universe: &Universe,
    merged: &MergedStrategy,
    n_vars: usize,
    y_offset: usize,
    costs: Option<&TransactionCosts>,
    scales: Option<&[f64]>,
) -> Result<(CscMatrix<f64>, Vec<f64>), CompileError> {
    let n = universe.n_assets();
    let mut q = vec![0.0; n_vars];

    // Collect P contributions
    let mut p_triplet_rows = Vec::new();
    let mut p_triplet_cols = Vec::new();
    let mut p_triplet_vals = Vec::new();

    // Quadratic dimensions all share one covariance term, so their lambdas sum.
    let mut lambda_quad = 0.0;
    let mut has_quadratic = false;

    for (k, dim) in merged.dimensions.iter().enumerate() {
        let scale = scales.and_then(|s| s.get(k)).copied().unwrap_or(1.0);
        let factor = dim.weight * dim.sense.sign() / scale;
        match &dim.dim_type {
            DimensionType::Quadratic => {
                has_quadratic = true;
                lambda_quad += factor;
            }
            DimensionType::Linear { score_key } => {
                // q += λ * sense * score_vector
                let scores = universe.score_vector(score_key);
                for (i, &s) in scores.iter().enumerate() {
                    q[i] += factor * s;
                }
                if let Some(tc) = costs {
                    if *score_key == tc.score_key && matches!(dim.sense, Sense::Maximize) {
                        for (i, &c) in tc.rates.iter().enumerate() {
                            q[n + i] -= factor * c;
                        }
                    }
                }
            }
        }
    }

    if has_quadratic {
        // P = 2λ·Σ (only upper triangle for Clarabel): the solver minimises
        // ½xᵀPx, so the factor 2 makes the risk term exactly λ·wᵀΣw — on the
        // same scale as the linear terms λ·sᵀw.
        add_covariance_to_p(
            universe,
            2.0 * lambda_quad,
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

/// Extract the covariance and add `scale`·Σ to the P matrix triplets (upper
/// triangle only). The caller passes `scale` = 2λ (see `build_objective`).
///
/// For the factor model Σ = BFBᵀ + D, the quadratic form is expressed through
/// auxiliary variables y = Bᵀw (linked by equality rows added in `compile`):
/// wᵀΣw = yᵀFy + wᵀDw, so P gets scale·diag(D) on the w block and scale·F on
/// the y block at `y_offset` — n + k(k+1)/2 nonzeros instead of n².
fn add_covariance_to_p(
    universe: &Universe,
    scale: f64,
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
                        vals.push(scale * v);
                    }
                }
            }
        }
        quartz_core::CovarianceModel::Factor {
            factor_cov,
            specific_variance,
            ..
        } => {
            // P += scale·D on the w block (diagonal)
            for (i, &d) in specific_variance.iter().enumerate() {
                if d != 0.0 {
                    rows.push(i);
                    cols.push(i);
                    vals.push(scale * d);
                }
            }
            // P += scale·F on the y block (upper triangle; F stored full-symmetric)
            for j in 0..factor_cov.n {
                for idx in factor_cov.colptr[j]..factor_cov.colptr[j + 1] {
                    let i = factor_cov.rowval[idx];
                    if i <= j {
                        rows.push(y_offset + i);
                        cols.push(y_offset + j);
                        vals.push(scale * factor_cov.nzval[idx]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quartz_core::{Asset, CovarianceModel, UniverseError};

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
    fn test_compile_rejects_hand_constructed_invalid_universe() {
        let strategy = Strategy::builder("S").minimize_risk(1.0).build();
        let restrictions = Restrictions::default();
        let cov2 = || CovarianceModel::Full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]]));

        // Duplicate asset ids
        let dup = Universe {
            assets: vec![Asset::new("A"), Asset::new("A")],
            covariance: cov2(),
            scenarios: None,
        };
        let err = compile(&dup, &strategy, None, &restrictions, None).unwrap_err();
        assert!(matches!(
            err,
            CompileError::InvalidUniverse(UniverseError::DuplicateAssetId(_))
        ));

        // NaN score
        let nan_score = Universe {
            assets: vec![Asset::new("A").score("ret", f64::NAN), Asset::new("B")],
            covariance: cov2(),
            scenarios: None,
        };
        let err = compile(&nan_score, &strategy, None, &restrictions, None).unwrap_err();
        assert!(matches!(
            err,
            CompileError::InvalidUniverse(UniverseError::InvalidScore { .. })
        ));

        // Covariance size does not match asset count
        let bad_cov = Universe {
            assets: vec![Asset::new("A"), Asset::new("B")],
            covariance: CovarianceModel::Full(CscMatrix::from(&[[0.04, 0.006, 0.002], [0.006, 0.09, 0.009], [0.002, 0.009, 0.01]])),
            scenarios: None,
        };
        let err = compile(&bad_cov, &strategy, None, &restrictions, None).unwrap_err();
        assert!(matches!(
            err,
            CompileError::InvalidUniverse(UniverseError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_compile_accepts_hand_constructed_valid_universe() {
        let universe = Universe {
            assets: vec![
                Asset::new("A").score("expected_return", 0.10),
                Asset::new("B").score("expected_return", 0.05),
            ],
            covariance: CovarianceModel::Full(CscMatrix::from(&[[0.04, 0.0], [0.0, 0.01]])),
            scenarios: None,
        };
        let strategy = Strategy::builder("S").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();
        assert_eq!(problem.n_assets, 2);
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
    fn test_risk_term_is_weight_times_variance() {
        // The solver minimises ½xᵀPx, so P must be 2·weight·Σ for the risk
        // term to be weight·wᵀΣw (same scale as the linear terms).
        let universe = test_universe();
        let strategy = Strategy::builder("Balanced")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // Upper triangle of 2·0.5·Σ = Σ, column by column
        assert_eq!(problem.p.nzval, vec![0.04, 0.006, 0.09, 0.002, 0.009, 0.01]);
        assert_eq!(problem.q, vec![-0.05, -0.025, -0.015]);
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
    fn test_group_matching_no_asset() {
        let universe = test_universe();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        // Bounds exclude 0: can never hold, so it is a compile error
        let s = Strategy::builder("X")
            .minimize_risk(1.0)
            .group("currency", "JPY", 0.10, 0.20)
            .build();
        assert!(matches!(
            compile(&universe, &s, None, &restrictions, None),
            Err(CompileError::EmptyGroup { lower, .. }) if lower == 0.10
        ));

        // Bounds allow 0: trivially satisfied, no rows
        let s = Strategy::builder("X")
            .minimize_risk(1.0)
            .group("currency", "JPY", 0.0, 0.20)
            .build();
        let problem = compile(&universe, &s, None, &restrictions, None).unwrap();
        assert_eq!(problem.b.len(), 1 + 3);
    }

    #[test]
    fn test_exclusion_rows_in_asset_order() {
        let universe = test_universe();
        let strategy = Strategy::builder("MinVar").minimize_risk(1.0).build();
        let restrictions = Restrictions::builder()
            .exclude_asset("C")
            .exclude_tag("currency", "EUR") // B and C; C must not be duplicated
            .exclude_asset("A")
            .build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // rows: 0 fully invested (strategy default), then exclusions A, B, C
        assert_eq!(problem.b.len(), 4);
        for (asset, row) in [(0, 1), (1, 2), (2, 3)] {
            assert!(a_col(&problem, asset).contains(&(row, 1.0)), "asset {asset}");
        }
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
    fn test_compile_tracking_error_soc() {
        let universe = test_universe();
        let strategy = Strategy::builder("TE")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .max_tracking_error(vec![0.4, 0.3, 0.3], 0.05)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // 1 eq + 3 ineq (long only) + SOC(n+1 = 4)
        assert_eq!(problem.b.len(), 1 + 3 + 4);
        assert_eq!(problem.n_aux, 0); // TE adds rows, not variables
        assert!(matches!(problem.cones[0], SupportedConeT::ZeroConeT(1)));
        assert!(matches!(problem.cones[1], SupportedConeT::NonnegativeConeT(3)));
        assert!(matches!(problem.cones[2], SupportedConeT::SecondOrderConeT(4)));
        // SOC s0 row: b = max_te at the first SOC row
        assert_eq!(problem.b[4], 0.05);
    }

    #[test]
    fn test_compile_tracking_error_factor_soc() {
        let universe = factor_universe(); // 3 assets, 2 factors
        let strategy = Strategy::builder("TE")
            .minimize_risk(1.0)
            .max_tracking_error(vec![0.4, 0.3, 0.3], 0.05)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();

        let problem = compile(&universe, &strategy, None, &restrictions, None).unwrap();

        // SOC dim = 1 + k + n = 6
        assert!(problem
            .cones
            .iter()
            .any(|c| matches!(c, SupportedConeT::SecondOrderConeT(6))));
    }

    #[test]
    fn test_compile_cvar_aux_layout() {
        let universe = Universe::builder()
            .add_asset(Asset::new("A").score("expected_return", 0.10))
            .add_asset(Asset::new("B").score("expected_return", 0.05))
            .add_asset(Asset::new("C").score("expected_return", 0.03))
            .covariance_full(CscMatrix::from(&[
                [0.04, 0.006, 0.002],
                [0.006, 0.09, 0.009],
                [0.002, 0.009, 0.01],
            ]))
            .scenarios(vec![
                vec![0.01, 0.02, -0.01],
                vec![-0.03, 0.01, 0.00],
                vec![0.02, -0.02, 0.01],
                vec![0.00, 0.01, 0.02],
            ])
            .build()
            .unwrap();
        let strategy = Strategy::builder("CVaR")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .max_cvar(0.75, 0.02)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 0.20);

        let problem =
            compile(&universe, &strategy, None, &restrictions, Some(&turnover)).unwrap();

        // aux = 3 turnover t + (1 + 4) cvar; S=4 scenarios
        assert_eq!(problem.n_aux, 3 + 5);
        assert_eq!(problem.n_vars(), 11);
        // 1 eq + 3 long-only + 10 turnover + (2*4+1) cvar = 23 rows
        assert_eq!(problem.b.len(), 1 + 3 + 10 + 9);
        // No SOC cones for CVaR
        assert!(!problem
            .cones
            .iter()
            .any(|c| matches!(c, SupportedConeT::SecondOrderConeT(_))));
    }

    #[test]
    fn test_compile_risk_constraint_errors() {
        let universe = test_universe();
        let restrictions = Restrictions::default();

        // CVaR without scenarios
        let s = Strategy::builder("X").minimize_risk(1.0).max_cvar(0.95, 0.02).build();
        assert!(matches!(
            compile(&universe, &s, None, &restrictions, None),
            Err(CompileError::CvarRequiresScenarios)
        ));

        // Bad benchmark length
        let s = Strategy::builder("X")
            .minimize_risk(1.0)
            .max_tracking_error(vec![1.0], 0.05)
            .build();
        assert!(matches!(
            compile(&universe, &s, None, &restrictions, None),
            Err(CompileError::BenchmarkDimensionMismatch { expected: 3, got: 1 })
        ));

        // Bad TE limit
        let s = Strategy::builder("X")
            .minimize_risk(1.0)
            .max_tracking_error(vec![0.4, 0.3, 0.3], 0.0)
            .build();
        assert!(matches!(
            compile(&universe, &s, None, &restrictions, None),
            Err(CompileError::InvalidTrackingErrorLimit(_))
        ));

        // Bad alpha
        let s = Strategy::builder("X").minimize_risk(1.0).max_cvar(1.5, 0.02).build();
        assert!(matches!(
            compile(&universe, &s, None, &restrictions, None),
            Err(CompileError::InvalidCvarAlpha(_))
        ));
    }

    #[test]
    fn test_strategy_serde_compat() {
        // Old JSON without the new fields must still deserialize
        let old_json = r#"{
            "name": "Old",
            "dimensions": [],
            "group_constraints": [],
            "score_constraints": [],
            "fully_invested": true
        }"#;
        let s: Strategy = serde_json::from_str(old_json).unwrap();
        assert!(s.tracking_error.is_none() && s.cvar.is_none());

        // Current wire shape: flattened internally-tagged dimension type,
        // lowercase sense (the format used by strategy files)
        let new_json = r#"{
            "name": "New shape",
            "dimensions": [
                {"name": "financial_risk", "type": "quadratic", "sense": "minimize", "weight": 0.5},
                {"name": "ret", "type": "linear", "score_key": "expected_return", "sense": "maximize", "weight": 0.5}
            ]
        }"#;
        let s: Strategy = serde_json::from_str(new_json).unwrap();
        assert_eq!(s.dimensions.len(), 2);
        assert!(matches!(s.dimensions[0].dim_type, DimensionType::Quadratic));
        assert!(s.fully_invested); // serde default matches the builder

        // Round trip with the fields set
        let s = Strategy::builder("New")
            .minimize_risk(1.0)
            .max_tracking_error(vec![0.5, 0.5, 0.0], 0.03)
            .max_cvar(0.95, 0.02)
            .build();
        let json = serde_json::to_string(&s).unwrap();
        let back: Strategy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tracking_error.unwrap().max_te, 0.03);
        assert_eq!(back.cvar.unwrap().alpha, 0.95);
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
            tracking_error: None,
            cvar: None,
            normalization: Default::default(),
        };
        let restrictions = Restrictions::default();

        let result = compile(&universe, &strategy, None, &restrictions, None);
        assert!(result.is_err());
    }

    // ---- transaction costs -------------------------------------------------

    fn costs() -> TransactionCosts {
        TransactionCosts::new(vec![0.01, 0.02, 0.03], "expected_return")
    }

    /// (row, value) entries of column `col` of A.
    fn a_col(p: &CompiledProblem, col: usize) -> Vec<(usize, f64)> {
        (p.a.colptr[col]..p.a.colptr[col + 1])
            .map(|k| (p.a.rowval[k], p.a.nzval[k]))
            .collect()
    }

    #[test]
    fn test_costs_net_a_maximised_score_in_its_weight() {
        let universe = test_universe();
        let strategy = Strategy::builder("Balanced")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 2.0);

        let gross = compile(&universe, &strategy, None, &restrictions, Some(&turnover)).unwrap();
        let net = compile_with_costs(
            &universe, &strategy, None, &restrictions, Some(&turnover), Some(&costs()),
        )
        .unwrap();

        assert_eq!(net.q[..3], gross.q[..3]);
        assert_eq!(gross.q[3..], [0.0; 3]);
        // −0.5·(μᵀw − cᵀt): the cost carries the return dimension's weight
        for (i, c) in [0.01, 0.02, 0.03].iter().enumerate() {
            assert!((net.q[3 + i] - 0.5 * c).abs() < 1e-15, "q[t{i}] = {}", net.q[3 + i]);
        }
        assert_eq!(net.b, gross.b);
    }

    #[test]
    fn test_costs_leave_risk_and_minimised_scores_gross() {
        let universe = test_universe();
        let strategy = Strategy::builder("LowEsg")
            .minimize_risk(0.5)
            .minimize("expected_return", 0.5)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 2.0);

        let net = compile_with_costs(
            &universe, &strategy, None, &restrictions, Some(&turnover), Some(&costs()),
        )
        .unwrap();
        assert_eq!(net.q[3..], [0.0; 3]);
    }

    #[test]
    fn test_costs_net_a_score_minimum_only() {
        let universe = test_universe();
        let strategy = Strategy::builder("MinVarAtReturn")
            .minimize_risk(1.0)
            .score_min("expected_return", 0.06)
            .score_max("expected_return", 0.09)
            .score_min("esg", 4.0)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 2.0);

        let gross = compile(&universe, &strategy, None, &restrictions, Some(&turnover)).unwrap();
        let net = compile_with_costs(
            &universe, &strategy, None, &restrictions, Some(&turnover), Some(&costs()),
        )
        .unwrap();

        assert_eq!(net.b, gross.b);
        let mut extra_rows = Vec::new();
        for (i, c) in [0.01, 0.02, 0.03].iter().enumerate() {
            let (g, n) = (a_col(&gross, 3 + i), a_col(&net, 3 + i));
            let extra: Vec<_> = n.iter().filter(|e| !g.contains(e)).collect();
            assert_eq!(extra.len(), 1, "one new entry in column t{i}");
            assert_eq!(extra[0].1, *c);
            extra_rows.push(extra[0].0);
        }
        // the same row for every t: the return minimum (−μᵀw ≤ −0.06), not its max or esg
        assert!(extra_rows.iter().all(|&r| r == extra_rows[0]));
        assert_eq!(net.b[extra_rows[0]], -0.06);
    }

    #[test]
    fn test_costs_keep_the_aux_layout() {
        let universe = factor_universe();
        let strategy = Strategy::builder("Balanced")
            .minimize_risk(0.5)
            .maximize("expected_return", 0.5)
            .build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 2.0);

        let gross = compile(&universe, &strategy, None, &restrictions, Some(&turnover)).unwrap();
        let net = compile_with_costs(
            &universe, &strategy, None, &restrictions, Some(&turnover), Some(&costs()),
        )
        .unwrap();

        // [w (3), t (3), y (2)]: costs only on t, the factor block untouched
        assert_eq!(net.n_aux, gross.n_aux);
        assert_eq!(net.n_aux, 3 + 2);
        assert!(net.q[3..6].iter().all(|&q| q > 0.0));
        assert_eq!(net.q[6..], gross.q[6..]);
        assert_eq!(net.b, gross.b);
    }

    #[test]
    fn test_costs_errors() {
        let universe = test_universe();
        let strategy = Strategy::builder("Max").maximize("expected_return", 1.0).build();
        let restrictions = Restrictions::builder().long_only().fully_invested().build();
        let turnover = TurnoverConstraint::new(vec![0.4, 0.3, 0.3], 2.0);
        let with = |tc: Option<&TurnoverConstraint>, rates: Vec<f64>| {
            let costs = TransactionCosts::new(rates, "expected_return");
            compile_with_costs(&universe, &strategy, None, &restrictions, tc, Some(&costs))
        };

        assert!(matches!(
            with(None, vec![0.01; 3]),
            Err(CompileError::TransactionCostsWithoutTurnover)
        ));
        assert!(matches!(
            with(Some(&turnover), vec![0.01; 2]),
            Err(CompileError::TransactionCostDimensionMismatch { expected: 3, got: 2 })
        ));
        for (bad, at) in [(f64::NAN, 1), (f64::INFINITY, 2), (-0.01, 0)] {
            let mut rates = vec![0.01; 3];
            rates[at] = bad;
            match with(Some(&turnover), rates) {
                Err(CompileError::InvalidTransactionCost { index, .. }) => assert_eq!(index, at),
                other => panic!("{bad}: expected InvalidTransactionCost, got {other:?}"),
            }
        }
        assert!(with(Some(&turnover), vec![0.0; 3]).is_ok());
    }

    // ---- reference validation ----------------------------------------------

    /// A: USD, return + esg. B: EUR, return + esg. C: EUR, return only.
    fn sparse_universe() -> Universe {
        Universe::builder()
            .add_asset(
                Asset::new("A")
                    .tag("currency", "USD")
                    .score("expected_return", 0.10)
                    .score("esg", 6.0),
            )
            .add_asset(
                Asset::new("B")
                    .tag("currency", "EUR")
                    .score("expected_return", 0.05)
                    .score("esg", 8.0),
            )
            .add_asset(Asset::new("C").tag("currency", "EUR").score("expected_return", 0.03))
            .covariance_full(CscMatrix::from(&[
                [0.04, 0.0, 0.0],
                [0.0, 0.02, 0.0],
                [0.0, 0.0, 0.01],
            ]))
            .build()
            .unwrap()
    }

    fn compile_sparse(
        strategy: &Strategy,
        tactic: Option<&Tactic>,
        restrictions: &Restrictions,
    ) -> Result<CompiledProblem, CompileError> {
        compile(&sparse_universe(), strategy, tactic, restrictions, None)
    }

    fn risk_and(key: &str) -> Strategy {
        Strategy::builder("S").minimize_risk(0.5).minimize(key, 0.5).build()
    }

    #[test]
    fn test_unknown_score_key_in_a_dimension() {
        let err = compile_sparse(&risk_and("esgg"), None, &Restrictions::default()).unwrap_err();
        match &err {
            CompileError::UnknownScoreKey { score_key, used_by, available } => {
                assert_eq!(score_key, "esgg");
                assert_eq!(used_by, "dimension 'esgg'");
                assert_eq!(available, &["esg", "expected_return"]);
            }
            other => panic!("{other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("'esgg'") && msg.contains("esg, expected_return"), "{msg}");
    }

    #[test]
    fn test_unknown_score_key_at_weight_zero_is_still_an_error() {
        // Anchor solves and frontier end points carry weight 0: same rule.
        let mut s = risk_and("esgg");
        s.dimensions[1].weight = 0.0;
        assert!(matches!(
            compile_sparse(&s, None, &Restrictions::default()),
            Err(CompileError::UnknownScoreKey { .. })
        ));
    }

    #[test]
    fn test_reserved_metric_names_get_a_hint() {
        let msg = compile_sparse(&risk_and("financial_risk"), None, &Restrictions::default())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("minimize_risk"), "{msg}");
        let s = Strategy::builder("S").minimize_risk(1.0).score_max("cvar", 0.1).build();
        let msg = compile_sparse(&s, None, &Restrictions::default()).unwrap_err().to_string();
        assert!(msg.contains("cvar constraint"), "{msg}");
    }

    #[test]
    fn test_score_missing_on_a_holdable_asset() {
        // C has no esg: read as 0 it would be the best asset for "minimise esg".
        let err = compile_sparse(&risk_and("esg"), None, &Restrictions::default()).unwrap_err();
        match &err {
            CompileError::MissingScores { score_key, assets, .. } => {
                assert_eq!(score_key, "esg");
                assert_eq!(assets, &["C"]);
            }
            other => panic!("{other:?}"),
        }
        assert!(err.to_string().contains("exclude the asset"));

        // Excluded assets need no score, by id or by tag
        let by_id = Restrictions::builder().exclude_asset("C").build();
        assert!(compile_sparse(&risk_and("esg"), None, &by_id).is_ok());
        let by_tag = Restrictions::builder().exclude_tag("currency", "EUR").build();
        let err = compile_sparse(&risk_and("esg"), None, &by_tag);
        assert!(err.is_ok(), "B and C excluded, A has the score");
    }

    #[test]
    fn test_score_constraints_need_the_score_too() {
        let s = Strategy::builder("S").minimize_risk(1.0).score_max("esg", 7.0).build();
        assert!(matches!(
            compile_sparse(&s, None, &Restrictions::default()),
            Err(CompileError::MissingScores { ref used_by, .. }) if used_by == "a score constraint"
        ));
        let s = Strategy::builder("S").minimize_risk(1.0).score_min("esgg", 7.0).build();
        assert!(matches!(
            compile_sparse(&s, None, &Restrictions::default()),
            Err(CompileError::UnknownScoreKey { .. })
        ));
        // ... and so does a bound that only the tactic adds
        let s = Strategy::builder("S").minimize_risk(1.0).build();
        let t = Tactic::builder("T")
            .override_score(crate::constraints::ScoreConstraint::min("esg", 7.0))
            .build();
        assert!(matches!(
            compile_sparse(&s, Some(&t), &Restrictions::default()),
            Err(CompileError::MissingScores { .. })
        ));
    }

    #[test]
    fn test_transaction_cost_key_must_exist_but_need_not_be_used() {
        let u = sparse_universe();
        let s = Strategy::builder("S").minimize_risk(1.0).build();
        let r = Restrictions::default();
        let turnover = TurnoverConstraint::new(vec![1.0 / 3.0; 3], 0.5);
        let compile_costs = |key: &str| {
            let tc = TransactionCosts::new(vec![0.01; 3], key);
            compile_with_costs(&u, &s, None, &r, Some(&turnover), Some(&tc))
        };
        match compile_costs("expected_retrun") {
            Err(CompileError::UnknownScoreKey { used_by, .. }) => {
                assert_eq!(used_by, "the transaction costs")
            }
            other => panic!("{:?}", other.map(|_| ())),
        }
        // Existing key nothing consumes: inert, as documented
        assert!(compile_costs("expected_return").is_ok());
        // The costs key is a name only: a partial score is fine here
        assert!(compile_costs("esg").is_ok());
    }

    #[test]
    fn test_unknown_tag_key() {
        let s = Strategy::builder("S").minimize_risk(1.0).group("curency", "EUR", 0.0, 0.5).build();
        match compile_sparse(&s, None, &Restrictions::default()) {
            Err(CompileError::UnknownTagKey { tag_key, available, used_by, .. }) => {
                assert_eq!(tag_key, "curency");
                assert_eq!(available, ["currency"]);
                assert_eq!(used_by, "a group constraint");
            }
            other => panic!("{:?}", other.map(|_| ())),
        }
        // a group only the tactic has
        let s = Strategy::builder("S").minimize_risk(1.0).build();
        let t = Tactic::builder("T").override_group("sector", "Tech", 0.0, 0.5).build();
        assert!(matches!(
            compile_sparse(&s, Some(&t), &Restrictions::default()),
            Err(CompileError::UnknownTagKey { .. })
        ));
        // a tag exclusion
        let r = Restrictions::builder().exclude_tag("sectr", "Tobacco").build();
        assert!(matches!(
            compile_sparse(&s, None, &r),
            Err(CompileError::UnknownTagKey { ref used_by, .. }) if used_by == "an exclusion"
        ));
    }

    #[test]
    fn test_absent_tag_value_is_legal_unless_a_near_miss() {
        let s = Strategy::builder("S").minimize_risk(1.0).group("currency", "JPY", 0.0, 0.2).build();
        assert!(compile_sparse(&s, None, &Restrictions::default()).is_ok());
        let r = Restrictions::builder().exclude_tag("currency", "GBP").build();
        assert!(compile_sparse(&s, None, &r).is_ok());

        // "usd" is "USD" misspelt: as a new tactic group it would cap nothing
        let base = Strategy::builder("S").minimize_risk(1.0).group("currency", "USD", 0.0, 0.5).build();
        let t = Tactic::builder("T").override_group("currency", "usd", 0.0, 0.1).build();
        match compile_sparse(&base, Some(&t), &Restrictions::default()) {
            Err(CompileError::TagValueNearMiss { tag_value, existing, .. }) => {
                assert_eq!((tag_value.as_str(), existing.as_str()), ("usd", "USD"));
            }
            other => panic!("{:?}", other.map(|_| ())),
        }
        let r = Restrictions::builder().exclude_tag("currency", " EUR").build();
        assert!(matches!(
            compile_sparse(&base, None, &r),
            Err(CompileError::TagValueNearMiss { .. })
        ));
    }

    #[test]
    fn test_unknown_excluded_assets() {
        let s = Strategy::builder("S").minimize_risk(1.0).build();
        let r = Restrictions::builder()
            .exclude_asset("A")
            .exclude_asset("XYZ")
            .exclude_asset("OLD")
            .exclude_asset("XYZ")
            .build();
        match compile_sparse(&s, None, &r) {
            Err(CompileError::UnknownExcludedAssets { ids }) => assert_eq!(ids, ["XYZ", "OLD"]),
            other => panic!("{:?}", other.map(|_| ())),
        }
        assert_eq!(r.unmatched_exclusions(&sparse_universe()).len(), 3);

        // Opt-out: the known id is still excluded, the unknown ones skipped
        let lenient = Restrictions::builder()
            .exclude_asset("A")
            .exclude_asset("XYZ")
            .ignore_unknown_assets()
            .build();
        let p = compile_sparse(&s, None, &lenient).unwrap();
        assert_eq!(p.b.len(), 2, "budget row + one exclusion row");

        // ... but a near-miss is an error even then
        let typo = Restrictions::builder().exclude_asset("a").ignore_unknown_assets().build();
        match compile_sparse(&s, None, &typo) {
            Err(CompileError::ExcludedAssetNearMiss { id, existing }) => {
                assert_eq!((id.as_str(), existing.as_str()), ("a", "A"));
            }
            other => panic!("{:?}", other.map(|_| ())),
        }
    }
}
