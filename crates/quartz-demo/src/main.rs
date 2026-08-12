//! JSON bridge for the Streamlit demo: reads a problem spec on stdin,
//! solves it with Quartz, and writes the solution as JSON on stdout.
//!
//! On any error, writes `{"error": "..."}` to stdout and exits with code 1,
//! so the calling process always gets parseable JSON.

use std::collections::HashMap;
use std::io::Read;

use clarabel::algebra::CscMatrix;
use serde::{Deserialize, Serialize};

use quartz_core::{Asset, Dimension, Sense, Universe};
use quartz_portfolio::{FrontierExplorer, PortfolioModel, Restrictions, Strategy};

#[derive(Deserialize)]
struct ProblemSpec {
    assets: Vec<AssetSpec>,
    /// Dense n×n covariance matrix, row-major. Exactly one of `covariance`
    /// and `factor_model` must be provided.
    #[serde(default)]
    covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    factor_model: Option<FactorModelSpec>,
    strategy: StrategySpec,
    restrictions: RestrictionsSpec,
    #[serde(default)]
    turnover: Option<TurnoverSpec>,
    /// When present, explore a Pareto frontier instead of a single solve.
    #[serde(default)]
    frontier: Option<FrontierSpec>,
}

#[derive(Deserialize)]
struct FrontierSpec {
    /// "sweep" (two dimensions) or "grid" (simplex lattice over all dimensions).
    mode: String,
    /// Dimension names: the score_key, or "financial_risk" for the risk dim.
    #[serde(default)]
    dim_a: Option<String>,
    #[serde(default)]
    dim_b: Option<String>,
    /// Sweep steps (default 25).
    #[serde(default)]
    n_points: Option<usize>,
    /// Grid lattice resolution (default 5).
    #[serde(default)]
    resolution: Option<usize>,
}

/// Factor covariance model Σ = BFBᵀ + D.
#[derive(Deserialize)]
struct FactorModelSpec {
    /// Dense n×k loadings B, row-major.
    loadings: Vec<Vec<f64>>,
    /// Dense k×k factor covariance F, row-major, full-symmetric.
    factor_cov: Vec<Vec<f64>>,
    /// Length-n specific variance diagonal d (all ≥ 0).
    specific_variance: Vec<f64>,
}

#[derive(Deserialize)]
struct AssetSpec {
    id: String,
    #[serde(default)]
    tags: HashMap<String, String>,
    #[serde(default)]
    scores: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct StrategySpec {
    name: String,
    dimensions: Vec<DimensionSpec>,
    #[serde(default)]
    groups: Vec<GroupSpec>,
    #[serde(default)]
    score_bounds: Vec<ScoreBoundSpec>,
}

#[derive(Deserialize)]
struct DimensionSpec {
    /// "risk" for the quadratic variance dimension, "linear" for score-based.
    kind: String,
    #[serde(default)]
    score_key: Option<String>,
    /// "minimize" or "maximize".
    sense: String,
    weight: f64,
}

#[derive(Deserialize)]
struct GroupSpec {
    tag_key: String,
    tag_value: String,
    lower: f64,
    upper: f64,
}

#[derive(Deserialize)]
struct ScoreBoundSpec {
    score_key: String,
    /// "min" or "max".
    bound: String,
    threshold: f64,
}

#[derive(Deserialize)]
struct RestrictionsSpec {
    #[serde(default)]
    long_only: bool,
    #[serde(default)]
    fully_invested: bool,
    #[serde(default)]
    max_single_weight: Option<f64>,
    #[serde(default)]
    exclude_assets: Vec<String>,
    #[serde(default)]
    exclude_tags: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct TurnoverSpec {
    previous_weights: Vec<f64>,
    max_turnover: f64,
}

#[derive(Serialize)]
struct SolutionOutput {
    status: String,
    weights: Vec<WeightOutput>,
    portfolio_scores: HashMap<String, f64>,
    objective_value: f64,
    solve_time_s: f64,
    iterations: u32,
}

#[derive(Serialize)]
struct WeightOutput {
    id: String,
    weight: f64,
}

fn dense_to_csc(rows: &[Vec<f64>]) -> Result<CscMatrix<f64>, String> {
    let m = rows.len();
    if m == 0 {
        return Err("covariance matrix is empty".into());
    }
    let n = rows[0].len();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "covariance row {i} has {} entries, expected {n}",
                    row.len()
                ));
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

fn parse_sense(s: &str) -> Result<Sense, String> {
    match s {
        "minimize" => Ok(Sense::Minimize),
        "maximize" => Ok(Sense::Maximize),
        other => Err(format!("unknown sense '{other}' (expected minimize|maximize)")),
    }
}

fn run(spec: ProblemSpec) -> Result<String, String> {
    // Universe
    let mut ub = Universe::builder();
    for a in &spec.assets {
        let mut asset = Asset::new(a.id.as_str());
        for (k, v) in &a.tags {
            asset = asset.tag(k, v);
        }
        for (k, &v) in &a.scores {
            asset = asset.score(k, v);
        }
        ub = ub.add_asset(asset);
    }
    let universe = match (&spec.covariance, &spec.factor_model) {
        (Some(cov), None) => ub.covariance_full(dense_to_csc(cov)?).build(),
        (None, Some(fm)) => ub
            .covariance_factor(
                dense_to_csc(&fm.loadings)?,
                dense_to_csc(&fm.factor_cov)?,
                fm.specific_variance.clone(),
            )
            .build(),
        (Some(_), Some(_)) => {
            return Err("provide either covariance or factor_model, not both".into())
        }
        (None, None) => return Err("one of covariance or factor_model is required".into()),
    }
    .map_err(|e| e.to_string())?;

    // Strategy
    let mut sb = Strategy::builder(&spec.strategy.name);
    for d in &spec.strategy.dimensions {
        let sense = parse_sense(&d.sense)?;
        let dim = match d.kind.as_str() {
            "risk" => Dimension::quadratic("financial_risk", sense, d.weight),
            "linear" => {
                let key = d
                    .score_key
                    .as_deref()
                    .ok_or("linear dimension requires score_key")?;
                Dimension::linear(key, key, sense, d.weight)
            }
            other => return Err(format!("unknown dimension kind '{other}' (expected risk|linear)")),
        };
        sb = sb.dimension(dim);
    }
    for g in &spec.strategy.groups {
        sb = sb.group(&g.tag_key, &g.tag_value, g.lower, g.upper);
    }
    for s in &spec.strategy.score_bounds {
        sb = match s.bound.as_str() {
            "min" => sb.score_min(&s.score_key, s.threshold),
            "max" => sb.score_max(&s.score_key, s.threshold),
            other => return Err(format!("unknown score bound '{other}' (expected min|max)")),
        };
    }
    // Fully-invested is controlled via restrictions in the demo spec.
    let strategy = sb.fully_invested(false).build();

    // Restrictions
    let mut rb = Restrictions::builder();
    if spec.restrictions.long_only {
        rb = rb.long_only();
    }
    if spec.restrictions.fully_invested {
        rb = rb.fully_invested();
    }
    if let Some(max_w) = spec.restrictions.max_single_weight {
        rb = rb.max_single_weight(max_w);
    }
    for id in &spec.restrictions.exclude_assets {
        rb = rb.exclude_asset(id.as_str());
    }
    for (k, v) in &spec.restrictions.exclude_tags {
        rb = rb.exclude_tag(k, v);
    }
    let restrictions = rb.build();

    // Frontier exploration mode
    if let Some(fs) = &spec.frontier {
        let mut explorer = FrontierExplorer::new(&universe, &strategy).restrictions(restrictions);
        if let Some(t) = &spec.turnover {
            explorer = explorer.turnover(t.previous_weights.clone(), t.max_turnover);
        }
        let result = match fs.mode.as_str() {
            "sweep" => {
                let a = fs.dim_a.as_deref().ok_or("frontier sweep requires dim_a")?;
                let b = fs.dim_b.as_deref().ok_or("frontier sweep requires dim_b")?;
                explorer.sweep(a, b, fs.n_points.unwrap_or(25))
            }
            "grid" => explorer.simplex_grid(fs.resolution.unwrap_or(5)),
            other => return Err(format!("unknown frontier mode '{other}' (expected sweep|grid)")),
        }
        .map_err(|e| e.to_string())?;
        return Ok(serde_json::to_string_pretty(&result).unwrap());
    }

    // Single solve
    let mut model = PortfolioModel::new(&universe)
        .strategy(&strategy)
        .restrictions(restrictions);
    if let Some(t) = &spec.turnover {
        model = model.turnover(t.previous_weights.clone(), t.max_turnover);
    }
    let solution = model.solve().map_err(|e| e.to_string())?;

    let out = SolutionOutput {
        status: format!("{:?}", solution.status),
        weights: universe
            .assets
            .iter()
            .zip(solution.weights_vec.iter())
            .map(|(a, &w)| WeightOutput {
                id: a.id.to_string(),
                weight: w,
            })
            .collect(),
        portfolio_scores: solution.portfolio_scores,
        objective_value: solution.objective_value,
        solve_time_s: solution.solve_time_s,
        iterations: solution.iterations,
    };
    Ok(serde_json::to_string_pretty(&out).unwrap())
}

fn main() {
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        println!("{}", serde_json::json!({ "error": format!("failed to read stdin: {e}") }));
        std::process::exit(1);
    }

    let spec: ProblemSpec = match serde_json::from_str(&input) {
        Ok(s) => s,
        Err(e) => {
            println!("{}", serde_json::json!({ "error": format!("invalid spec JSON: {e}") }));
            std::process::exit(1);
        }
    };

    match run(spec) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            println!("{}", serde_json::json!({ "error": e }));
            std::process::exit(1);
        }
    }
}
