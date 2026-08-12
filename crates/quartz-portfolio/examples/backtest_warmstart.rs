//! Sequential backtest with turnover chaining and OSQP warm starts.
//!
//! Unlike the parallel `backtest` example (independent dates), this one chains
//! dates: date t's turnover constraint uses date t−1's *actual* solved weights,
//! which forces sequential solving — the workload warm starts exist for.
//!
//! Compares three ways to run the same 200-date chain:
//!   1. Clarabel (interior-point — cannot warm-start)
//!   2. OSQP cold (fresh ADMM run per date)
//!   3. OSQP warm (each date seeded with the previous solution)
//!
//! Run with:
//!   cargo run --release --example backtest_warmstart -p quartz-portfolio --features osqp

use std::time::Instant;

use clarabel::algebra::CscMatrix;
use quartz_core::{Asset, Universe};
use quartz_portfolio::solution::{PortfolioSolution, SolveStatus};
use quartz_portfolio::{Backend, PortfolioModel, Restrictions, SolverSettings, Strategy};

const N_ASSETS: usize = 50;
const N_DATES: usize = 200;
const MAX_TURNOVER: f64 = 0.10;

/// Deterministic per-date universe: slowly drifting expected returns and a
/// slowly breathing covariance — adjacent dates are near-identical, exactly
/// the regime where warm starts pay off.
fn universe_for_date(t: usize) -> Universe {
    let vol_regime = 1.0 + 0.3 * (t as f64 / 40.0).sin();

    let mut colptr = vec![0];
    let (mut rowval, mut nzval) = (Vec::new(), Vec::new());
    for j in 0..N_ASSETS {
        for i in 0..N_ASSETS {
            let base = if i == j {
                0.03 + 0.02 * (i as f64 / N_ASSETS as f64)
            } else {
                0.002
            };
            rowval.push(i);
            nzval.push(base * vol_regime);
        }
        colptr.push(rowval.len());
    }
    let cov = CscMatrix::new(N_ASSETS, N_ASSETS, colptr, rowval, nzval);

    let mut builder = Universe::builder();
    for i in 0..N_ASSETS {
        let er = 0.06 + 0.03 * ((t as f64 * 0.05) + i as f64 * 0.7).sin();
        let id = format!("A{i}");
        builder = builder.add_asset(Asset::new(id.as_str()).score("expected_return", er));
    }
    builder.covariance_full(cov).build().unwrap()
}

struct RunStats {
    total_time_s: f64,
    mean_iterations: f64,
    solutions: Vec<PortfolioSolution>,
}

/// Solve the chained backtest: date t's turnover references date t−1's weights.
fn run_chain(
    universes: &[Universe],
    strategy: &Strategy,
    backend: Backend,
    settings: &SolverSettings,
    warm: bool,
) -> RunStats {
    let restrictions =
        || Restrictions::builder().long_only().fully_invested().build();
    let equal_weight = vec![1.0 / N_ASSETS as f64; N_ASSETS];

    let mut solutions: Vec<PortfolioSolution> = Vec::with_capacity(universes.len());
    let t0 = Instant::now();
    for universe in universes {
        let previous_weights = solutions
            .last()
            .map(|s| s.weights_vec.clone())
            .unwrap_or_else(|| equal_weight.clone());

        let mut model = PortfolioModel::new(universe)
            .strategy(strategy)
            .restrictions(restrictions())
            .turnover(previous_weights, MAX_TURNOVER)
            .backend(backend)
            .solver_settings(settings.clone());
        if warm {
            if let Some(prev) = solutions.last() {
                model = model.warm_start(prev);
            }
        }
        let sol = model.solve().unwrap();
        assert!(
            matches!(sol.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal),
            "unexpected status {:?}",
            sol.status
        );
        solutions.push(sol);
    }
    let total_time_s = t0.elapsed().as_secs_f64();
    let mean_iterations =
        solutions.iter().map(|s| s.iterations as f64).sum::<f64>() / solutions.len() as f64;
    RunStats {
        total_time_s,
        mean_iterations,
        solutions,
    }
}

fn max_weight_diff(a: &[PortfolioSolution], b: &[PortfolioSolution]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(sa, sb)| {
            sa.weights_vec
                .iter()
                .zip(&sb.weights_vec)
                .map(|(x, y)| (x - y).abs())
        })
        .fold(0.0, f64::max)
}

fn main() {
    let strategy = Strategy::builder("Chained backtest")
        .minimize_risk(0.6)
        .maximize("expected_return", 0.4)
        .build();

    println!("Building {N_DATES} per-date universes ({N_ASSETS} assets each)...");
    let universes: Vec<Universe> = (0..N_DATES).map(universe_for_date).collect();

    let clarabel = run_chain(
        &universes,
        &strategy,
        Backend::Clarabel,
        &SolverSettings::default_for(Backend::Clarabel),
        false,
    );
    let osqp_settings = SolverSettings::default_for(Backend::Osqp);
    let osqp_cold = run_chain(&universes, &strategy, Backend::Osqp, &osqp_settings, false);
    let osqp_warm = run_chain(&universes, &strategy, Backend::Osqp, &osqp_settings, true);

    println!(
        "\n{N_DATES}-date sequential backtest with turnover chaining (max turnover {MAX_TURNOVER}):\n"
    );
    println!(
        "{:<12} {:>12} {:>12} {:>22}",
        "backend", "total ms", "mean iters", "max |Δw| vs Clarabel"
    );
    for (name, stats) in [
        ("Clarabel", &clarabel),
        ("OSQP cold", &osqp_cold),
        ("OSQP warm", &osqp_warm),
    ] {
        println!(
            "{:<12} {:>12.1} {:>12.1} {:>22.2e}",
            name,
            stats.total_time_s * 1000.0,
            stats.mean_iterations,
            max_weight_diff(&stats.solutions, &clarabel.solutions),
        );
    }
    println!(
        "\nwarm start: {:.1}x fewer ADMM iterations than cold OSQP",
        osqp_cold.mean_iterations / osqp_warm.mean_iterations
    );
}
