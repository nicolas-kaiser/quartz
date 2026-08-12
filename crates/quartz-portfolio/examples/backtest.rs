//! Parallel batch solving: backtest 1000 dates in parallel.
//!
//! Builds 1000 per-date universes (deterministic pseudo-data: sine-based vol
//! regimes and drifting expected returns), solves them serially and then via
//! `solve_batch`, and prints the measured speedup.
//!
//! Run with: cargo run --release --example backtest -p quartz-portfolio
//! (--release matters: debug-mode Clarabel is 10-20x slower)

use std::time::Instant;

use clarabel::algebra::CscMatrix;
use quartz_core::{Asset, Universe};
use quartz_portfolio::solution::SolveStatus;
use quartz_portfolio::{solve_batch, BatchProblem, PortfolioModel, Restrictions, Strategy};
use quartz_solver::SolverSettings;

const N_ASSETS: usize = 10;
const N_DATES: usize = 1000;

/// Deterministic per-date universe: base covariance scaled by a slow vol
/// regime, expected returns drifting with mixed-frequency sines.
fn universe_for_date(t: usize) -> Universe {
    let vol_regime = 1.0 + 0.5 * (t as f64 / 50.0).sin();

    let mut cov = vec![vec![0.0; N_ASSETS]; N_ASSETS];
    for i in 0..N_ASSETS {
        for j in 0..N_ASSETS {
            let base = if i == j { 0.04 + 0.01 * i as f64 / N_ASSETS as f64 } else { 0.004 };
            cov[i][j] = base * vol_regime;
        }
    }
    let mut colptr = vec![0];
    let (mut rowval, mut nzval) = (Vec::new(), Vec::new());
    for j in 0..N_ASSETS {
        for (i, row) in cov.iter().enumerate() {
            rowval.push(i);
            nzval.push(row[j]);
        }
        colptr.push(rowval.len());
    }
    let cov = CscMatrix::new(N_ASSETS, N_ASSETS, colptr, rowval, nzval);

    let mut builder = Universe::builder();
    for i in 0..N_ASSETS {
        let er = 0.05 + 0.04 * ((t * 31 + i * 17) as f64 * 0.618).sin();
        let id = format!("A{i}");
        builder = builder.add_asset(Asset::new(id.as_str()).score("expected_return", er));
    }
    builder.covariance_full(cov).build().unwrap()
}

fn main() {
    let strategy = Strategy::builder("Backtest")
        .minimize_risk(0.6)
        .maximize("expected_return", 0.4)
        .build();
    let restrictions = Restrictions::builder()
        .long_only()
        .fully_invested()
        .max_single_weight(0.30)
        .build();
    let settings = SolverSettings::default();

    println!("Building {N_DATES} per-date universes ({N_ASSETS} assets each)...");
    let universes: Vec<Universe> = (0..N_DATES).map(universe_for_date).collect();
    let problems: Vec<BatchProblem> = universes
        .iter()
        .map(|u| BatchProblem::new(u, &strategy).restrictions(restrictions.clone()))
        .collect();

    // Serial baseline
    let t0 = Instant::now();
    let serial: Vec<_> = problems
        .iter()
        .map(|p| {
            PortfolioModel::new(p.universe)
                .strategy(p.strategy)
                .restrictions(p.restrictions.clone())
                .solve()
        })
        .collect();
    let serial_time = t0.elapsed();

    // Parallel batch
    let t0 = Instant::now();
    let parallel = solve_batch(&problems, &settings);
    let parallel_time = t0.elapsed();

    let n_optimal = parallel
        .iter()
        .filter(|r| matches!(r, Ok(s) if s.status == SolveStatus::Optimal))
        .count();

    // Sanity: parallel results match the serial baseline
    for (s, p) in serial.iter().zip(&parallel) {
        let (s, p) = (s.as_ref().unwrap(), p.as_ref().unwrap());
        assert_eq!(s.iterations, p.iterations);
    }

    println!("\n{N_DATES} solves, {n_optimal} optimal");
    println!("  serial:   {:>8.1} ms", serial_time.as_secs_f64() * 1000.0);
    println!("  parallel: {:>8.1} ms", parallel_time.as_secs_f64() * 1000.0);
    println!(
        "  speedup:  {:>8.1}x",
        serial_time.as_secs_f64() / parallel_time.as_secs_f64()
    );
}
