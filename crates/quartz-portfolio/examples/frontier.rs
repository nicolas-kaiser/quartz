//! Pareto frontier exploration: sweep risk vs. return on a small ESG universe.
//!
//! Run with: cargo run --example frontier -p quartz-portfolio

use clarabel::algebra::CscMatrix;
use quartz_core::Asset;
use quartz_portfolio::{FrontierExplorer, Restrictions, Strategy};

fn main() {
    let universe = quartz_core::Universe::builder()
        .add_asset(
            Asset::new("AAPL")
                .score("expected_return", 0.10)
                .score("environmental_impact", 6.5),
        )
        .add_asset(
            Asset::new("NESN")
                .score("expected_return", 0.04)
                .score("environmental_impact", 8.0),
        )
        .add_asset(
            Asset::new("GREEN_BOND")
                .score("expected_return", 0.025)
                .score("environmental_impact", 9.5),
        )
        .covariance_full(CscMatrix::from(&[
            [0.04, 0.002, 0.001],
            [0.002, 0.01, 0.001],
            [0.001, 0.001, 0.003],
        ]))
        .build()
        .unwrap();

    // Note: score units matter under weighted scalarization — mixing raw ESG
    // scores (0-10) with returns (~0.05) in one objective lets the larger unit
    // dominate. Sweep risk vs return only; env score is still reported.
    let strategy = Strategy::builder("Risk/Return")
        .minimize_risk(0.5)
        .maximize("expected_return", 0.5)
        .build();

    let restrictions = Restrictions::builder().long_only().fully_invested().build();

    let result = FrontierExplorer::new(&universe, &strategy)
        .restrictions(restrictions)
        .sweep("expected_return", "financial_risk", 11)
        .unwrap();

    println!("Pareto frontier: {} points ({} skipped)", result.points.len(), result.n_skipped);
    println!("Objective dimensions: {:?}\n", result.objective_dims);
    println!("{:>6} {:>10} {:>10} {:>8} {:>10}", "α_ret", "return", "vol", "env", "efficient");
    for p in &result.points {
        let alpha = p
            .dimension_weights
            .iter()
            .find(|(name, _)| name == "expected_return")
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        println!(
            "{:>6.2} {:>9.2}% {:>9.2}% {:>8.2} {:>10}",
            alpha, // pair mass s = 1.0, so the weight IS α
            p.portfolio_scores["expected_return"] * 100.0,
            p.portfolio_scores["financial_risk"].sqrt() * 100.0,
            p.portfolio_scores["environmental_impact"],
            p.is_efficient,
        );
    }
}
