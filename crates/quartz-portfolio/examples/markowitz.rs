//! Minimum variance portfolio with 3 assets.
//!
//! Demonstrates the basic Quartz workflow: define a universe, a strategy,
//! restrictions, and solve.

use clarabel::algebra::CscMatrix;
use quartz_core::Asset;
use quartz_portfolio::{PortfolioModel, Restrictions, Strategy};

fn main() {
    // --- Universe: 3 assets with returns and a covariance matrix ---
    let universe = quartz_core::Universe::builder()
        .add_asset(
            Asset::new("AAPL")
                .tag("currency", "USD")
                .tag("sector", "Technology")
                .score("expected_return", 0.10),
        )
        .add_asset(
            Asset::new("BNP")
                .tag("currency", "EUR")
                .tag("sector", "Financials")
                .score("expected_return", 0.06),
        )
        .add_asset(
            Asset::new("NESN")
                .tag("currency", "CHF")
                .tag("sector", "Consumer")
                .score("expected_return", 0.04),
        )
        // Covariance matrix (annual, symmetric)
        .covariance_full(CscMatrix::from(&[
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.009],
            [0.002, 0.009, 0.01],
        ]))
        .build()
        .expect("Failed to build universe");

    // --- Strategy: minimize risk only ---
    let strategy = Strategy::builder("Minimum Variance")
        .minimize_risk(1.0)
        .build();

    // --- Restrictions: long-only, fully invested ---
    let restrictions = Restrictions::builder().long_only().fully_invested().build();

    // --- Solve ---
    let solution = PortfolioModel::new(&universe)
        .strategy(&strategy)
        .restrictions(restrictions)
        .verbose(true)
        .solve()
        .expect("Solve failed");

    // --- Print results ---
    println!("\n=== Minimum Variance Portfolio ===");
    println!("Status: {:?}", solution.status);
    println!("Objective: {:.6}", solution.objective_value);
    println!("Solve time: {:.3}ms", solution.solve_time_s * 1000.0);
    println!("\nWeights:");
    for (id, w) in &solution.weights {
        println!("  {}: {:.4}%", id, w * 100.0);
    }
    println!("\nPortfolio metrics:");
    if let Some(var) = solution.variance() {
        println!("  Variance:  {:.6}", var);
        println!("  Volatility: {:.4}%", var.sqrt() * 100.0);
    }
    if let Some(ret) = solution.expected_return() {
        println!("  Expected return: {:.4}%", ret * 100.0);
    }
}
