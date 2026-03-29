//! Multi-dimensional portfolio optimization with ESG constraints.
//!
//! Demonstrates optimization across financial risk, expected return,
//! environmental impact, and transition risk, with group constraints
//! on currency and sector allocations.

use clarabel::algebra::CscMatrix;
use quartz_core::Asset;
use quartz_portfolio::{PortfolioModel, Restrictions, Strategy, Tactic};

fn main() {
    // --- Universe: 5 assets with multi-dimensional scores ---
    let universe = quartz_core::Universe::builder()
        .add_asset(
            Asset::new("AAPL")
                .tag("currency", "USD")
                .tag("sector", "Technology")
                .tag("asset_class", "Equity")
                .score("expected_return", 0.10)
                .score("environmental_impact", 6.5)
                .score("social_impact", 7.0)
                .score("transition_risk", 3.0)
                .score("physical_risk", 1.5),
        )
        .add_asset(
            Asset::new("BNP")
                .tag("currency", "EUR")
                .tag("sector", "Financials")
                .tag("asset_class", "Equity")
                .score("expected_return", 0.06)
                .score("environmental_impact", 4.0)
                .score("social_impact", 5.5)
                .score("transition_risk", 5.0)
                .score("physical_risk", 2.0),
        )
        .add_asset(
            Asset::new("NESN")
                .tag("currency", "CHF")
                .tag("sector", "Consumer")
                .tag("asset_class", "Equity")
                .score("expected_return", 0.04)
                .score("environmental_impact", 8.0)
                .score("social_impact", 8.5)
                .score("transition_risk", 1.5)
                .score("physical_risk", 1.0),
        )
        .add_asset(
            Asset::new("BNDS_EU")
                .tag("currency", "EUR")
                .tag("sector", "Government")
                .tag("asset_class", "Bond")
                .score("expected_return", 0.02)
                .score("environmental_impact", 5.0)
                .score("social_impact", 6.0)
                .score("transition_risk", 2.0)
                .score("physical_risk", 3.0),
        )
        .add_asset(
            Asset::new("GREEN_BOND")
                .tag("currency", "EUR")
                .tag("sector", "Government")
                .tag("asset_class", "Bond")
                .score("expected_return", 0.025)
                .score("environmental_impact", 9.5)
                .score("social_impact", 9.0)
                .score("transition_risk", 0.5)
                .score("physical_risk", 0.5),
        )
        .covariance_full(CscMatrix::from(&[
            [0.0400, 0.0060, 0.0020, 0.0005, 0.0003],
            [0.0060, 0.0900, 0.0090, 0.0010, 0.0008],
            [0.0020, 0.0090, 0.0100, 0.0003, 0.0002],
            [0.0005, 0.0010, 0.0003, 0.0025, 0.0020],
            [0.0003, 0.0008, 0.0002, 0.0020, 0.0030],
        ]))
        .build()
        .expect("Failed to build universe");

    // --- Strategy: multi-dimensional ESG balanced ---
    let strategy = Strategy::builder("ESG Balanced")
        .minimize_risk(0.40)
        .maximize("expected_return", 0.25)
        .minimize("transition_risk", 0.15)
        .maximize("environmental_impact", 0.10)
        .maximize("social_impact", 0.10)
        // Currency allocation
        .group("currency", "USD", 0.05, 0.30)
        .group("currency", "EUR", 0.30, 0.70)
        .group("currency", "CHF", 0.05, 0.30)
        // Asset class allocation
        .group("asset_class", "Equity", 0.40, 0.70)
        .group("asset_class", "Bond", 0.30, 0.60)
        // Minimum ESG scores
        .score_min("environmental_impact", 6.0)
        .score_max("transition_risk", 3.0)
        .build();

    // --- Restrictions ---
    let restrictions = Restrictions::builder()
        .long_only()
        .fully_invested()
        .max_single_weight(0.40)
        .build();

    // --- Solve without tactic ---
    println!("=== Strategic Allocation ===");
    let solution = PortfolioModel::new(&universe)
        .strategy(&strategy)
        .restrictions(restrictions.clone())
        .verbose(false)
        .solve()
        .expect("Solve failed");

    print_solution(&solution);

    // --- Tactic: overweight EUR, reduce risk ---
    let tactic = Tactic::builder("Q2 2026 Defensive")
        .override_group("currency", "EUR", 0.50, 0.70)
        .override_weight("financial_risk", 0.55)
        .override_weight("expected_return", 0.15)
        .build();

    println!("\n=== Tactical Allocation (Q2 2026 Defensive) ===");
    let solution = PortfolioModel::new(&universe)
        .strategy(&strategy)
        .tactic(&tactic)
        .restrictions(restrictions)
        .verbose(false)
        .solve()
        .expect("Solve failed");

    print_solution(&solution);
}

fn print_solution(solution: &quartz_portfolio::PortfolioSolution) {
    println!("Status: {:?}", solution.status);
    println!("Solve time: {:.3}ms", solution.solve_time_s * 1000.0);
    println!("\nWeights:");
    for (id, w) in &solution.weights {
        if *w > 0.001 {
            println!("  {:12} {:.2}%", id, w * 100.0);
        }
    }
    println!("\nPortfolio scores:");
    let mut scores: Vec<_> = solution.portfolio_scores.iter().collect();
    scores.sort_by_key(|(k, _)| (*k).clone());
    for (key, val) in scores {
        println!("  {:25} {:.4}", key, val);
    }
}
