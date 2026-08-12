//! Load a strategy from a YAML file and solve with it.
//!
//! Run with: cargo run --example strategy_file -p quartz-portfolio

use clarabel::algebra::CscMatrix;
use quartz_core::Asset;
use quartz_portfolio::{PortfolioModel, Restrictions, Strategy};

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/strategies/esg_balanced.yaml"
    );
    let strategy = Strategy::load(path).unwrap();
    println!("Loaded strategy '{}' from {path}", strategy.name);
    println!(
        "Dimensions: {:?}",
        strategy
            .dimensions
            .iter()
            .map(|d| format!("{} ({:.0}%)", d.name, d.weight * 100.0))
            .collect::<Vec<_>>()
    );

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

    let solution = PortfolioModel::new(&universe)
        .strategy(&strategy)
        .restrictions(Restrictions::builder().long_only().fully_invested().build())
        .solve()
        .unwrap();

    println!("\nStatus: {:?}", solution.status);
    for (id, w) in &solution.weights {
        println!("  {id}: {:.2}%", w * 100.0);
    }

    println!("\nRound-trip YAML:\n{}", strategy.to_yaml_string().unwrap());
}
