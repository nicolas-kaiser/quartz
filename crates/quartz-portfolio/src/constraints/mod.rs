pub mod allocation;
pub mod bounds;
pub mod cvar;
pub mod exclusion;
pub mod factor;
pub mod scoring;
pub mod tracking;
pub mod turnover;

pub use allocation::{FullyInvested, GroupConstraint};
pub use bounds::WeightBounds;
pub use cvar::CvarConstraint;
pub use exclusion::Exclusion;
pub use factor::FactorLink;
pub use scoring::{ScoreBound, ScoreConstraint};
pub use tracking::TrackingErrorConstraint;
pub use turnover::TurnoverConstraint;

/// A triplet (row, col, value) for building sparse constraint matrices.
#[derive(Debug, Clone)]
pub struct Triplet {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

/// The output of compiling a single constraint: rows in A, entries in b, and cone info.
#[derive(Debug, Clone)]
pub struct ConstraintContribution {
    /// Triplets for the A matrix (row indices are local, will be offset by the compiler).
    pub triplets: Vec<Triplet>,
    /// Right-hand side entries (one per row added).
    pub b_entries: Vec<f64>,
    /// Number of equality rows (these go into ZeroConeT).
    pub n_equality: usize,
    /// Number of inequality rows (these go into NonnegativeConeT).
    pub n_inequality: usize,
    /// Second-order cone block dimensions (each becomes a SecondOrderConeT).
    /// SOC local rows follow the equality and inequality rows; the compiler
    /// places all SOC blocks after every inequality row in the assembled A.
    pub soc_blocks: Vec<usize>,
}

impl ConstraintContribution {
    pub fn new() -> Self {
        Self {
            triplets: Vec::new(),
            b_entries: Vec::new(),
            n_equality: 0,
            n_inequality: 0,
            soc_blocks: Vec::new(),
        }
    }

    /// Total number of constraint rows.
    pub fn n_rows(&self) -> usize {
        self.n_equality + self.n_inequality + self.soc_blocks.iter().sum::<usize>()
    }
}

impl Default for ConstraintContribution {
    fn default() -> Self {
        Self::new()
    }
}
