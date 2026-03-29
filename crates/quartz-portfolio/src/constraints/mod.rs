pub mod allocation;
pub mod bounds;
pub mod exclusion;
pub mod scoring;
pub mod turnover;

pub use allocation::{FullyInvested, GroupConstraint};
pub use bounds::WeightBounds;
pub use exclusion::Exclusion;
pub use scoring::{ScoreBound, ScoreConstraint};
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
}

impl ConstraintContribution {
    pub fn new() -> Self {
        Self {
            triplets: Vec::new(),
            b_entries: Vec::new(),
            n_equality: 0,
            n_inequality: 0,
        }
    }

    /// Total number of constraint rows.
    pub fn n_rows(&self) -> usize {
        self.n_equality + self.n_inequality
    }
}

impl Default for ConstraintContribution {
    fn default() -> Self {
        Self::new()
    }
}
