use serde::{Deserialize, Serialize};

/// Direction of optimization for a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sense {
    Minimize,
    Maximize,
}

impl Sense {
    /// Returns the sign multiplier for the objective function.
    /// Minimize → +1.0 (we minimize f), Maximize → -1.0 (we minimize -f).
    pub fn sign(&self) -> f64 {
        match self {
            Sense::Minimize => 1.0,
            Sense::Maximize => -1.0,
        }
    }
}

/// Type of optimization dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionType {
    /// A linear dimension based on a score key.
    /// The objective contribution is: λ * sense * Σᵢ(wᵢ * scoreᵢ)
    Linear { score_key: String },

    /// The quadratic risk dimension (portfolio variance).
    /// The objective contribution is: λ * wᵀΣw
    Quadratic,
}

/// A single dimension of optimization.
///
/// In a multi-dimensional portfolio optimization, each dimension represents
/// something to optimize: financial risk, expected return, ESG scores, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimension {
    /// Human-readable name for this dimension.
    pub name: String,
    /// Type: linear (score-based) or quadratic (covariance-based).
    pub dim_type: DimensionType,
    /// Whether to minimize or maximize this dimension.
    pub sense: Sense,
    /// Weight (λ) in the scalarized multi-objective function.
    pub weight: f64,
}

impl Dimension {
    pub fn linear(
        name: impl Into<String>,
        score_key: impl Into<String>,
        sense: Sense,
        weight: f64,
    ) -> Self {
        Self {
            name: name.into(),
            dim_type: DimensionType::Linear {
                score_key: score_key.into(),
            },
            sense,
            weight,
        }
    }

    pub fn quadratic(name: impl Into<String>, sense: Sense, weight: f64) -> Self {
        Self {
            name: name.into(),
            dim_type: DimensionType::Quadratic,
            sense,
            weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sense_sign() {
        assert_eq!(Sense::Minimize.sign(), 1.0);
        assert_eq!(Sense::Maximize.sign(), -1.0);
    }

    #[test]
    fn test_dimension_constructors() {
        let d = Dimension::linear("Expected Return", "expected_return", Sense::Maximize, 0.3);
        assert_eq!(d.name, "Expected Return");
        assert_eq!(d.sense, Sense::Maximize);
        assert_eq!(d.weight, 0.3);
        assert!(matches!(d.dim_type, DimensionType::Linear { .. }));

        let d = Dimension::quadratic("Financial Risk", Sense::Minimize, 0.5);
        assert!(matches!(d.dim_type, DimensionType::Quadratic));
    }
}
