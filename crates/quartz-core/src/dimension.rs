use serde::{Deserialize, Serialize};

/// Direction of optimization for a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sense {
    #[serde(alias = "Minimize")]
    Minimize,
    #[serde(alias = "Maximize")]
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
///
/// Serialized internally tagged so strategy files read naturally:
/// `type: quadratic` or `type: linear` + `score_key: ...`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
///
/// Deserialization goes through [`DimensionRepr`] so that a misspelt key
/// (`wieght`) is an error; `flatten` alone would ignore it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "DimensionRepr")]
pub struct Dimension {
    /// Human-readable name for this dimension.
    pub name: String,
    /// Type: linear (score-based) or quadratic (covariance-based).
    #[serde(flatten)]
    pub dim_type: DimensionType,
    /// Whether to minimize or maximize this dimension.
    pub sense: Sense,
    /// Weight (λ) in the scalarized multi-objective function.
    pub weight: f64,
}

/// The kind tag of a dimension on the wire.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DimensionKind {
    Linear,
    Quadratic,
}

/// Strict wire shape of a [`Dimension`]: the same keys the serializer writes,
/// and no others.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DimensionRepr {
    name: String,
    #[serde(rename = "type")]
    kind: DimensionKind,
    #[serde(default)]
    score_key: Option<String>,
    sense: Sense,
    weight: f64,
}

impl TryFrom<DimensionRepr> for Dimension {
    type Error = String;

    fn try_from(r: DimensionRepr) -> Result<Self, String> {
        let dim_type = match (r.kind, r.score_key) {
            (DimensionKind::Linear, Some(score_key)) => DimensionType::Linear { score_key },
            (DimensionKind::Linear, None) => {
                return Err(format!(
                    "dimension '{}': type linear needs a score_key",
                    r.name
                ))
            }
            (DimensionKind::Quadratic, None) => DimensionType::Quadratic,
            (DimensionKind::Quadratic, Some(key)) => {
                return Err(format!(
                    "dimension '{}': type quadratic is the portfolio variance and takes no \
                     score_key (got '{key}'); use type linear to optimise a score",
                    r.name
                ))
            }
        };
        Ok(Dimension {
            name: r.name,
            dim_type,
            sense: r.sense,
            weight: r.weight,
        })
    }
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
    fn test_serde_shape() {
        // Human-friendly wire shape: flattened internally-tagged type + lowercase sense
        let d = Dimension::quadratic("financial_risk", Sense::Minimize, 0.5);
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains(r#""type":"quadratic""#), "{json}");
        assert!(json.contains(r#""sense":"minimize""#), "{json}");

        let d = Dimension::linear("ret", "expected_return", Sense::Maximize, 0.5);
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains(r#""type":"linear""#), "{json}");
        assert!(json.contains(r#""score_key":"expected_return""#), "{json}");

        // Old capitalized sense still accepted via alias
        let d: Dimension = serde_json::from_str(
            r#"{"name":"x","type":"quadratic","sense":"Minimize","weight":1.0}"#,
        )
        .unwrap();
        assert_eq!(d.sense, Sense::Minimize);
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
