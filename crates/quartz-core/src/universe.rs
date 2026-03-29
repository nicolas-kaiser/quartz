use clarabel::algebra::CscMatrix;
use serde::{Deserialize, Serialize};

use crate::asset::{Asset, AssetId};

/// How the covariance matrix is represented.
#[derive(Debug, Clone)]
pub enum CovarianceModel {
    /// Full n×n covariance matrix Σ.
    Full(CscMatrix<f64>),

    /// Factor model: Σ = B Fᶜᵒᵛ Bᵀ + D
    /// where loadings is n×k, factor_cov is k×k, and specific_variance is diagonal (length n).
    Factor {
        loadings: CscMatrix<f64>,
        factor_cov: CscMatrix<f64>,
        specific_variance: Vec<f64>,
    },
}

/// Error type for Universe construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UniverseError {
    NoAssets,
    NoCovariance,
    DimensionMismatch {
        n_assets: usize,
        cov_size: usize,
    },
    FactorDimensionMismatch {
        n_assets: usize,
        loadings_rows: usize,
    },
    DuplicateAssetId(String),
    InvalidScore {
        asset_id: String,
        score_key: String,
    },
}

impl std::fmt::Display for UniverseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UniverseError::NoAssets => write!(f, "Universe must contain at least one asset"),
            UniverseError::NoCovariance => write!(f, "Covariance model is required"),
            UniverseError::DimensionMismatch { n_assets, cov_size } => {
                write!(
                    f,
                    "Covariance matrix size ({cov_size}) does not match number of assets ({n_assets})"
                )
            }
            UniverseError::FactorDimensionMismatch {
                n_assets,
                loadings_rows,
            } => {
                write!(
                    f,
                    "Factor loadings rows ({loadings_rows}) does not match number of assets ({n_assets})"
                )
            }
            UniverseError::DuplicateAssetId(id) => {
                write!(f, "Duplicate asset id: {id}")
            }
            UniverseError::InvalidScore {
                asset_id,
                score_key,
            } => {
                write!(
                    f,
                    "Invalid score (NaN or infinite) for asset {asset_id}, key {score_key}"
                )
            }
        }
    }
}

impl std::error::Error for UniverseError {}

/// The investment universe: all available assets and their covariance structure.
#[derive(Debug, Clone)]
pub struct Universe {
    pub assets: Vec<Asset>,
    pub covariance: CovarianceModel,
}

impl Universe {
    pub fn builder() -> UniverseBuilder {
        UniverseBuilder::new()
    }

    /// Number of assets in the universe.
    pub fn n_assets(&self) -> usize {
        self.assets.len()
    }

    /// Returns indices of assets matching a tag key-value pair.
    pub fn asset_indices(&self, tag_key: &str, tag_value: &str) -> Vec<usize> {
        self.assets
            .iter()
            .enumerate()
            .filter(|(_, a)| a.get_tag(tag_key) == Some(tag_value))
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns the index of a specific asset by id, or None.
    pub fn asset_index(&self, id: &AssetId) -> Option<usize> {
        self.assets.iter().position(|a| &a.id == id)
    }

    /// Compute a weighted portfolio score: Σᵢ wᵢ * scoreᵢ(key).
    /// Assets missing the score are treated as 0.
    pub fn portfolio_score(&self, weights: &[f64], score_key: &str) -> f64 {
        self.assets
            .iter()
            .zip(weights.iter())
            .map(|(a, &w)| w * a.get_score(score_key).unwrap_or(0.0))
            .sum()
    }

    /// Collect scores for a given key across all assets.
    /// Assets missing the score get 0.0.
    pub fn score_vector(&self, score_key: &str) -> Vec<f64> {
        self.assets
            .iter()
            .map(|a| a.get_score(score_key).unwrap_or(0.0))
            .collect()
    }

    /// Returns all distinct values for a given tag key.
    pub fn tag_values(&self, tag_key: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .assets
            .iter()
            .filter_map(|a| a.get_tag(tag_key).map(|v| v.to_string()))
            .collect();
        values.sort();
        values.dedup();
        values
    }
}

/// Builder for constructing a valid Universe.
pub struct UniverseBuilder {
    assets: Vec<Asset>,
    covariance: Option<CovarianceModel>,
}

impl UniverseBuilder {
    pub fn new() -> Self {
        Self {
            assets: Vec::new(),
            covariance: None,
        }
    }

    pub fn add_asset(mut self, asset: Asset) -> Self {
        self.assets.push(asset);
        self
    }

    pub fn assets(mut self, assets: Vec<Asset>) -> Self {
        self.assets = assets;
        self
    }

    /// Set a full n×n covariance matrix.
    pub fn covariance_full(mut self, matrix: CscMatrix<f64>) -> Self {
        self.covariance = Some(CovarianceModel::Full(matrix));
        self
    }

    /// Set a factor covariance model: Σ = B * F * Bᵀ + D.
    pub fn covariance_factor(
        mut self,
        loadings: CscMatrix<f64>,
        factor_cov: CscMatrix<f64>,
        specific_variance: Vec<f64>,
    ) -> Self {
        self.covariance = Some(CovarianceModel::Factor {
            loadings,
            factor_cov,
            specific_variance,
        });
        self
    }

    /// Validate and build the universe.
    pub fn build(self) -> Result<Universe, UniverseError> {
        if self.assets.is_empty() {
            return Err(UniverseError::NoAssets);
        }

        let covariance = self.covariance.ok_or(UniverseError::NoCovariance)?;

        let n = self.assets.len();

        // Check for duplicate asset ids
        let mut seen = std::collections::HashSet::new();
        for asset in &self.assets {
            if !seen.insert(&asset.id) {
                return Err(UniverseError::DuplicateAssetId(asset.id.0.clone()));
            }
        }

        // Validate scores are finite
        for asset in &self.assets {
            for (key, &val) in &asset.scores {
                if !val.is_finite() {
                    return Err(UniverseError::InvalidScore {
                        asset_id: asset.id.0.clone(),
                        score_key: key.clone(),
                    });
                }
            }
        }

        // Validate covariance dimensions
        match &covariance {
            CovarianceModel::Full(cov) => {
                if cov.m != n || cov.n != n {
                    return Err(UniverseError::DimensionMismatch {
                        n_assets: n,
                        cov_size: cov.m,
                    });
                }
            }
            CovarianceModel::Factor {
                loadings,
                specific_variance,
                ..
            } => {
                if loadings.m != n {
                    return Err(UniverseError::FactorDimensionMismatch {
                        n_assets: n,
                        loadings_rows: loadings.m,
                    });
                }
                if specific_variance.len() != n {
                    return Err(UniverseError::FactorDimensionMismatch {
                        n_assets: n,
                        loadings_rows: specific_variance.len(),
                    });
                }
            }
        }

        Ok(Universe {
            assets: self.assets,
            covariance,
        })
    }
}

impl Default for UniverseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cov_3x3() -> CscMatrix<f64> {
        CscMatrix::from(&[
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.009],
            [0.002, 0.009, 0.01],
        ])
    }

    #[test]
    fn test_universe_builder() {
        let universe = Universe::builder()
            .add_asset(Asset::new("A").tag("currency", "USD").score("ret", 0.08))
            .add_asset(Asset::new("B").tag("currency", "EUR").score("ret", 0.05))
            .add_asset(Asset::new("C").tag("currency", "EUR").score("ret", 0.03))
            .covariance_full(sample_cov_3x3())
            .build()
            .unwrap();

        assert_eq!(universe.n_assets(), 3);
    }

    #[test]
    fn test_asset_indices() {
        let universe = Universe::builder()
            .add_asset(Asset::new("A").tag("currency", "USD"))
            .add_asset(Asset::new("B").tag("currency", "EUR"))
            .add_asset(Asset::new("C").tag("currency", "EUR"))
            .covariance_full(sample_cov_3x3())
            .build()
            .unwrap();

        assert_eq!(universe.asset_indices("currency", "USD"), vec![0]);
        assert_eq!(universe.asset_indices("currency", "EUR"), vec![1, 2]);
        assert_eq!(universe.asset_indices("currency", "GBP"), Vec::<usize>::new());
    }

    #[test]
    fn test_portfolio_score() {
        let universe = Universe::builder()
            .add_asset(Asset::new("A").score("ret", 0.10))
            .add_asset(Asset::new("B").score("ret", 0.05))
            .add_asset(Asset::new("C").score("ret", 0.03))
            .covariance_full(sample_cov_3x3())
            .build()
            .unwrap();

        let weights = [0.5, 0.3, 0.2];
        let score = universe.portfolio_score(&weights, "ret");
        assert!((score - 0.071).abs() < 1e-10);
    }

    #[test]
    fn test_score_vector() {
        let universe = Universe::builder()
            .add_asset(Asset::new("A").score("ret", 0.10))
            .add_asset(Asset::new("B").score("ret", 0.05))
            .add_asset(Asset::new("C")) // missing score
            .covariance_full(sample_cov_3x3())
            .build()
            .unwrap();

        assert_eq!(universe.score_vector("ret"), vec![0.10, 0.05, 0.0]);
    }

    #[test]
    fn test_duplicate_asset_id() {
        let result = Universe::builder()
            .add_asset(Asset::new("A"))
            .add_asset(Asset::new("A"))
            .covariance_full(CscMatrix::from(&[[1.0, 0.0], [0.0, 1.0]]))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let result = Universe::builder()
            .add_asset(Asset::new("A"))
            .add_asset(Asset::new("B"))
            .covariance_full(sample_cov_3x3()) // 3x3 for 2 assets
            .build();

        assert!(result.is_err());
    }
}
