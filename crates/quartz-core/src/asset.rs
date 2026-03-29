use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for an asset (ISIN, ticker, or internal id).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub String);

impl AssetId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for AssetId {
    fn from(s: &str) -> Self {
        AssetId(s.to_string())
    }
}

/// An investable asset with categorical tags and numerical scores.
///
/// Tags are used for group constraints (e.g. currency, sector, asset_class).
/// Scores are used for optimization dimensions (e.g. expected_return, physical_risk).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub id: AssetId,
    pub tags: HashMap<String, String>,
    pub scores: HashMap<String, f64>,
}

impl Asset {
    /// Create a new asset with the given identifier.
    pub fn new(id: impl Into<AssetId>) -> Self {
        Self {
            id: id.into(),
            tags: HashMap::new(),
            scores: HashMap::new(),
        }
    }

    /// Add a categorical tag (e.g. "currency" = "USD").
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Add a numerical score (e.g. "physical_risk" = 2.1).
    pub fn score(mut self, key: impl Into<String>, value: f64) -> Self {
        self.scores.insert(key.into(), value);
        self
    }

    /// Get a tag value by key.
    pub fn get_tag(&self, key: &str) -> Option<&str> {
        self.tags.get(key).map(|s| s.as_str())
    }

    /// Get a score value by key.
    pub fn get_score(&self, key: &str) -> Option<f64> {
        self.scores.get(key).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_builder() {
        let asset = Asset::new("AAPL")
            .tag("currency", "USD")
            .tag("sector", "Technology")
            .score("expected_return", 0.08)
            .score("physical_risk", 2.1);

        assert_eq!(asset.id, AssetId::from("AAPL"));
        assert_eq!(asset.get_tag("currency"), Some("USD"));
        assert_eq!(asset.get_tag("sector"), Some("Technology"));
        assert_eq!(asset.get_score("expected_return"), Some(0.08));
        assert_eq!(asset.get_score("physical_risk"), Some(2.1));
        assert_eq!(asset.get_tag("missing"), None);
        assert_eq!(asset.get_score("missing"), None);
    }
}
