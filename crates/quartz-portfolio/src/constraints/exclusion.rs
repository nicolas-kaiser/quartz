use serde::{Deserialize, Serialize};

use quartz_core::{AssetId, Universe};

/// An exclusion rule: assets matching this rule are forced to weight = 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Exclusion {
    /// Exclude all assets matching a tag key-value pair.
    ByTag { tag_key: String, tag_value: String },
    /// Exclude a specific asset by id.
    ByAsset(AssetId),
}

impl Exclusion {
    pub fn by_tag(key: impl Into<String>, value: impl Into<String>) -> Self {
        Exclusion::ByTag {
            tag_key: key.into(),
            tag_value: value.into(),
        }
    }

    pub fn by_asset(id: impl Into<AssetId>) -> Self {
        Exclusion::ByAsset(id.into())
    }

    /// Returns the indices of excluded assets in the universe.
    pub fn excluded_indices(&self, universe: &Universe) -> Vec<usize> {
        match self {
            Exclusion::ByTag { tag_key, tag_value } => {
                universe.asset_indices(tag_key, tag_value)
            }
            Exclusion::ByAsset(id) => {
                universe.asset_index(id).into_iter().collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clarabel::algebra::CscMatrix;
    use quartz_core::Asset;

    fn test_universe() -> Universe {
        Universe::builder()
            .add_asset(Asset::new("A").tag("sector", "Technology"))
            .add_asset(Asset::new("B").tag("sector", "Tobacco"))
            .add_asset(Asset::new("C").tag("sector", "Financials"))
            .covariance_full(CscMatrix::identity(3))
            .build()
            .unwrap()
    }

    #[test]
    fn test_exclude_by_tag() {
        let excl = Exclusion::by_tag("sector", "Tobacco");
        let indices = excl.excluded_indices(&test_universe());
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn test_exclude_by_asset() {
        let excl = Exclusion::by_asset("C");
        let indices = excl.excluded_indices(&test_universe());
        assert_eq!(indices, vec![2]);
    }
}
