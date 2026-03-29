use serde::{Deserialize, Serialize};

use crate::constraints::exclusion::Exclusion;

/// Hard constraints from compliance / regulatory requirements.
///
/// These are applied after strategy/tactic merge and override any softer bounds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Restrictions {
    /// Assets to completely exclude (weight = 0).
    pub exclusions: Vec<Exclusion>,
    /// Maximum weight per single asset.
    pub max_single_weight: Option<f64>,
    /// Require all weights ≥ 0 (no shorting).
    pub long_only: bool,
    /// Require Σw = 1.
    pub fully_invested: bool,
}

impl Restrictions {
    pub fn builder() -> RestrictionsBuilder {
        RestrictionsBuilder::default()
    }
}

#[derive(Default)]
pub struct RestrictionsBuilder {
    exclusions: Vec<Exclusion>,
    max_single_weight: Option<f64>,
    long_only: bool,
    fully_invested: bool,
}

impl RestrictionsBuilder {
    pub fn exclude_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.exclusions.push(Exclusion::by_tag(key, value));
        self
    }

    pub fn exclude_asset(mut self, id: impl Into<quartz_core::AssetId>) -> Self {
        self.exclusions.push(Exclusion::by_asset(id));
        self
    }

    pub fn max_single_weight(mut self, max: f64) -> Self {
        self.max_single_weight = Some(max);
        self
    }

    pub fn long_only(mut self) -> Self {
        self.long_only = true;
        self
    }

    pub fn fully_invested(mut self) -> Self {
        self.fully_invested = true;
        self
    }

    pub fn build(self) -> Restrictions {
        Restrictions {
            exclusions: self.exclusions,
            max_single_weight: self.max_single_weight,
            long_only: self.long_only,
            fully_invested: self.fully_invested,
        }
    }
}
