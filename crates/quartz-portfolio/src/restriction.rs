use serde::{Deserialize, Serialize};

use crate::constraints::exclusion::Exclusion;

/// Hard constraints from compliance / regulatory requirements.
///
/// These are applied after strategy/tactic merge and override any softer bounds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Restrictions {
    /// Assets to completely exclude (weight = 0).
    pub exclusions: Vec<Exclusion>,
    /// Maximum weight per single asset.
    pub max_single_weight: Option<f64>,
    /// Require all weights ≥ 0 (no shorting).
    pub long_only: bool,
    /// Require Σw = 1.
    pub fully_invested: bool,
    /// Accept `by_asset` exclusions whose id is not in the universe (default
    /// false: they are a compile error). For one restricted list applied to
    /// changing universes. An id that differs from an existing one only by
    /// case or surrounding spaces is an error either way.
    pub ignore_unknown_assets: bool,
}

impl Restrictions {
    #[must_use]
    pub fn builder() -> RestrictionsBuilder {
        RestrictionsBuilder::default()
    }

    /// The exclusions that match no asset of `universe`: unknown asset ids,
    /// and tags no asset carries. Useful for logging what a restricted list
    /// did not hit (see `ignore_unknown_assets`).
    pub fn unmatched_exclusions(&self, universe: &quartz_core::Universe) -> Vec<Exclusion> {
        self.exclusions
            .iter()
            .filter(|e| e.excluded_indices(universe).is_empty())
            .cloned()
            .collect()
    }
}

#[derive(Default)]
pub struct RestrictionsBuilder {
    exclusions: Vec<Exclusion>,
    max_single_weight: Option<f64>,
    long_only: bool,
    fully_invested: bool,
    ignore_unknown_assets: bool,
}

impl RestrictionsBuilder {
    #[must_use]
    pub fn exclude_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.exclusions.push(Exclusion::by_tag(key, value));
        self
    }

    #[must_use]
    pub fn exclude_asset(mut self, id: impl Into<quartz_core::AssetId>) -> Self {
        self.exclusions.push(Exclusion::by_asset(id));
        self
    }

    #[must_use]
    pub fn max_single_weight(mut self, max: f64) -> Self {
        self.max_single_weight = Some(max);
        self
    }

    #[must_use]
    pub fn long_only(mut self) -> Self {
        self.long_only = true;
        self
    }

    #[must_use]
    pub fn fully_invested(mut self) -> Self {
        self.fully_invested = true;
        self
    }

    /// Accept excluded asset ids that are not in the universe instead of
    /// failing (see [`Restrictions::ignore_unknown_assets`]).
    #[must_use]
    pub fn ignore_unknown_assets(mut self) -> Self {
        self.ignore_unknown_assets = true;
        self
    }

    pub fn build(self) -> Restrictions {
        Restrictions {
            exclusions: self.exclusions,
            max_single_weight: self.max_single_weight,
            long_only: self.long_only,
            fully_invested: self.fully_invested,
            ignore_unknown_assets: self.ignore_unknown_assets,
        }
    }
}
