//! Load and save strategies, tactics, and restrictions as JSON or YAML files.
//!
//! The file format is the types' serde shape, tuned for hand-authoring:
//!
//! ```yaml
//! name: ESG Balanced
//! dimensions:
//! - name: financial_risk
//!   type: quadratic
//!   sense: minimize
//!   weight: 0.4
//! - name: expected_return
//!   type: linear
//!   score_key: expected_return
//!   sense: maximize
//!   weight: 0.6
//! group_constraints:
//! - {tag_key: currency, tag_value: EUR, lower: 0.4, upper: 0.6}
//! score_constraints:
//! - score_key: environmental_impact
//!   bound: !min 5.0            # or !max 3.0 / !range [4.0, 7.0]
//! tracking_error: {benchmark_weights: [0.5, 0.5], max_te: 0.03}
//! cvar: {alpha: 0.95, max_cvar: 0.02}
//! ```
//!
//! In YAML, enum variants with values use YAML tags (`bound: !min 5.0`,
//! exclusions `- !by_asset XYZ` / `- !by_tag {tag_key: sector, tag_value: Tobacco}`);
//! in JSON the same data is an object (`"bound": {"min": 5.0}`).
//!
//! Strategy loaders normalize dimension weights to sum to 1 (matching the
//! builder), so `40/30/30` behaves like `0.4/0.3/0.3`. No further validation
//! happens at load time — `compile()` validates at solve time. Note that
//! unknown keys inside dimensions are silently ignored (a consequence of the
//! flattened `type` tag); missing required keys still error.

use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::restriction::Restrictions;
use crate::strategy::Strategy;
use crate::tactic::Tactic;

/// Errors from loading or saving configuration files.
#[derive(Debug, Clone)]
pub enum ConfigError {
    Io(String),
    Json(String),
    Yaml(String),
    /// Path has no recognized extension (.json, .yaml, .yml).
    UnknownExtension(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(msg) => write!(f, "I/O error: {msg}"),
            ConfigError::Json(msg) => write!(f, "JSON error: {msg}"),
            ConfigError::Yaml(msg) => write!(f, "YAML error: {msg}"),
            ConfigError::UnknownExtension(path) => {
                write!(f, "Unknown file extension for '{path}' (expected .json, .yaml, or .yml)")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn parse_json<T: DeserializeOwned>(s: &str) -> Result<T, ConfigError> {
    serde_json::from_str(s).map_err(|e| ConfigError::Json(e.to_string()))
}

fn parse_yaml<T: DeserializeOwned>(s: &str) -> Result<T, ConfigError> {
    serde_yaml::from_str(s).map_err(|e| ConfigError::Yaml(e.to_string()))
}

fn to_json<T: Serialize>(value: &T) -> Result<String, ConfigError> {
    serde_json::to_string_pretty(value).map_err(|e| ConfigError::Json(e.to_string()))
}

fn to_yaml<T: Serialize>(value: &T) -> Result<String, ConfigError> {
    serde_yaml::to_string(value).map_err(|e| ConfigError::Yaml(e.to_string()))
}

enum Format {
    Json,
    Yaml,
}

fn format_for(path: &Path) -> Result<Format, ConfigError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => Ok(Format::Json),
        Some("yaml") | Some("yml") => Ok(Format::Yaml),
        _ => Err(ConfigError::UnknownExtension(path.display().to_string())),
    }
}

fn read_dispatch<T: DeserializeOwned>(path: &Path) -> Result<T, ConfigError> {
    let format = format_for(path)?;
    let text = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;
    match format {
        Format::Json => parse_json(&text),
        Format::Yaml => parse_yaml(&text),
    }
}

fn write_dispatch<T: Serialize>(value: &T, path: &Path) -> Result<(), ConfigError> {
    let text = match format_for(path)? {
        Format::Json => to_json(value)?,
        Format::Yaml => to_yaml(value)?,
    };
    std::fs::write(path, text).map_err(|e| ConfigError::Io(e.to_string()))
}

impl Strategy {
    pub fn from_json_str(s: &str) -> Result<Self, ConfigError> {
        let mut strategy: Strategy = parse_json(s)?;
        strategy.normalize_dimension_weights();
        Ok(strategy)
    }

    pub fn from_yaml_str(s: &str) -> Result<Self, ConfigError> {
        let mut strategy: Strategy = parse_yaml(s)?;
        strategy.normalize_dimension_weights();
        Ok(strategy)
    }

    pub fn to_json_string(&self) -> Result<String, ConfigError> {
        to_json(self)
    }

    pub fn to_yaml_string(&self) -> Result<String, ConfigError> {
        to_yaml(self)
    }

    /// Load from a .json/.yaml/.yml file (dimension weights are normalized).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let mut strategy: Strategy = read_dispatch(path.as_ref())?;
        strategy.normalize_dimension_weights();
        Ok(strategy)
    }

    /// Save to a .json/.yaml/.yml file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        write_dispatch(self, path.as_ref())
    }
}

macro_rules! impl_config_io {
    ($ty:ty) => {
        impl $ty {
            pub fn from_json_str(s: &str) -> Result<Self, ConfigError> {
                parse_json(s)
            }
            pub fn from_yaml_str(s: &str) -> Result<Self, ConfigError> {
                parse_yaml(s)
            }
            pub fn to_json_string(&self) -> Result<String, ConfigError> {
                to_json(self)
            }
            pub fn to_yaml_string(&self) -> Result<String, ConfigError> {
                to_yaml(self)
            }
            /// Load from a .json/.yaml/.yml file.
            pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
                read_dispatch(path.as_ref())
            }
            /// Save to a .json/.yaml/.yml file.
            pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
                write_dispatch(self, path.as_ref())
            }
        }
    };
}

impl_config_io!(Tactic);
impl_config_io!(Restrictions);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{Exclusion, ScoreBound};
    use quartz_core::Sense;

    fn full_strategy() -> Strategy {
        Strategy::builder("Full")
            .minimize_risk(0.4)
            .maximize("expected_return", 0.3)
            .minimize("transition_risk", 0.3)
            .group("currency", "EUR", 0.3, 0.6)
            .score_min("environmental_impact", 5.0)
            .max_tracking_error(vec![0.5, 0.5, 0.0], 0.03)
            .max_cvar(0.95, 0.02)
            .build()
    }

    #[test]
    fn test_strategy_yaml_round_trip() {
        let s = full_strategy();
        let yaml = s.to_yaml_string().unwrap();
        let back = Strategy::from_yaml_str(&yaml).unwrap();
        assert_eq!(back.name, s.name);
        assert_eq!(back.dimensions.len(), 3);
        assert_eq!(back.group_constraints.len(), 1);
        assert_eq!(back.tracking_error.as_ref().unwrap().max_te, 0.03);
        assert_eq!(back.cvar.as_ref().unwrap().alpha, 0.95);
        for (a, b) in s.dimensions.iter().zip(&back.dimensions) {
            assert!((a.weight - b.weight).abs() < 1e-12);
        }
    }

    #[test]
    fn test_strategy_json_round_trip() {
        let s = full_strategy();
        let json = s.to_json_string().unwrap();
        let back = Strategy::from_json_str(&json).unwrap();
        assert_eq!(back.dimensions.len(), 3);
        assert!(back.cvar.is_some());
    }

    #[test]
    fn test_hand_authored_yaml_format() {
        // Locks the human-facing format: flattened type tag, lowercase sense,
        // externally-tagged lowercase bound.
        let yaml = r#"
name: ESG Balanced
dimensions:
- name: financial_risk
  type: quadratic
  sense: minimize
  weight: 40
- name: expected_return
  type: linear
  score_key: expected_return
  sense: maximize
  weight: 60
score_constraints:
- score_key: environmental_impact
  bound: !min 5.0
tracking_error:
  benchmark_weights: [0.5, 0.5]
  max_te: 0.05
"#;
        let s = Strategy::from_yaml_str(yaml).unwrap();
        assert_eq!(s.name, "ESG Balanced");
        assert!(matches!(
            s.dimensions[0].dim_type,
            quartz_core::DimensionType::Quadratic
        ));
        assert_eq!(s.dimensions[1].sense, Sense::Maximize);
        assert!(matches!(
            s.score_constraints[0].bound,
            ScoreBound::Min(t) if t == 5.0
        ));
        // Weights 40/60 normalized to 0.4/0.6
        assert!((s.dimensions[0].weight - 0.4).abs() < 1e-12);
        assert!((s.dimensions[1].weight - 0.6).abs() < 1e-12);
        // fully_invested defaults true when absent
        assert!(s.fully_invested);
        assert_eq!(s.tracking_error.unwrap().max_te, 0.05);
    }

    #[test]
    fn test_raw_serde_does_not_normalize() {
        let json = r#"{"name":"Raw","dimensions":[
            {"name":"a","type":"linear","score_key":"a","sense":"maximize","weight":40.0},
            {"name":"b","type":"linear","score_key":"b","sense":"maximize","weight":60.0}]}"#;
        let raw: Strategy = serde_json::from_str(json).unwrap();
        assert_eq!(raw.dimensions[0].weight, 40.0); // pure Deserialize: untouched
        let loaded = Strategy::from_json_str(json).unwrap();
        assert!((loaded.dimensions[0].weight - 0.4).abs() < 1e-12); // file API normalizes
    }

    #[test]
    fn test_restrictions_and_tactic_round_trip() {
        let r = Restrictions::builder()
            .long_only()
            .fully_invested()
            .max_single_weight(0.3)
            .exclude_tag("sector", "Tobacco")
            .exclude_asset("XYZ")
            .build();
        let yaml = r.to_yaml_string().unwrap();
        let back = Restrictions::from_yaml_str(&yaml).unwrap();
        assert!(back.long_only);
        assert_eq!(back.max_single_weight, Some(0.3));
        assert_eq!(back.exclusions.len(), 2);
        assert!(matches!(back.exclusions[0], Exclusion::ByTag { .. }));
        assert!(matches!(back.exclusions[1], Exclusion::ByAsset(_)));

        let t = crate::Tactic::builder("Q3")
            .override_group("currency", "EUR", 0.5, 0.7)
            .override_weight("expected_return", 0.8)
            .build();
        let yaml = t.to_yaml_string().unwrap();
        let back = Tactic::from_yaml_str(&yaml).unwrap();
        assert_eq!(back.group_overrides.len(), 1);
        assert_eq!(back.dimension_weight_overrides["expected_return"], 0.8);

        // Minimal YAML parses via the serde defaults
        let minimal: Restrictions = Restrictions::from_yaml_str("long_only: true").unwrap();
        assert!(minimal.long_only && !minimal.fully_invested);
    }

    #[test]
    fn test_load_save_dispatch() {
        let dir = std::env::temp_dir().join("quartz_config_test");
        std::fs::create_dir_all(&dir).unwrap();
        let s = full_strategy();

        for name in ["s.json", "s.yaml", "s.yml"] {
            let path = dir.join(name);
            s.save(&path).unwrap();
            let back = Strategy::load(&path).unwrap();
            assert_eq!(back.name, s.name);
            std::fs::remove_file(&path).ok();
        }

        assert!(matches!(
            Strategy::load(dir.join("s.toml")),
            Err(ConfigError::UnknownExtension(_))
        ));
        assert!(matches!(
            Strategy::load(dir.join("missing.yaml")),
            Err(ConfigError::Io(_))
        ));
    }

    #[test]
    fn test_malformed_input_errors() {
        assert!(matches!(
            Strategy::from_yaml_str("dimensions: {not: [valid"),
            Err(ConfigError::Yaml(_))
        ));
        assert!(matches!(
            Strategy::from_json_str("{ not json"),
            Err(ConfigError::Json(_))
        ));
    }
}
