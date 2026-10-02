//====== periodica/rust/periodica_core/src/rules.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # rules
//!
//! `data/config/composition_rules.json`, parsed once per registry load.
//!
//! Every behaviour of the composer that could be chemistry -- which tiers
//! exist, which properties add up, which of those are integral quantum
//! numbers, which tiers an unscoped spec is tried against first, where
//! generated tiers take their curated masses from -- lives in that file, not
//! in code. This module turns it into a typed value so the registry and the
//! composer stop re-reading and re-parsing it (the Python implementation read
//! it 394 times per registry build).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use periodica_mat::jsonc;
use serde_json::Value;

/// Location of the rules file relative to the `data/` directory.
pub const RULES_RELATIVE_PATH: &str = "config/composition_rules.json";

/// A composed total of an integral property is snapped to the nearest integer
/// when it lies within this distance of one. Quark charges are stored as
/// `0.6666666667`, so `{u=2,d=1}` sums to `1.0000000001`; the proton's charge
/// is nevertheless exactly `+1`.
pub const INTEGRAL_SNAP_TOLERANCE: f64 = 1e-9;

/// One tier as declared under `tier_definitions`, or discovered under
/// `derived_root`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierSource {
    pub name: String,
    /// Path relative to the `data/` directory, e.g. `active/quarks`.
    pub source: String,
    /// `is_active` from the rules. Feeds the name-resolution priority table
    /// (see [`crate::registry`]).
    pub is_active: bool,
}

/// The parsed rules file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompositionRules {
    /// Hand-curated tiers, in declaration order.
    pub tier_definitions: Vec<TierSource>,
    /// Directory (relative to `data/`) whose every subdirectory is a tier.
    pub derived_root: String,
    /// Properties a composition sums, weighted by count. Output order.
    pub additive_properties: Vec<String>,
    /// The subset of [`Self::additive_properties`] that are integral quantum
    /// numbers (charge, baryon and lepton number). Only these are snapped to
    /// an integer; masses never are -- snapping every total is what zeroed
    /// `Mass_kg` (~1e-26) in every composed entry.
    pub integral_properties: Vec<String>,
    /// Lower-cased file-stem prefixes the registry skips (`demo_`, ...).
    pub placeholder_prefixes: Vec<String>,
    /// Tier groups an unscoped brace spec is resolved against, in order: the
    /// first level whose tiers contain every constituent symbol exactly wins.
    /// See [`crate::get`] for the full rule.
    pub resolution_levels: Vec<Vec<String>>,
    /// The raw `mass_sources` block, interpreted by [`crate::mass_source`].
    pub mass_sources: Option<Value>,
}

impl CompositionRules {
    /// Read and parse `<data_root>/config/composition_rules.json`.
    ///
    /// There is no built-in fallback: a missing or malformed rules file fails
    /// the registry load rather than silently composing with defaults.
    pub fn load(data_root: &Path) -> Result<Self> {
        let path = data_root.join(RULES_RELATIVE_PATH);
        let value = jsonc::from_path(&path)
            .with_context(|| format!("composition rules: reading {}", path.display()))?;
        Self::from_value(&value).with_context(|| format!("composition rules: {}", path.display()))
    }

    /// Interpret an already-parsed rules document.
    pub fn from_value(cfg: &Value) -> Result<Self> {
        let mut rules = CompositionRules {
            derived_root: cfg
                .get("derived_root")
                .and_then(Value::as_str)
                .unwrap_or("derived")
                .to_string(),
            additive_properties: string_list(cfg, "additive_properties")?,
            integral_properties: string_list(cfg, "integral_properties")?,
            placeholder_prefixes: string_list(cfg, "placeholder_prefixes")?
                .into_iter()
                .map(|p| p.to_lowercase())
                .collect(),
            resolution_levels: levels(cfg)?,
            mass_sources: cfg.get("mass_sources").cloned(),
            ..Default::default()
        };

        if let Some(defs) = cfg.get("tier_definitions").and_then(Value::as_array) {
            for e in defs {
                let (Some(name), Some(source)) = (
                    e.get("name").and_then(Value::as_str),
                    e.get("source").and_then(Value::as_str),
                ) else {
                    return Err(anyhow!(
                        "tier_definitions entry without a name and a source: {e}"
                    ));
                };
                rules.tier_definitions.push(TierSource {
                    name: name.to_string(),
                    source: source.to_string(),
                    is_active: e.get("is_active").and_then(Value::as_bool).unwrap_or(false),
                });
            }
        }

        if let Some(bad) = rules
            .integral_properties
            .iter()
            .find(|p| !rules.additive_properties.contains(p))
        {
            return Err(anyhow!(
                "integral property {bad:?} is not listed in additive_properties"
            ));
        }
        Ok(rules)
    }

    /// Whether `property` is snapped to an integer when composed.
    pub fn is_integral(&self, property: &str) -> bool {
        self.integral_properties.iter().any(|p| p == property)
    }

    /// Whether a file stem is a placeholder the registry skips.
    pub fn is_placeholder(&self, stem: &str) -> bool {
        crate::registry::is_placeholder(stem, &self.placeholder_prefixes)
    }

    /// Every tier, in resolution order: the declared tiers, then each
    /// existing subdirectory of [`Self::derived_root`] sorted by name.
    ///
    /// Declared tiers whose directory does not exist are kept here; the loader
    /// skips them, exactly as Python's registry does.
    pub fn tier_sources(&self, data_root: &Path) -> Vec<TierSource> {
        let mut out = self.tier_definitions.clone();
        let derived_dir = data_root.join(&self.derived_root);
        if let Ok(entries) = std::fs::read_dir(&derived_dir) {
            let mut names: Vec<String> = entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().to_str().map(str::to_string))
                .collect();
            names.sort();
            for n in names {
                out.push(TierSource {
                    source: format!("{}/{n}", self.derived_root),
                    name: n,
                    is_active: false,
                });
            }
        }
        out
    }
}

fn string_list(cfg: &Value, key: &str) -> Result<Vec<String>> {
    match cfg.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("{key}: expected strings, found {v}"))
            })
            .collect(),
        Some(other) => Err(anyhow!("{key}: expected a list of strings, found {other}")),
    }
}

fn levels(cfg: &Value) -> Result<Vec<Vec<String>>> {
    let Some(raw) = cfg.get("resolution_levels") else {
        return Ok(Vec::new());
    };
    let items = raw
        .as_array()
        .ok_or_else(|| anyhow!("resolution_levels: expected a list of tier lists"))?;
    items
        .iter()
        .map(|level| match level {
            Value::String(t) => Ok(vec![t.clone()]),
            Value::Array(ts) => ts
                .iter()
                .map(|t| {
                    t.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| anyhow!("resolution_levels: tier names must be strings"))
                })
                .collect(),
            other => Err(anyhow!(
                "resolution_levels: each level is a tier name or a list of them, found {other}"
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Value {
        json!({
            "tier_definitions": [
                {"name": "fundamentals", "source": "active/quarks", "is_active": true},
                {"name": "subatomic", "source": "active/subatomic", "is_active": true}
            ],
            "derived_root": "derived",
            "additive_properties": ["Charge_e", "Mass_amu", "Mass_kg"],
            "integral_properties": ["Charge_e"],
            "placeholder_prefixes": ["Demo_", "example_"],
            "resolution_levels": ["atoms", ["subatomic", "fundamentals"]]
        })
    }

    #[test]
    fn parses_every_section() {
        let r = CompositionRules::from_value(&sample()).unwrap();
        assert_eq!(r.tier_definitions.len(), 2);
        assert!(r.tier_definitions[0].is_active);
        assert_eq!(r.additive_properties, ["Charge_e", "Mass_amu", "Mass_kg"]);
        assert!(r.is_integral("Charge_e"));
        assert!(!r.is_integral("Mass_kg"), "masses must never be snapped (D1)");
        assert_eq!(r.placeholder_prefixes, ["demo_", "example_"]);
        assert_eq!(
            r.resolution_levels,
            vec![
                vec!["atoms".to_string()],
                vec!["subatomic".to_string(), "fundamentals".to_string()]
            ]
        );
        assert!(r.is_placeholder("demo_graviton"));
    }

    #[test]
    fn an_integral_property_must_also_be_additive() {
        let mut cfg = sample();
        cfg["integral_properties"] = json!(["Spin_hbar"]);
        let e = CompositionRules::from_value(&cfg).unwrap_err();
        assert!(e.to_string().contains("Spin_hbar"), "{e}");
    }

    #[test]
    fn malformed_sections_are_errors_not_defaults() {
        let mut cfg = sample();
        cfg["additive_properties"] = json!("Charge_e");
        assert!(CompositionRules::from_value(&cfg).is_err());
        let mut cfg = sample();
        cfg["resolution_levels"] = json!([1]);
        assert!(CompositionRules::from_value(&cfg).is_err());
        let mut cfg = sample();
        cfg["tier_definitions"] = json!([{"name": "x"}]);
        assert!(CompositionRules::from_value(&cfg).is_err());
    }

    #[test]
    fn missing_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(CompositionRules::load(dir.path()).is_err());
    }

    #[test]
    fn shipped_rules_parse_and_declare_the_integral_quantum_numbers() {
        let data =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
        if !data.is_dir() {
            return;
        }
        let r = CompositionRules::load(&data).expect("shipped rules");
        for p in ["Charge_e", "BaryonNumber_B", "LeptonNumber_L"] {
            assert!(r.is_integral(p), "{p} must be integral");
        }
        for p in ["Mass_amu", "Mass_MeVc2", "Mass_kg"] {
            assert!(!r.is_integral(p), "{p} must not be snapped");
        }
        assert!(!r.resolution_levels.is_empty());
        assert!(r.mass_sources.is_some());
    }
}
