//====== periodica/rust/periodica_core/src/mass_source.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # mass_source
//!
//! Curated masses for generated tiers (D3).
//!
//! Composing an atom from free protons, neutrons and electrons ignores nuclear
//! binding energy, so the old generated tiers were systematically heavy: the
//! C-12 atom came out at 12.0989 u instead of exactly 12, water at 18.148 u
//! instead of 18.015 u. Composition stays the right model for *what* an atom
//! is made of; its *mass* has to come from measurement.
//!
//! The rules' `mass_sources` block says which tier takes its mass from which
//! reference dataset:
//!
//! - `element` -- the CIAAW 2021 standard atomic weight of element Z
//!   (`reference/masses/ciaaw2021_standard_atomic_weights.json`; the abridged
//!   value where the standard weight is an interval). Elements CIAAW gives no
//!   weight for (no stable isotope) fall back to the AME2020 mass of the
//!   nuclide the entry's own Composition names.
//! - `nuclide` -- the AME2020 atomic mass of nuclide (Z, A)
//!   (`reference/masses/ame2020_atomic_masses.json`).
//!
//! Both are neutral-atom masses. An entry with fewer or more electrons than
//! protons (an ion) is corrected by that many electron masses; the electrons'
//! binding energies (eV against GeV) are below 1e-8 relative and ignored.
//!
//! [`MassSources::apply`] overwrites `Mass_amu` with the curated value,
//! derives `Mass_MeVc2` and `Mass_kg` from it with the CODATA 2018 factors in
//! the rules, and records where the number came from in `MassSource`. It is a
//! build-time step: `scripts/build_*.py` compose an entry, apply this, and save
//! it; the registry then serves the generated file. Molecules and every other
//! composed tier inherit the curated values by composing curated atoms.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use periodica_mat::jsonc;
use serde_json::Value;

use crate::rules::CompositionRules;

/// How a tier's entries get their mass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassPolicy {
    /// Standard atomic weight of the element (Z from the composition).
    Element,
    /// Atomic mass of the nuclide (Z, A) from the composition.
    Nuclide,
}

/// Errors applying a curated mass to an entry.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MassSourceError {
    #[error("MassSource: entry has no integer Composition.{0} count")]
    MissingCount(String),
    #[error("MassSource: no curated mass for element Z={z} or nuclide {nuclide}")]
    NoReference { z: u32, nuclide: String },
}

#[derive(Debug, Clone)]
struct ElementWeight {
    symbol: String,
    mass_amu: Option<f64>,
    /// `standard`, `abridged` or `none` (see the dataset's provenance).
    kind: String,
}

#[derive(Debug, Clone)]
struct NuclideMass {
    name: String,
    mass_u: f64,
    estimated: bool,
}

/// The `mass_sources` rules plus the two loaded reference datasets.
#[derive(Debug, Clone)]
pub struct MassSources {
    pub mev_per_u: f64,
    pub kg_per_u: f64,
    pub electron_mass_u: f64,
    /// Composition keys holding the proton, neutron and electron counts.
    keys: [String; 3],
    tiers: HashMap<String, MassPolicy>,
    element_label: String,
    nuclide_label: String,
    elements: HashMap<u32, ElementWeight>,
    nuclides: HashMap<(u32, u32), NuclideMass>,
}

impl MassSources {
    /// Build from the rules. `Ok(None)` when the rules declare no
    /// `mass_sources` block. Dataset paths are relative to `data_root`.
    pub fn load(rules: &CompositionRules, data_root: &Path) -> Result<Option<Self>> {
        let Some(cfg) = rules.mass_sources.as_ref() else {
            return Ok(None);
        };
        let f = |key: &str| {
            cfg.get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow!("mass_sources.{key}: expected a number"))
        };
        let keys_cfg = cfg.get("composition_keys");
        let key = |role: &str, default: &str| {
            keys_cfg
                .and_then(|k| k.get(role))
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_string()
        };
        let dataset = |key: &str| -> Result<(String, Value)> {
            let spec = cfg
                .get(key)
                .ok_or_else(|| anyhow!("mass_sources.{key} is missing"))?;
            let path = spec
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("mass_sources.{key}.path: expected a string"))?;
            let label = spec
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("mass_sources.{key}.label: expected a string"))?;
            let full = data_root.join(path);
            let doc = jsonc::from_path(&full)
                .with_context(|| format!("mass_sources: reading {}", full.display()))?;
            Ok((label.to_string(), doc))
        };

        let (element_label, element_doc) = dataset("element_weights")?;
        let (nuclide_label, nuclide_doc) = dataset("nuclide_masses")?;

        let mut tiers = HashMap::new();
        if let Some(map) = cfg.get("tiers").and_then(Value::as_object) {
            for (tier, policy) in map {
                let p = match policy.as_str() {
                    Some("element") => MassPolicy::Element,
                    Some("nuclide") => MassPolicy::Nuclide,
                    _ => {
                        return Err(anyhow!(
                            "mass_sources.tiers.{tier}: expected \"element\" or \"nuclide\""
                        ))
                    }
                };
                tiers.insert(tier.clone(), p);
            }
        }

        let mut elements = HashMap::new();
        for e in element_doc
            .get("elements")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("element weights: no `elements` list"))?
        {
            let z = e
                .get("Z")
                .and_then(Value::as_u64)
                .ok_or_else(|| anyhow!("element weights: entry without Z: {e}"))? as u32;
            elements.insert(
                z,
                ElementWeight {
                    symbol: e.get("symbol").and_then(Value::as_str).unwrap_or("?").to_string(),
                    mass_amu: e.get("Mass_amu").and_then(Value::as_f64),
                    kind: e
                        .get("value_kind")
                        .and_then(Value::as_str)
                        .unwrap_or("standard")
                        .to_string(),
                },
            );
        }

        let mut nuclides = HashMap::new();
        for n in nuclide_doc
            .get("nuclides")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("nuclide masses: no `nuclides` list"))?
        {
            let z = n.get("Z").and_then(Value::as_u64);
            let a = n.get("A").and_then(Value::as_u64);
            let m = n.get("atomic_mass_u").and_then(Value::as_f64);
            let (Some(z), Some(a), Some(m)) = (z, a, m) else {
                return Err(anyhow!("nuclide masses: incomplete entry {n}"));
            };
            nuclides.insert(
                (z as u32, a as u32),
                NuclideMass {
                    name: n
                        .get("nuclide")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("Z{z}-{a}")),
                    mass_u: m,
                    estimated: n.get("estimated").and_then(Value::as_bool).unwrap_or(false),
                },
            );
        }

        Ok(Some(MassSources {
            mev_per_u: f("MeV_per_u")?,
            kg_per_u: f("kg_per_u")?,
            electron_mass_u: f("electron_mass_u")?,
            keys: [
                key("protons", "P"),
                key("neutrons", "N"),
                key("electrons", "E"),
            ],
            tiers,
            element_label,
            nuclide_label,
            elements,
            nuclides,
        }))
    }

    /// The policy for `tier`, if it has a curated mass source.
    pub fn policy(&self, tier: &str) -> Option<MassPolicy> {
        self.tiers.get(tier).copied()
    }

    /// Overwrite the entry's masses with the curated value for its tier.
    ///
    /// Returns `Ok(false)` (entry untouched) for a tier with no declared
    /// source.
    pub fn apply(&self, tier: &str, entry: &mut Value) -> Result<bool, MassSourceError> {
        let Some(policy) = self.policy(tier) else {
            return Ok(false);
        };
        let [p_key, n_key, e_key] = &self.keys;
        let count = |k: &str| -> Result<u32, MassSourceError> {
            entry
                .get("Composition")
                .and_then(|c| c.get(k))
                .and_then(integral_count)
                .ok_or_else(|| MassSourceError::MissingCount(k.to_string()))
        };
        let (z, n, electrons) = (count(p_key)?, count(n_key)?, count(e_key)?);
        let a = z + n;

        let (neutral, mut source) = self.neutral_mass(policy, z, a)?;
        let excess = i64::from(electrons) - i64::from(z);
        let mass_amu = neutral + excess as f64 * self.electron_mass_u;
        if excess != 0 {
            let k = excess.unsigned_abs();
            let noun = if k == 1 { "electron mass" } else { "electron masses" };
            let verb = if excess > 0 { "plus" } else { "minus" };
            source.push_str(&format!(", {verb} {k} {noun}"));
        }

        if let Some(obj) = entry.as_object_mut() {
            obj.insert("Mass_amu".into(), number(mass_amu));
            obj.insert("Mass_MeVc2".into(), number(mass_amu * self.mev_per_u));
            obj.insert("Mass_kg".into(), number(mass_amu * self.kg_per_u));
            obj.insert("MassSource".into(), Value::String(source));
        }
        Ok(true)
    }

    fn neutral_mass(&self, policy: MassPolicy, z: u32, a: u32) -> Result<(f64, String), MassSourceError> {
        if policy == MassPolicy::Element {
            if let Some(el) = self.elements.get(&z) {
                if let Some(m) = el.mass_amu {
                    let what = match el.kind.as_str() {
                        "abridged" => "abridged atomic weight (the standard weight is an interval)",
                        _ => "standard atomic weight",
                    };
                    return Ok((m, format!("{} {what} of {}", self.element_label, el.symbol)));
                }
            }
        }
        match self.nuclides.get(&(z, a)) {
            Some(nuc) => {
                let mut s = format!("{} atomic mass of {}", self.nuclide_label, nuc.name);
                if nuc.estimated {
                    s.push_str(" (estimated)");
                }
                if policy == MassPolicy::Element {
                    s.push_str(&format!(
                        "; {} gives no standard atomic weight for this element",
                        self.element_label
                    ));
                }
                Ok((nuc.mass_u, s))
            }
            None => Err(MassSourceError::NoReference {
                z,
                nuclide: format!("Z={z} A={a}"),
            }),
        }
    }
}

fn integral_count(v: &Value) -> Option<u32> {
    if let Some(u) = v.as_u64() {
        return u32::try_from(u).ok();
    }
    let f = v.as_f64()?;
    (f >= 0.0 && f.fract() == 0.0 && f <= f64::from(u32::MAX)).then_some(f as u32)
}

fn number(x: f64) -> Value {
    serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bundled() -> Option<(CompositionRules, std::path::PathBuf)> {
        let data =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
        if !data.is_dir() {
            return None;
        }
        Some((CompositionRules::load(&data).expect("rules"), data))
    }

    fn sources() -> Option<MassSources> {
        let (rules, data) = bundled()?;
        Some(MassSources::load(&rules, &data).expect("load").expect("declared"))
    }

    fn atom(p: u32, n: u32, e: u32) -> Value {
        json!({"Composition": {"P": p, "N": n, "E": e}, "Mass_amu": 0.0, "Mass_kg": 0})
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        ((a - b) / b).abs() <= rel
    }

    #[test]
    fn atoms_take_the_ciaaw_standard_atomic_weight() {
        let Some(s) = sources() else { return };
        let mut fe = atom(26, 30, 26);
        assert!(s.apply("atoms", &mut fe).unwrap());
        assert_eq!(fe["Mass_amu"], 55.845);
        assert!(close(fe["Mass_MeVc2"].as_f64().unwrap(), 55.845 * 931.49410242, 1e-15));
        assert!(close(fe["Mass_kg"].as_f64().unwrap(), 55.845 * 1.66053906660e-27, 1e-15));
        assert_eq!(fe["MassSource"], "CIAAW 2021 standard atomic weight of Fe");
    }

    #[test]
    fn interval_elements_use_the_abridged_value() {
        let Some(s) = sources() else { return };
        let mut h = atom(1, 0, 1);
        s.apply("atoms", &mut h).unwrap();
        assert_eq!(h["Mass_amu"], 1.008);
        assert!(h["MassSource"].as_str().unwrap().contains("abridged"));
    }

    #[test]
    fn elements_without_a_standard_weight_use_their_nuclide() {
        let Some(s) = sources() else { return };
        let mut tc = atom(43, 54, 43);
        s.apply("atoms", &mut tc).unwrap();
        assert!(close(tc["Mass_amu"].as_f64().unwrap(), 96.90636072, 1e-12));
        let src = tc["MassSource"].as_str().unwrap();
        assert!(src.contains("AME2020") && src.contains("Tc-97"), "{src}");
    }

    #[test]
    fn isotopes_take_the_ame2020_atomic_mass() {
        let Some(s) = sources() else { return };
        let mut c12 = atom(6, 6, 6);
        s.apply("isotopes", &mut c12).unwrap();
        assert_eq!(c12["Mass_amu"], 12.0, "C-12 is 12 u by definition");
        let mut h2 = atom(1, 1, 1);
        s.apply("isotopes", &mut h2).unwrap();
        assert!(close(h2["Mass_amu"].as_f64().unwrap(), 2.014101777844, 1e-12));
    }

    #[test]
    fn ions_are_corrected_by_their_missing_or_extra_electrons() {
        let Some(s) = sources() else { return };
        let me = s.electron_mass_u;
        let mut k = atom(19, 20, 18);
        s.apply("ions", &mut k).unwrap();
        assert!(close(k["Mass_amu"].as_f64().unwrap(), 39.0983 - me, 1e-14));
        assert!(k["MassSource"].as_str().unwrap().ends_with("minus 1 electron mass"));
        let mut o2 = atom(8, 8, 10);
        s.apply("ions", &mut o2).unwrap();
        assert!(close(o2["Mass_amu"].as_f64().unwrap(), 15.999 + 2.0 * me, 1e-14));
        assert!(o2["MassSource"].as_str().unwrap().ends_with("plus 2 electron masses"));
    }

    #[test]
    fn other_tiers_are_untouched() {
        let Some(s) = sources() else { return };
        let mut m = json!({"Composition": {"H": 2, "O": 1}, "Mass_amu": 18.015});
        assert!(!s.apply("molecules", &mut m).unwrap());
        assert_eq!(m, json!({"Composition": {"H": 2, "O": 1}, "Mass_amu": 18.015}));
    }

    #[test]
    fn a_missing_count_or_reference_is_an_error() {
        let Some(s) = sources() else { return };
        let mut bad = json!({"Composition": {"P": 1}});
        assert_eq!(
            s.apply("atoms", &mut bad),
            Err(MassSourceError::MissingCount("N".into()))
        );
        let mut unknown = atom(6, 40, 6);
        assert!(matches!(
            s.apply("isotopes", &mut unknown),
            Err(MassSourceError::NoReference { .. })
        ));
    }

    #[test]
    fn rules_without_mass_sources_load_none() {
        let rules = CompositionRules::default();
        assert!(MassSources::load(&rules, Path::new(".")).unwrap().is_none());
    }

    #[test]
    fn every_ciaaw_element_with_a_weight_or_nuclide_is_covered() {
        // Each of the 118 atoms the periodic-table input builds must have a
        // curated mass: either a CIAAW weight or an AME2020 nuclide.
        let Some(s) = sources() else { return };
        for z in 1..=118u32 {
            let el = s.elements.get(&z).expect("CIAAW lists every element");
            if el.mass_amu.is_none() {
                assert!(
                    s.nuclides.keys().any(|(nz, _)| *nz == z),
                    "Z={z} ({}) has neither a weight nor a nuclide",
                    el.symbol
                );
            }
        }
    }
}
