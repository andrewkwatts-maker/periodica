//====== periodica/rust/periodica-mat/src/ingest.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # ingest
//!
//! Turns a raw `serde_json::Value` datasheet into a [`MaterialRecord`].
//!
//! ## The two schemas, one code path
//!
//! `active/materials/*.json` nests properties in typed blocks:
//! ```json
//! { "PhysicalProperties": { "Density_g_cm3": 2.775 },
//!   "MechanicalProperties": { "YieldStrength_MPa": 271.0 } }
//! ```
//! while `derived/alloys/*.json` uses one flat block with different unit
//! spellings:
//! ```json
//! { "Properties": { "Density_kgm3": 7870, "ThermalConductivity_WmK": 51.9 } }
//! ```
//! and `active/elements/*.json` puts bare snake_case scalars at the root:
//! ```json
//! { "density": 8.988e-05, "melting_point": 14.01, "ionization_energy": 13.598 }
//! ```
//!
//! All three are handled by one recursive walk. Canonicalising each key (see
//! [`crate::canon`]) collapses the unit-punctuation differences, and the alias
//! table resolves the genuine synonyms.
//!
//! ## Strict inside blocks, opportunistic at the root
//!
//! Inside a recognised property block every numeric leaf must be either mapped
//! or explicitly ignored; anything else is recorded in
//! [`Provenance::unknown_keys`] so new data cannot quietly lose a property.
//!
//! At the **root** the rule is relaxed to opportunistic: element datasheets mix
//! properties (`density`) with descriptive scalars (`group`, `period`,
//! `atomic_number`), and reporting every one of those as "unknown" would bury
//! the real signal. Root keys that map are taken; the rest are passed over.

use serde_json::Value;

use crate::canon::{canon, is_ignored, lookup};
use crate::derive::fill_derived;
use crate::property::PropertyId;
use crate::record::{
    CrystalSystem, Inclusions, Lattice, MaterialRecord, Microstructure, NoiseKind, Phase,
    Provenance, Texture, Tier,
};
use crate::table::{PropertyTable, Source};

/// Blocks whose numeric leaves are material properties.
///
/// Derived from a census of all 954 bundled datasheets; these are the only
/// containers that hold SI-meaningful scalars.
const PROPERTY_BLOCKS: &[&str] = &[
    "physicalproperties",
    "mechanicalproperties",
    "elasticproperties",
    "strengthproperties",
    "thermalproperties",
    "electricalproperties",
    "opticalproperties",
    "fracturemechanics",
    "fatigueproperties",
    "failurecriteria",
    "crackgrowthparameters",
    "charpyimpactj",
    "hardness",
    "ductility",
    "properties",
];

/// Blocks handled by dedicated extractors, or with no property content. The
/// property walk must not descend into these -- doing so would, for example,
/// read a phase's `Hardness_HV` as if it were the bulk hardness.
const NON_PROPERTY_BLOCKS: &[&str] = &[
    "composition",
    "components",
    "microstructure",
    "latticeproperties",
    "phasecomposition",
    // Per-mineral data (`MineralComposition.Quartz.properties.Density_kg_m3`
    // in Granite_Westerly). Its nested `properties` blocks would otherwise be
    // read as bulk, and the first mineral's density beat the real bulk value.
    "mineralcomposition",
    "crystallographicorientation",
    "field",
    "derivation",
    "simulationparameters",
    "serviceconditions",
    "processingmethods",
    "applications",
    "corrosionresistance",
    "decaymodes",
    "isotopes",
    "residues",
    "atoms3d",
    "bonds3d",
    "bonds",
    "atoms",
    "discovery",
    "secondarystructures",
    "aminoacidcomposition",
    "basecomposition",
];

/// Result of ingesting one datasheet.
pub type Result<T> = std::result::Result<T, IngestError>;

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("ingest: datasheet root must be a JSON object, got {0}")]
    NotAnObject(&'static str),
}

/// Ingest a raw datasheet into a canonical [`MaterialRecord`].
///
/// `name_hint` is used when the JSON carries no `Name` field -- normally the
/// file stem, which is how the registry keys entries.
pub fn ingest(value: &Value, name_hint: &str) -> Result<MaterialRecord> {
    let obj = value
        .as_object()
        .ok_or_else(|| IngestError::NotAnObject(json_type_name(value)))?;

    let mut bulk = PropertyTable::new();
    let mut unknown = Vec::new();

    // 1. Root-level scalars (element datasheets), opportunistic.
    for (k, v) in obj {
        if let Some(n) = as_number(v) {
            let c = canon(k);
            if let Some(m) = lookup(&c) {
                bulk.set_if_better(m.property, m.conv.apply(n), Source::Datasheet);
            }
        }
    }

    // 2. Recognised property blocks, strict.
    walk_blocks(value, false, &mut bulk, &mut unknown);

    // 3. Structured sub-objects.
    let lattice = extract_lattice(obj);
    let micro = extract_microstructure(obj);
    let texture = extract_texture(obj);
    let mut phases = extract_phases(obj);
    if phases.is_empty() {
        phases = extract_field_phases(obj);
    }

    let mut record = MaterialRecord {
        name: obj
            .get("Name")
            .or_else(|| obj.get("name"))
            .and_then(Value::as_str)
            .unwrap_or(name_hint)
            .to_string(),
        tier: Tier::Unknown,
        category: obj
            .get("Category")
            .or_else(|| obj.get("category"))
            .and_then(Value::as_str)
            .map(str::to_string),
        composition: extract_composition(obj),
        bulk,
        lattice,
        phases,
        micro,
        texture,
        color_srgb: extract_color(obj),
        provenance: Provenance {
            source: Some(name_hint.to_string()),
            root: None,
            unknown_keys: {
                unknown.sort_unstable();
                unknown.dedup();
                unknown
            },
        },
    };

    // 4. Fill exact relations (conductivity from resistivity, G/K from E+nu...).
    fill_derived(&mut record.bulk);

    Ok(record)
}

/// Recursively walk for property blocks.
///
/// `inside` tracks whether the current subtree is within a recognised property
/// block. Once inside, every numeric leaf is accounted for.
fn walk_blocks(value: &Value, inside: bool, bulk: &mut PropertyTable, unknown: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                let c = canon(k);
                if NON_PROPERTY_BLOCKS.contains(&c.as_str()) {
                    continue;
                }
                let now_inside = inside || PROPERTY_BLOCKS.contains(&c.as_str());
                match v {
                    Value::Object(_) | Value::Array(_) => walk_blocks(v, now_inside, bulk, unknown),
                    _ => {
                        if !inside {
                            // Root-level scalars were handled opportunistically
                            // in step 1; do not re-report them as unknown.
                            continue;
                        }
                        let Some(n) = as_number(v) else { continue };
                        match lookup(&c) {
                            Some(m) => {
                                bulk.set_if_better(m.property, m.conv.apply(n), Source::Datasheet);
                            }
                            None if is_ignored(&c) => {}
                            None => unknown.push(c),
                        }
                    }
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                walk_blocks(item, inside, bulk, unknown);
            }
        }
        _ => {}
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "object",
        Value::Array(_) => "array",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "bool",
        Value::Null => "null",
    }
}

/// Read a JSON number, rejecting booleans (`serde_json` keeps those separate,
/// but be explicit) and non-finite values.
fn as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
        _ => None,
    }
}

fn get_num(map: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    map.get(key).and_then(as_number)
}

/// Case/punctuation-insensitive object lookup, so `LatticeProperties` and
/// `lattice_properties` both resolve.
fn get_obj<'a>(
    map: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Option<&'a serde_json::Map<String, Value>> {
    let want = canon(key);
    map.iter()
        .find(|(k, _)| canon(k) == want)
        .and_then(|(_, v)| v.as_object())
}

// ── Structured extractors ─────────────────────────────────────────────────

fn extract_lattice(obj: &serde_json::Map<String, Value>) -> Option<Lattice> {
    let lp = get_obj(obj, "LatticeProperties")?;
    let mut l = Lattice {
        system: lp
            .get("PrimaryStructure")
            .and_then(Value::as_str)
            .map(CrystalSystem::parse)
            .unwrap_or_default(),
        ..Default::default()
    };
    if let Some(p) = get_obj(lp, "LatticeParameters") {
        // Datasheets store picometres and degrees; the record is SI.
        l.abc_m = [
            get_num(p, "a_pm").unwrap_or(0.0) * 1e-12,
            get_num(p, "b_pm").unwrap_or(0.0) * 1e-12,
            get_num(p, "c_pm").unwrap_or(0.0) * 1e-12,
        ];
        l.angles_rad = [
            get_num(p, "alpha_deg").unwrap_or(90.0).to_radians(),
            get_num(p, "beta_deg").unwrap_or(90.0).to_radians(),
            get_num(p, "gamma_deg").unwrap_or(90.0).to_radians(),
        ];
    }
    l.packing_factor = get_num(lp, "AtomicPackingFactor");
    l.coordination_number = get_num(lp, "CoordinationNumber").map(|v| v as u32);
    l.atoms_per_cell = get_num(lp, "AtomsPerUnitCell").map(|v| v as u32);
    l.space_group = get_num(lp, "SpaceGroupNumber").map(|v| v as u32);
    l.burgers_vector_m = get_num(lp, "BurgersVector_pm").map(|v| v * 1e-12);
    Some(l)
}

fn extract_microstructure(obj: &serde_json::Map<String, Value>) -> Option<Microstructure> {
    let ms = get_obj(obj, "Microstructure")?;
    let mut m = Microstructure::default();

    if let Some(g) = get_obj(ms, "GrainStructure") {
        let d = &mut m.grains;
        if let Some(v) = get_num(g, "AverageGrainSize_um") {
            d.average_size_m = v * 1e-6;
        }
        if let Some(v) = get_num(g, "GrainSizeStdDev") {
            d.size_std_dev = v;
        }
        d.astm_number = get_num(g, "ASTMGrainSizeNumber");
        if let Some(v) = get_num(g, "VoronoiSeedDensity_per_mm2") {
            // The datasheets quote an areal seed density, measured on a polished
            // section. Raising it to a volumetric one uses the standard
            // stereological relation for equiaxed grains, n_v = n_a^(3/2).
            //
            // This is self-consistent with the quoted grain size: 400/mm^2 is
            // 4e8/m^2, so n_v = 8e12/m^3, and 1/(50 um)^3 is also 8e12/m^3.
            let areal_per_m2 = v * 1.0e6;
            d.seed_density_per_m3 = areal_per_m2.powf(1.5);
        } else if d.average_size_m > 0.0 {
            d.seed_density_per_m3 = 1.0 / d.average_size_m.powi(3);
        }
        if let Some(v) = get_num(g, "GrainBoundaryWidth_nm") {
            d.boundary_width_m = v * 1e-9;
        }
        if let Some(v) = get_num(g, "GrainAspectRatio") {
            d.aspect_ratio = v;
        }
        d.twin_density_per_m = get_num(g, "TwinDensity_per_mm").map(|v| v * 1e3);
    }

    if let Some(p) = get_obj(ms, "PhaseDistribution") {
        let d = &mut m.phase_distribution;
        if let Some(s) = p.get("NoiseType").and_then(Value::as_str) {
            d.noise = NoiseKind::parse(s);
        }
        if let Some(v) = get_num(p, "NoiseScale") {
            d.scale_rel = v;
        }
        if let Some(v) = get_num(p, "NoiseOctaves") {
            d.octaves = v.clamp(1.0, 8.0) as u8;
        }
        if let Some(v) = get_num(p, "NoisePersistence") {
            d.persistence = v;
        }
        if let Some(v) = get_num(p, "PhaseThreshold") {
            d.threshold = v;
        }
    }

    if let Some(df) = get_obj(ms, "Defects") {
        let d = &mut m.defects;
        // Two shapes exist on disk: flat, and grouped under PointDefects /
        // Dislocations / GrainBoundaries. Try the group first, then the flat
        // form, so both schemas land in the same struct.
        let point = get_obj(df, "PointDefects").unwrap_or(df);
        d.vacancy_concentration = get_num(point, "VacancyConcentration")
            .or_else(|| get_num(df, "VacancyConcentration"))
            .unwrap_or(0.0);
        d.interstitial_concentration = get_num(point, "InterstitialConcentration")
            .or_else(|| get_num(df, "InterstitialConcentration"))
            .unwrap_or(0.0);

        let disl = get_obj(df, "Dislocations").unwrap_or(df);
        d.dislocation_density_per_m2 = get_num(disl, "DislocationDensity_per_m2")
            .or_else(|| get_num(df, "DislocationDensity_per_m2"))
            .unwrap_or(0.0);

        let gb = get_obj(df, "GrainBoundaries").unwrap_or(df);
        d.low_angle_boundary_fraction = get_num(gb, "LowAngleBoundaryFraction").unwrap_or(0.0);
        d.high_angle_boundary_fraction = get_num(gb, "HighAngleBoundaryFraction").unwrap_or(0.0);
        d.twin_boundary_fraction = get_num(gb, "TwinBoundaryFraction").unwrap_or(0.0);

        if let Some(sf) = get_obj(df, "StressField") {
            d.residual_stress_pa = get_num(sf, "MaxStress_MPa").map(|v| v * 1e6);
        }
    }

    if let Some(inc) = get_obj(ms, "Inclusions") {
        m.inclusions = Inclusions {
            // per mm^3 -> per m^3
            density_per_m3: get_num(inc, "Density_per_mm3").unwrap_or(0.0) * 1e9,
            average_size_m: get_num(inc, "AverageSize_um").unwrap_or(0.0) * 1e-6,
        };
    }

    Some(m)
}

fn extract_texture(obj: &serde_json::Map<String, Value>) -> Option<Texture> {
    let co = get_obj(obj, "CrystallographicOrientation")?;
    Some(Texture {
        preferred: co
            .get("PreferredOrientation")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        strength_mrd: get_num(co, "TextureStrength_mrd").unwrap_or(1.0),
        // The datasheets name a texture *type* ("Random", "Rolling") but never
        // a vector. Rolling textures align with the processing direction, which
        // the runtime takes as local +X; a random texture has no axis.
        axis: match co.get("TextureType").and_then(Value::as_str).map(canon) {
            Some(ref t) if t != "random" && !t.is_empty() => Some([1.0, 0.0, 0.0]),
            _ => None,
        },
    })
}

fn extract_phases(obj: &serde_json::Map<String, Value>) -> Vec<Phase> {
    let Some(pc) = get_obj(obj, "PhaseComposition") else {
        return Vec::new();
    };
    let Some(list) = pc.get("Phases").and_then(Value::as_array) else {
        return Vec::new();
    };
    list.iter()
        .filter_map(Value::as_object)
        .map(|p| {
            let mut props = PropertyTable::new();
            for (k, v) in p {
                if let (Some(n), Some(m)) = (as_number(v), lookup(&canon(k))) {
                    props.set(m.property, m.conv.apply(n), Source::Datasheet);
                }
            }
            Phase {
                name: p
                    .get("Name")
                    .and_then(Value::as_str)
                    .unwrap_or("phase")
                    .to_string(),
                symbol: p.get("Symbol").and_then(Value::as_str).map(str::to_string),
                structure: p
                    .get("Structure")
                    .and_then(Value::as_str)
                    .map(CrystalSystem::parse)
                    .unwrap_or_default(),
                volume_fraction: get_num(p, "VolumePercent").unwrap_or(0.0) * 0.01,
                magnetic: p.get("Magnetic").and_then(Value::as_bool).unwrap_or(false),
                properties: props,
            }
        })
        .collect()
}

/// Phases declared the other way: in a `Field` block of a `derived/alloys`
/// entry, where fractions are already 0..1 rather than percentages.
fn extract_field_phases(obj: &serde_json::Map<String, Value>) -> Vec<Phase> {
    let Some(field) = get_obj(obj, "Field") else {
        return Vec::new();
    };
    let Some(phases) = get_obj(field, "phases") else {
        return Vec::new();
    };
    let mut out: Vec<Phase> = phases
        .iter()
        .filter_map(|(name, v)| v.as_object().map(|o| (name, o)))
        .map(|(name, o)| {
            let mut props = PropertyTable::new();
            for (k, v) in o {
                if canon(k) == "fraction" {
                    continue;
                }
                if let (Some(n), Some(m)) = (as_number(v), lookup(&canon(k))) {
                    props.set(m.property, m.conv.apply(n), Source::Datasheet);
                }
            }
            Phase {
                name: name.clone(),
                symbol: None,
                structure: CrystalSystem::Unknown,
                volume_fraction: get_num(o, "fraction").unwrap_or(0.0),
                magnetic: false,
                properties: props,
            }
        })
        .collect();
    // Deterministic order regardless of JSON key iteration order, so a
    // re-ingest of the same file always produces an identical record.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Composition as element symbol -> mass fraction in `0..=1`.
///
/// Three encodings appear on disk: a `Components` array of min/max percent
/// ranges, a `Composition` map of already-normalised fractions, and a
/// `Composition` map of atom counts (molecules). They are distinguished by
/// whether the values sum to about 1.
fn extract_composition(obj: &serde_json::Map<String, Value>) -> Vec<(String, f64)> {
    if let Some(list) = obj.get("Components").and_then(Value::as_array) {
        let mut out: Vec<(String, f64)> = list
            .iter()
            .filter_map(Value::as_object)
            .filter_map(|c| {
                let sym = c
                    .get("Element")
                    .or_else(|| c.get("Symbol"))
                    .or_else(|| c.get("Name"))
                    .and_then(Value::as_str)?;
                // Take the midpoint of the specified range.
                let lo = get_num(c, "MinPercent")?;
                let hi = get_num(c, "MaxPercent").unwrap_or(lo);
                Some((sym.to_string(), (lo + hi) * 0.5 * 0.01))
            })
            .collect();
        if !out.is_empty() {
            out.sort_by(|a, b| a.0.cmp(&b.0));
            return out;
        }
    }

    if let Some(map) = get_obj(obj, "Composition") {
        let mut out: Vec<(String, f64)> = map
            .iter()
            .filter_map(|(k, v)| as_number(v).map(|n| (k.clone(), n)))
            .collect();
        let total: f64 = out.iter().map(|(_, v)| *v).sum();
        if total > 0.0 {
            // Atom counts (or percentages) get normalised to fractions;
            // values that already sum to 1 pass through unchanged.
            for (_, v) in out.iter_mut() {
                *v /= total;
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        return out;
    }

    Vec::new()
}

/// Parse the `Color` hex string into linear sRGB.
fn extract_color(obj: &serde_json::Map<String, Value>) -> [f32; 3] {
    let default = [0.75, 0.75, 0.78];
    let Some(s) = obj
        .get("Color")
        .or_else(|| obj.get("color"))
        .and_then(Value::as_str)
    else {
        return default;
    };
    let hex = s.trim_start_matches('#');
    if hex.len() < 6 {
        return default;
    }
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    match (byte(0), byte(2), byte(4)) {
        (Some(r), Some(g), Some(b)) => [
            srgb_to_linear(r as f32 / 255.0),
            srgb_to_linear(g as f32 / 255.0),
            srgb_to_linear(b as f32 / 255.0),
        ],
        _ => default,
    }
}

/// sRGB electro-optical transfer function. Colours must be linearised before
/// they reach a shader or any averaging maths.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convenience: does this datasheet describe something the runtime can treat
/// as a material? Used by the registry to decide what to expose in the app's
/// material library.
pub fn is_material_like(record: &MaterialRecord) -> bool {
    record.bulk.has(PropertyId::Density)
        && (record.bulk.has(PropertyId::YoungsModulus)
            || record.bulk.has(PropertyId::UltimateTensileStrength)
            || record.bulk.has(PropertyId::ThermalConductivity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::PropertyId as P;
    use approx::assert_relative_eq;
    use serde_json::json;

    #[test]
    fn nested_active_materials_schema() {
        let v = json!({
            "Name": "Fe-Cd_90_10_Hot_rolled",
            "Category": "Ternary Alloy",
            "PhysicalProperties": {
                "Density_g_cm3": 7.777,
                "MeltingPoint_K": 1726.5,
                "ThermalConductivity_W_mK": 54.2,
                "ElectricalResistivity_Ohm_m": 1.474e-07
            },
            "MechanicalProperties": {
                "TensileStrength_MPa": 653.0,
                "YieldStrength_MPa": 392.0,
                "Hardness_HV": 196.0
            }
        });
        let r = ingest(&v, "stem").unwrap();
        assert_eq!(r.name, "Fe-Cd_90_10_Hot_rolled");
        assert_eq!(r.category.as_deref(), Some("Ternary Alloy"));
        // g/cm^3 -> kg/m^3
        assert_relative_eq!(r.bulk.get(P::Density).unwrap(), 7777.0, max_relative = 1e-9);
        // MPa -> Pa
        assert_relative_eq!(
            r.bulk.get(P::UltimateTensileStrength).unwrap(),
            6.53e8,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            r.bulk.get(P::YieldStrength).unwrap(),
            3.92e8,
            max_relative = 1e-9
        );
        assert_eq!(r.bulk.get(P::HardnessVickers), Some(196.0));
        assert!(
            r.is_fully_ingested(),
            "unknown: {:?}",
            r.provenance.unknown_keys
        );
    }

    #[test]
    fn flat_derived_alloys_schema_with_other_unit_spelling() {
        let v = json!({
            "Name": "Steel-1018",
            "Composition": {"C": 0.01, "Fe": 0.99},
            "Field": {
                "model": "microstructure_voronoi",
                "macro_scale_m": 1e-4,
                "phases": {
                    "ferrite":  {"fraction": 0.85, "YoungsModulus_GPa": 195, "HardnessHB": 100},
                    "pearlite": {"fraction": 0.15, "YoungsModulus_GPa": 220, "HardnessHB": 200}
                }
            },
            "Properties": {
                "Density_kgm3": 7870,
                "YoungsModulus_GPa": 200,
                "ThermalConductivity_WmK": 51.9,
                "ElectricalResistivity_ohmm": 1.43e-7,
                "PoissonsRatio": 0.29
            }
        });
        let r = ingest(&v, "Steel-1018").unwrap();
        assert_eq!(r.bulk.get(P::Density), Some(7870.0));
        assert_relative_eq!(
            r.bulk.get(P::YoungsModulus).unwrap(),
            2.0e11,
            max_relative = 1e-9
        );
        // The `_WmK` spelling must reach the same property as `_W_mK`.
        assert_eq!(r.bulk.get(P::ThermalConductivity), Some(51.9));
        assert_eq!(r.composition, vec![("C".into(), 0.01), ("Fe".into(), 0.99)]);

        // Phases come from the Field block, sorted, with fractions already 0..1.
        assert_eq!(r.phases.len(), 2);
        assert_eq!(r.phases[0].name, "ferrite");
        assert_relative_eq!(r.phases[0].volume_fraction, 0.85, max_relative = 1e-9);
        assert_relative_eq!(
            r.phases[0].properties.get(P::YoungsModulus).unwrap(),
            1.95e11,
            max_relative = 1e-9
        );
        assert_eq!(r.phases[1].properties.get(P::HardnessBrinell), Some(200.0));
    }

    #[test]
    fn both_schemas_agree_on_the_same_material() {
        // The whole point of canonicalisation: two spellings, one answer.
        let nested = ingest(
            &json!({"PhysicalProperties": {"Density_g_cm3": 7.87, "ThermalConductivity_W_mK": 51.9}}),
            "a",
        )
        .unwrap();
        let flat = ingest(
            &json!({"Properties": {"Density_kgm3": 7870.0, "ThermalConductivity_WmK": 51.9}}),
            "b",
        )
        .unwrap();
        assert_relative_eq!(
            nested.bulk.get(P::Density).unwrap(),
            flat.bulk.get(P::Density).unwrap(),
            max_relative = 1e-9
        );
        assert_eq!(
            nested.bulk.get(P::ThermalConductivity),
            flat.bulk.get(P::ThermalConductivity)
        );
    }

    #[test]
    fn element_root_scalars_are_ingested() {
        let v = json!({
            "name": "Hydrogen", "symbol": "H",
            "atomic_number": 1, "atomic_mass": 1.008,
            "density": 8.988e-05, "melting_point": 14.01, "boiling_point": 20.28,
            "electronegativity": 2.2, "atomic_radius": 53,
            "ionization_energy": 13.598, "electron_affinity": 72.8,
            "group": 1, "period": 1, "valence_electrons": 1
        });
        let r = ingest(&v, "H").unwrap();
        assert_eq!(r.name, "Hydrogen");
        assert_relative_eq!(
            r.bulk.get(P::Density).unwrap(),
            0.08988,
            max_relative = 1e-9
        );
        assert_eq!(r.bulk.get(P::MeltingPoint), Some(14.01));
        assert_eq!(r.bulk.get(P::Electronegativity), Some(2.2));
        assert_relative_eq!(
            r.bulk.get(P::AtomicRadius).unwrap(),
            53e-12,
            max_relative = 1e-9
        );
        // `group` and `period` are descriptive, not properties -- and because
        // they sit at the root they must not be reported as unknown.
        assert!(
            r.is_fully_ingested(),
            "unknown: {:?}",
            r.provenance.unknown_keys
        );
    }

    #[test]
    fn microstructure_block_is_read() {
        let v = json!({
            "Microstructure": {
                "GrainStructure": {
                    "AverageGrainSize_um": 50.0, "GrainSizeStdDev": 0.35,
                    "ASTMGrainSizeNumber": 5, "VoronoiSeedDensity_per_mm2": 400.0,
                    "GrainBoundaryWidth_nm": 0.5, "GrainAspectRatio": 1.0,
                    "TwinDensity_per_mm": 5.0
                },
                "PhaseDistribution": {
                    "NoiseType": "Simplex", "NoiseScale": 0.08,
                    "NoiseOctaves": 3, "NoisePersistence": 0.5, "PhaseThreshold": 0.95
                },
                "Defects": {
                    "VacancyConcentration": 1e-05,
                    "DislocationDensity_per_m2": 1e12,
                    "StressField": {"Type": "VonMises", "MaxStress_MPa": 117.6}
                },
                "Inclusions": {"Density_per_mm3": 50, "AverageSize_um": 5.0}
            }
        });
        let m = ingest(&v, "x").unwrap().micro.unwrap();
        assert_relative_eq!(m.grains.average_size_m, 50e-6, max_relative = 1e-9);
        assert_relative_eq!(m.grains.boundary_width_m, 0.5e-9, max_relative = 1e-9);
        assert_relative_eq!(
            m.grains.twin_density_per_m.unwrap(),
            5000.0,
            max_relative = 1e-9
        );
        assert_eq!(m.phase_distribution.noise, NoiseKind::Simplex);
        assert_relative_eq!(m.phase_distribution.scale_rel, 0.08, max_relative = 1e-9);
        assert_relative_eq!(
            m.defects.dislocation_density_per_m2,
            1e12,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            m.defects.residual_stress_pa.unwrap(),
            117.6e6,
            max_relative = 1e-9
        );
        assert_relative_eq!(m.inclusions.density_per_m3, 5e10, max_relative = 1e-9);
        assert_relative_eq!(m.inclusions.average_size_m, 5e-6, max_relative = 1e-9);
    }

    #[test]
    fn areal_seed_density_matches_the_quoted_grain_size() {
        // 400 seeds/mm^2 and 50 um grains describe the same microstructure;
        // the n_a^(3/2) conversion must reconcile them to within a factor of 2.
        let v = json!({"Microstructure": {"GrainStructure": {
            "AverageGrainSize_um": 50.0, "VoronoiSeedDensity_per_mm2": 400.0
        }}});
        let m = ingest(&v, "x").unwrap().micro.unwrap();
        let from_size = 1.0 / (50e-6_f64).powi(3);
        let ratio = m.grains.seed_density_per_m3 / from_size;
        assert!(ratio > 0.5 && ratio < 2.0, "seed density ratio {ratio}");
    }

    #[test]
    fn lattice_and_texture_are_read_in_si() {
        let v = json!({
            "LatticeProperties": {
                "PrimaryStructure": "BCC",
                "LatticeParameters": {"a_pm": 286.6, "b_pm": 286.6, "c_pm": 286.6,
                                      "alpha_deg": 90, "beta_deg": 90, "gamma_deg": 90},
                "AtomicPackingFactor": 0.68, "CoordinationNumber": 8, "AtomsPerUnitCell": 2,
                "BurgersVector_pm": 286.3
            },
            "CrystallographicOrientation": {
                "PreferredOrientation": true, "TextureType": "Rolling", "TextureStrength_mrd": 2.5
            }
        });
        let r = ingest(&v, "x").unwrap();
        let l = r.lattice.unwrap();
        assert_eq!(l.system, CrystalSystem::Bcc);
        assert_relative_eq!(l.abc_m[0], 286.6e-12, max_relative = 1e-9);
        assert_relative_eq!(
            l.angles_rad[0],
            std::f64::consts::FRAC_PI_2,
            max_relative = 1e-12
        );
        assert_relative_eq!(l.burgers_vector_m.unwrap(), 286.3e-12, max_relative = 1e-9);
        // BCC cleaves -- this is what makes fracture planar.
        assert!(r.crystal_system().is_cleavable());

        let t = r.texture.unwrap();
        assert!(t.preferred);
        assert_relative_eq!(t.strength_mrd, 2.5, max_relative = 1e-9);
        assert!(t.axis.is_some(), "a rolling texture has an axis");
    }

    #[test]
    fn random_texture_has_no_axis() {
        let v = json!({"CrystallographicOrientation":
            {"PreferredOrientation": true, "TextureType": "Random", "TextureStrength_mrd": 1.0}});
        assert!(ingest(&v, "x").unwrap().texture.unwrap().axis.is_none());
    }

    #[test]
    fn phase_composition_percentages_become_fractions() {
        let v = json!({"PhaseComposition": {"Phases": [
            {"Name": "Ferrite", "Symbol": "alpha", "Structure": "BCC",
             "VolumePercent": 98, "Magnetic": true, "Hardness_HV": 196.0}
        ]}});
        let r = ingest(&v, "x").unwrap();
        assert_eq!(r.phases.len(), 1);
        assert_relative_eq!(r.phases[0].volume_fraction, 0.98, max_relative = 1e-9);
        assert_eq!(r.phases[0].structure, CrystalSystem::Bcc);
        assert!(r.phases[0].magnetic);
        // A phase's hardness must land on the phase, never on the bulk table.
        assert_eq!(r.phases[0].properties.get(P::HardnessVickers), Some(196.0));
        assert!(!r.bulk.has(P::HardnessVickers));
    }

    #[test]
    fn components_ranges_become_midpoint_fractions() {
        let v = json!({"Components": [
            {"Element": "Fe", "MinPercent": 92.4, "MaxPercent": 94.4},
            {"Element": "Cr", "MinPercent": 5.6,  "MaxPercent": 7.6}
        ]});
        let r = ingest(&v, "x").unwrap();
        assert_eq!(r.composition.len(), 2);
        // Sorted by symbol; Cr midpoint is 6.6 % -> 0.066.
        assert_eq!(r.composition[0].0, "Cr");
        assert_relative_eq!(r.composition[0].1, 0.066, max_relative = 1e-9);
        assert_relative_eq!(r.composition[1].1, 0.934, max_relative = 1e-9);
    }

    #[test]
    fn atom_counts_are_normalised_to_fractions() {
        let r = ingest(&json!({"Composition": {"H": 2, "O": 1}}), "H2O").unwrap();
        assert_relative_eq!(r.composition[0].1, 2.0 / 3.0, max_relative = 1e-9);
        assert_relative_eq!(r.composition[1].1, 1.0 / 3.0, max_relative = 1e-9);
    }

    #[test]
    fn color_is_parsed_and_linearised() {
        let r = ingest(&json!({"Color": "#C0C0C0"}), "x").unwrap();
        // 0xC0/255 = 0.753 sRGB -> ~0.527 linear. Definitely not equal to the
        // sRGB value, which is the bug this guards against.
        assert!(
            r.color_srgb[0] > 0.5 && r.color_srgb[0] < 0.55,
            "{:?}",
            r.color_srgb
        );
        assert_eq!(r.color_srgb[0], r.color_srgb[1]);

        // Malformed values fall back rather than panicking.
        let bad = ingest(&json!({"Color": "not-a-color"}), "x").unwrap();
        assert_eq!(bad.color_srgb, [0.75, 0.75, 0.78]);
    }

    #[test]
    fn unknown_keys_inside_a_block_are_reported() {
        let v = json!({"MechanicalProperties": {"SomeNewProperty_MPa": 42.0}});
        let r = ingest(&v, "x").unwrap();
        assert!(!r.is_fully_ingested());
        assert_eq!(r.provenance.unknown_keys, vec!["somenewpropertympa"]);
    }

    #[test]
    fn ignored_keys_are_not_reported_as_unknown() {
        let v = json!({"ElasticProperties": {"YoungsModulus_0_GPa": 130.0}});
        let r = ingest(&v, "x").unwrap();
        assert!(r.is_fully_ingested());
        // ...and must not have been mistaken for the isotropic bulk modulus.
        assert!(!r.bulk.has(P::YoungsModulus));
    }

    #[test]
    fn derived_properties_are_filled_during_ingest() {
        let v = json!({"Properties": {
            "ElectricalResistivity_ohmm": 1.43e-7,
            "YoungsModulus_GPa": 200, "PoissonsRatio": 0.29
        }});
        let r = ingest(&v, "x").unwrap();
        assert!(r.bulk.has(P::ElectricalConductivity));
        assert_relative_eq!(
            r.bulk.get(P::ElectricalConductivity).unwrap(),
            1.0 / 1.43e-7,
            max_relative = 1e-9
        );
        assert!(r.bulk.has(P::ShearModulus));
        assert_eq!(r.bulk.source(P::ShearModulus), Source::Derived);
    }

    #[test]
    fn non_object_root_is_rejected() {
        assert!(ingest(&json!([1, 2, 3]), "x").is_err());
        assert!(ingest(&json!("text"), "x").is_err());
    }

    #[test]
    fn ingest_is_deterministic() {
        let v = json!({
            "Properties": {"Density_kgm3": 7870, "YoungsModulus_GPa": 200},
            "Field": {"phases": {"b": {"fraction": 0.2}, "a": {"fraction": 0.8}}}
        });
        let a = ingest(&v, "x").unwrap();
        let b = ingest(&v, "x").unwrap();
        assert_eq!(a, b);
        // Phase order must not depend on JSON key order.
        assert_eq!(a.phases[0].name, "a");
    }

    #[test]
    fn name_falls_back_to_the_hint() {
        let r = ingest(&json!({"Properties": {"Density_kgm3": 1.0}}), "FileStem").unwrap();
        assert_eq!(r.name, "FileStem");
    }

    #[test]
    fn per_mineral_properties_do_not_leak_into_bulk() {
        // Shaped like active/materials/Granite_Westerly.json, where the
        // mineral list precedes the bulk block. First-writer-wins made the
        // quartz density (2650) the bulk density instead of 2640.
        let v = json!({
            "MineralComposition": {
                "Quartz":   {"VolumeFraction": 0.3, "properties": {"Density_kg_m3": 2650}},
                "Feldspar": {"VolumeFraction": 0.6, "properties": {"Density_kg_m3": 2560}}
            },
            "PhysicalProperties": {"Density_kg_m3": 2640}
        });
        let r = ingest(&v, "Granite").unwrap();
        assert_eq!(r.bulk.get(P::Density), Some(2640.0));
    }
}
