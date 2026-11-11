//! S3a: the ten new scalar properties, their sheet keys and the sound-speed rules.

use std::path::{Path, PathBuf};

use periodica_mat::canon::lookup;
use periodica_mat::derive::{fill_derived_with, DeriveGates};
use periodica_mat::{canon, fill_derived, ingest, jsonc, PropertyId as P, PropertyTable, Source};

const NEW_IDS: [(P, &str); 10] = [
    (P::LatentHeatFusion, "J/kg"),
    (P::LatentHeatVaporization, "J/kg"),
    (P::LatentHeatSublimation, "J/kg"),
    (P::RelativePermeability, "1"),
    (P::SoundSpeedLongitudinal, "m/s"),
    (P::SoundSpeedShear, "m/s"),
    (P::MechanicalLossFactor, "1"),
    (P::AcousticAttenuationOverF2, "Np s^2/m"),
    (P::DynamicViscosity, "Pa s"),
    (P::ViscosityReferenceTemperature, "K"),
];

fn table(pairs: &[(P, f64)]) -> PropertyTable {
    let mut t = PropertyTable::new();
    for (p, v) in pairs {
        t.set(*p, *v, Source::Datasheet);
    }
    t
}

fn key(k: &str) -> Option<(P, f64)> {
    lookup(&canon(k)).map(|m| (m.property, m.conv.scale))
}

#[test]
fn ids_follow_the_last_old_id_in_order_with_units() {
    let base = P::CharacteristicCompressiveStrength.index();
    for (i, (p, unit)) in NEW_IDS.iter().enumerate() {
        assert_eq!(p.index(), base + 1 + i, "{p:?}");
        assert_eq!(p.si_unit(), *unit, "{p:?}");
        assert_eq!(P::from_name(p.name()), Some(*p));
    }
    // P2 appended `WaterContent` after the ten.
    assert_eq!(P::WaterContent.index(), base + 1 + NEW_IDS.len());
    assert_eq!(P::COUNT, base + 2 + NEW_IDS.len());
}

#[test]
fn sheet_keys_map_with_the_right_scale() {
    let cases: &[(&str, P, f64)] = &[
        ("LatentHeatFusion_J_kg", P::LatentHeatFusion, 1.0),
        ("LatentHeatFusion_kJ_kg", P::LatentHeatFusion, 1e3),
        (
            "LatentHeatVaporization_J_kg",
            P::LatentHeatVaporization,
            1.0,
        ),
        (
            "LatentHeatVaporization_kJ_kg",
            P::LatentHeatVaporization,
            1e3,
        ),
        (
            "LatentHeatVaporisation_J_kg",
            P::LatentHeatVaporization,
            1.0,
        ),
        (
            "LatentHeatVaporisation_kJ_kg",
            P::LatentHeatVaporization,
            1e3,
        ),
        ("LatentHeatSublimation_J_kg", P::LatentHeatSublimation, 1.0),
        ("LatentHeatSublimation_kJ_kg", P::LatentHeatSublimation, 1e3),
        ("RelativePermeability", P::RelativePermeability, 1.0),
        ("SoundSpeedLongitudinal_m_s", P::SoundSpeedLongitudinal, 1.0),
        ("SoundSpeedShear_m_s", P::SoundSpeedShear, 1.0),
        ("SoundSpeed_m_s", P::SoundSpeedLongitudinal, 1.0),
        ("MechanicalLossFactor", P::MechanicalLossFactor, 1.0),
        (
            "AcousticAttenuationOverF2_Np_s2_m",
            P::AcousticAttenuationOverF2,
            1.0,
        ),
        ("DynamicViscosity_Pa_s", P::DynamicViscosity, 1.0),
        ("DynamicViscosity_mPa_s", P::DynamicViscosity, 1e-3),
        ("DynamicViscosity_cP", P::DynamicViscosity, 1e-3),
        (
            "ViscosityReferenceTemperature_K",
            P::ViscosityReferenceTemperature,
            1.0,
        ),
    ];
    for (k, p, s) in cases {
        assert_eq!(key(k), Some((*p, *s)), "{k}");
    }
}

#[test]
fn ambiguous_and_molar_keys_map_to_nothing() {
    for k in [
        "LossTangent",
        "SpeedOfSound",
        "LatentHeatFusion_kJ_mol",
        "LatentHeatVaporization_kJ_mol",
    ] {
        assert!(key(k).is_none(), "{k}");
    }
}

#[test]
fn steel_sound_speeds_follow_the_unbounded_medium_formulas() {
    let (e, nu, rho) = (200e9_f64, 0.29_f64, 7850.0_f64);
    let g = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    let vl = ((k + 4.0 * g / 3.0) / rho).sqrt();
    let vs = (g / rho).sqrt();
    let mut t = table(&[
        (P::YoungsModulus, e),
        (P::PoissonsRatio, nu),
        (P::Density, rho),
    ]);
    fill_derived(&mut t);
    let l = t.get(P::SoundSpeedLongitudinal).unwrap();
    let s = t.get(P::SoundSpeedShear).unwrap();
    assert!((l / vl - 1.0).abs() < 0.005 && (l / 5778.0 - 1.0).abs() < 0.005);
    assert!((s / vs - 1.0).abs() < 0.005 && (s / 3142.0 - 1.0).abs() < 0.005);
    assert!((l / (e / rho).sqrt() - 1.0).abs() > 0.10);
    assert_eq!(t.source(P::SoundSpeedLongitudinal), Source::Estimated);
    assert_eq!(t.source(P::SoundSpeedShear), Source::Estimated);
}

#[test]
fn an_estimated_input_derives_no_sound_speed() {
    let mut t = table(&[
        (P::YoungsModulus, 200e9),
        (P::PoissonsRatio, 0.29),
        (P::Density, 7850.0),
    ]);
    t.set(P::Density, 7850.0, Source::Estimated);
    fill_derived(&mut t);
    assert_eq!(t.get(P::SoundSpeedLongitudinal), None);
    assert_eq!(t.get(P::SoundSpeedShear), None);
}

#[test]
fn an_estimated_modulus_derives_no_sound_speed() {
    let mut t = table(&[(P::PoissonsRatio, 0.29), (P::Density, 7850.0)]);
    t.set(P::YoungsModulus, 200e9, Source::Estimated);
    fill_derived(&mut t);
    assert_eq!(t.get(P::SoundSpeedLongitudinal), None);
    assert_eq!(t.get(P::SoundSpeedShear), None);
}

fn steel_with(extra: &[(P, f64)]) -> PropertyTable {
    let mut base = vec![
        (P::YoungsModulus, 200e9),
        (P::PoissonsRatio, 0.29),
        (P::Density, 7850.0),
    ];
    base.extend_from_slice(extra);
    let mut t = table(&base);
    fill_derived(&mut t);
    t
}

#[test]
fn pore_fluid_closes_the_solid_longitudinal_speed_only() {
    for extra in [(P::WaterContent, 0.7), (P::Porosity, 0.1)] {
        let t = steel_with(&[extra]);
        assert!(t.get(P::SoundSpeedShear).is_some(), "{extra:?}");
        assert_eq!(t.get(P::SoundSpeedLongitudinal), None, "{extra:?}");
    }
    let t = steel_with(&[(P::WaterContent, 0.0)]);
    assert!(t.get(P::SoundSpeedShear).is_some());
    assert!(t.get(P::SoundSpeedLongitudinal).is_some());
}

#[test]
fn saturated_fluid_keeps_the_fluid_form() {
    let mut t = table(&[
        (P::BulkModulus, 2.2e9),
        (P::Density, 998.0),
        (P::DynamicViscosity, 1.0e-3),
        (P::WaterContent, 1.0),
    ]);
    fill_derived(&mut t);
    let v = t.get(P::SoundSpeedLongitudinal).unwrap();
    assert!((v / (2.2e9_f64 / 998.0).sqrt() - 1.0).abs() < 1e-12);
    assert_eq!(t.source(P::SoundSpeedLongitudinal), Source::Estimated);
    assert_eq!(t.get(P::SoundSpeedShear), None);
}

#[test]
fn water_has_one_estimated_longitudinal_speed_and_no_shear() {
    let mut t = table(&[
        (P::BulkModulus, 2.2e9),
        (P::Density, 998.0),
        (P::DynamicViscosity, 1.0e-3),
        (P::ViscosityReferenceTemperature, 293.15),
    ]);
    fill_derived(&mut t);
    let v = t.get(P::SoundSpeedLongitudinal).unwrap();
    assert!((v / (2.2e9_f64 / 998.0).sqrt() - 1.0).abs() < 1e-9);
    assert!((v / 1485.0 - 1.0).abs() < 0.005, "{v}");
    assert_eq!(t.source(P::SoundSpeedLongitudinal), Source::Estimated);
    assert_eq!(t.get(P::SoundSpeedShear), None);
}

#[test]
fn bulk_modulus_and_density_alone_derive_no_sound_speed() {
    let mut t = table(&[(P::BulkModulus, 2.2e9), (P::Density, 998.0)]);
    fill_derived(&mut t);
    assert_eq!(t.get(P::SoundSpeedLongitudinal), None);
    assert_eq!(t.get(P::SoundSpeedShear), None);
}

#[test]
fn a_measured_longitudinal_speed_survives() {
    let mut t = table(&[
        (P::YoungsModulus, 200e9),
        (P::PoissonsRatio, 0.29),
        (P::Density, 7850.0),
        (P::SoundSpeedLongitudinal, 5900.0),
    ]);
    fill_derived(&mut t);
    assert_eq!(t.get(P::SoundSpeedLongitudinal), Some(5900.0));
    assert_eq!(t.source(P::SoundSpeedLongitudinal), Source::Datasheet);
}

#[test]
fn a_closed_elastic_gate_derives_no_sound_speed() {
    let mut t = table(&[
        (P::BulkModulus, 140e9),
        (P::ShearModulus, 80e9),
        (P::Density, 7850.0),
    ]);
    let gates = DeriveGates {
        isotropic_elastic: false,
    };
    fill_derived_with(&mut t, gates);
    assert_eq!(t.get(P::SoundSpeedLongitudinal), None);
    assert_eq!(t.get(P::SoundSpeedShear), None);
}

#[test]
fn viscosity_and_its_reference_temperature_ingest_to_si() {
    let v = serde_json::json!({
        "Name": "Oil",
        "Properties": {
            "DynamicViscosity_mPa_s": 50.0,
            "ViscosityReferenceTemperature_K": 293.15
        }
    });
    let r = ingest(&v, "Oil").unwrap();
    let eta = r.bulk.get(P::DynamicViscosity).unwrap();
    assert!((eta - 0.05).abs() < 1e-12);
    assert_eq!(r.bulk.get(P::ViscosityReferenceTemperature), Some(293.15));
}

fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_json(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(p);
        }
    }
}

#[test]
fn every_record_with_viscosity_states_its_reference_temperature() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
    let mut files = Vec::new();
    for r in ["active", "derived", "defaults", "reference"] {
        collect_json(&root.join(r), &mut files);
    }
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = jsonc::from_str(&text) else {
            continue;
        };
        let Ok(r) = ingest(&value, "sheet") else {
            continue;
        };
        if r.bulk.has(P::DynamicViscosity) {
            assert!(
                r.bulk.has(P::ViscosityReferenceTemperature),
                "{}",
                path.display()
            );
        }
    }
}
