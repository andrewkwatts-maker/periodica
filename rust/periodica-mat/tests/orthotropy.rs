//! Off-axis modulus and Tsai-Hill strength of the T300/5208 laminate, and ingest of the bundled CFRP sheet.

use periodica_mat::{ingest, jsonc, Orthotropy};
use serde_json::json;

fn t300() -> Orthotropy {
    let v = json!({"Field": {
        "axial": {"YoungsModulus_GPa": 181.0, "ShearModulus_GPa": 7.17, "PoissonRatio": 0.28,
                  "TensileStrength_MPa": 1500.0, "ShearStrength_MPa": 68.0},
        "transverse": {"YoungsModulus_GPa": 10.3, "TensileStrength_MPa": 40.0},
        "fiber_direction": [1.0, 0.0, 0.0]}});
    let field = v["Field"].as_object().unwrap();
    Orthotropy::from_field(field).unwrap()
}

#[test]
fn modulus_follows_compliance_transformation() {
    let o = t300();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!((o.modulus_along([1.0, 0.0, 0.0]).unwrap() / 181.0e9 - 1.0).abs() < 1e-12);
    assert!((o.modulus_along([0.0, 0.0, 5.0]).unwrap() / 10.3e9 - 1.0).abs() < 1e-12);
    let e45 = o.modulus_along([h, h, 0.0]).unwrap() / 1.0e9;
    println!("E45 = {e45}");
    assert!((e45 / 16.74 - 1.0).abs() < 1e-3, "{e45}");
}

#[test]
fn strength_follows_tsai_hill() {
    let o = t300();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let x45 = o.tensile_along([h, h, 0.0]).unwrap() / 1.0e6;
    println!("X45 = {x45}");
    assert!((x45 / 68.9 - 1.0).abs() < 5e-3, "{x45}");
    assert!((o.tensile_along([1.0, 0.0, 0.0]).unwrap() / 1.5e9 - 1.0).abs() < 1e-12);
}

#[test]
fn zero_direction_is_an_error() {
    assert!(t300().modulus_along([0.0; 3]).is_err());
    assert!(t300().tensile_along([0.0; 3]).is_err());
}

#[test]
fn bundled_cfrp_names_missing_keys() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/periodica/data/derived/composites/CFRP.json");
    let text = std::fs::read_to_string(path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&jsonc::strip_comments(&text)).unwrap();
    let o = ingest(&value, "CFRP").unwrap().orthotropy.unwrap();
    assert_eq!(o.e[0], Some(230.0e9));
    let err = o.modulus_along([1.0, 1.0, 0.0]).unwrap_err();
    assert_eq!(err.missing, vec!["G12", "nu12"]);
    assert!(err.to_string().contains("G12"));
    assert_eq!(o.tensile_along([1.0, 0.0, 0.0]).unwrap_err().missing, vec!["S"]);
}
