//! Census of every bundled sheet that ingests as a material: volume-ready, or its missing keys.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use periodica_mat::census::volume_readiness;
use periodica_mat::{ingest, is_material_like, jsonc};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
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

fn load(path: &Path) -> Option<periodica_mat::record::MaterialRecord> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = jsonc::from_str(&text).ok()?;
    let stem = path.file_stem()?.to_str()?;
    ingest(&value, stem).ok()
}

#[test]
fn every_material_is_ready_or_lists_what_it_lacks() {
    let mut files = Vec::new();
    for r in ["active", "derived", "defaults", "reference"] {
        collect_json(&data_dir().join(r), &mut files);
    }
    files.sort();
    let (mut total, mut ready, mut ortho) = (0usize, 0usize, 0usize);
    let mut by_key: BTreeMap<String, usize> = BTreeMap::new();
    for rec in files
        .iter()
        .filter_map(|p| load(p))
        .filter(is_material_like)
    {
        let r = volume_readiness(&rec);
        assert!(
            r.volume_ready || !r.missing.is_empty(),
            "{}: not ready yet nothing missing",
            rec.name
        );
        assert_eq!(r.volume_ready, r.missing.is_empty(), "{}", rec.name);
        total += 1;
        ready += usize::from(r.volume_ready);
        ortho += usize::from(r.orthotropic);
        for k in r.missing {
            *by_key.entry(k).or_default() += 1;
        }
    }
    println!("census: {total} materials, {ready} volume-ready, {ortho} orthotropic");
    for k in ["micro.grains", "phases", "bulk.YoungsModulus"] {
        println!(
            "  totals missing {k}: {}",
            by_key.get(k).copied().unwrap_or(0)
        );
    }
    let mut keys: Vec<_> = by_key.into_iter().collect();
    keys.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (k, n) in keys {
        println!("  missing {k}: {n}");
    }
    assert!(total > 0);
}

#[test]
fn steel_1018_has_grains_from_the_field_block() {
    let path = data_dir().join("derived/alloys/Steel-1018.json");
    let rec = load(&path).expect("Steel-1018 ingests");
    let micro = rec.micro.expect("micro.grains from Field.grain_density");
    let grains = micro.grains.expect("grains from Field.grain_density");
    assert_eq!(grains.seed_density_per_m3, 1.0e15);
    assert!((grains.average_size_m - 1.0e-5).abs() < 1e-9);
    assert!(!rec.phases.is_empty());
}

#[test]
fn a_sheet_without_grain_data_has_no_grains() {
    let v = serde_json::json!({"Name": "x", "Properties": {"Density_kgm3": 1000, "YoungsModulus_GPa": 1.0}});
    let rec = ingest(&v, "x").expect("ingests");
    assert!(rec.micro.and_then(|m| m.grains).is_none());
}

#[test]
fn an_orthotropic_sheet_without_bulk_modulus_is_not_missing_it() {
    let v = serde_json::json!({
        "Name": "x",
        "Properties": {"Density_kgm3": 1500},
        "Field": {
            "axial": {"YoungsModulus_GPa": 130.0},
            "transverse": {"YoungsModulus_GPa": 9.0},
            "fiber_direction": [1.0, 0.0, 0.0]
        }
    });
    let rec = ingest(&v, "x").expect("ingests");
    assert!(rec.orthotropy.is_some());
    let r = volume_readiness(&rec);
    assert!(
        !r.missing.contains(&"bulk.YoungsModulus".to_string()),
        "{:?}",
        r.missing
    );
    assert!(r.missing.iter().any(|k| k.starts_with("orthotropy.")));
}
