//! F2d: a mix of one part is the part; E and nu are the primary pair; stored
//! elastic mismatch is measured, not judged.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use periodica_mat::{
    elastic_mismatch, ingest, ingest_stated, jsonc, mix_tagged, MaterialRecord, MixError,
    PropertyId as P, Source,
};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
}

fn load_with(
    path: &Path,
    f: fn(&serde_json::Value, &str) -> Result<MaterialRecord, periodica_mat::IngestError>,
) -> Option<MaterialRecord> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = jsonc::from_str(&text).ok()?;
    f(&value, path.file_stem()?.to_str()?).ok()
}

fn load(path: &Path) -> Option<MaterialRecord> {
    load_with(path, ingest)
}

fn load_stated(path: &Path) -> Option<MaterialRecord> {
    load_with(path, ingest_stated)
}

fn steel() -> MaterialRecord {
    load(&data_dir().join("derived/alloys/Steel-1018.json")).expect("Steel-1018 ingests")
}

fn rec(name: &str, props: &[(P, f64)]) -> MaterialRecord {
    let mut r = MaterialRecord {
        name: name.to_string(),
        ..MaterialRecord::default()
    };
    for (p, v) in props {
        r.bulk.set(*p, *v, Source::Datasheet);
    }
    r
}

fn assert_returns_stored(rec: &MaterialRecord, entries: &[(&MaterialRecord, f64)]) {
    for (p, v, s) in rec.bulk.iter() {
        let (mv, ms) = mix_tagged(entries, p).unwrap();
        assert_eq!(mv.to_bits(), v.to_bits(), "{p:?}");
        assert_eq!(ms, s, "{p:?}");
    }
}

#[test]
fn one_part_at_fraction_one_is_the_stored_record() {
    let s = steel();
    assert_returns_stored(&s, &[(&s, 1.0)]);
    assert_eq!(s.bulk.get(P::YoungsModulus), Some(200e9));
    assert_eq!(s.bulk.get(P::PoissonsRatio), Some(0.29));
    assert_eq!(s.bulk.get(P::ShearModulus), Some(79.3e9));
}

#[test]
fn the_same_record_twice_or_a_clone_is_still_the_record() {
    let s = steel();
    let c = s.clone();
    assert_returns_stored(&s, &[(&s, 0.5), (&s, 0.5)]);
    assert_returns_stored(&s, &[(&s, 0.3), (&c, 0.7)]);
}

#[test]
fn missing_and_unmixable_properties() {
    let s = steel();
    let miss = mix_tagged(&[(&s, 1.0)], P::RefractiveIndexN);
    assert!(matches!(miss, Err(MixError::Missing { .. })), "{miss:?}");
    let a = rec("a", &[(P::HardnessVickers, 150.0)]);
    let b = rec("b", &[(P::HardnessVickers, 200.0)]);
    let one = mix_tagged(&[(&a, 1.0)], P::HardnessVickers).unwrap();
    assert_eq!(one, (150.0, Source::Datasheet));
    let two = mix_tagged(&[(&a, 0.5), (&b, 0.5)], P::HardnessVickers);
    assert_eq!(two, Err(MixError::NotMixable(P::HardnessVickers)));
}

#[test]
fn the_mix_is_continuous_as_one_part_tends_to_one() {
    let s = steel();
    let al = rec(
        "al",
        &[
            (P::YoungsModulus, 69e9),
            (P::PoissonsRatio, 0.33),
            (P::Density, 2700.0),
        ],
    );
    let entries = [(&s, 1.0 - 1e-6), (&al, 1e-6)];
    let (e, _) = mix_tagged(&entries, P::YoungsModulus).unwrap();
    let (nu, _) = mix_tagged(&entries, P::PoissonsRatio).unwrap();
    assert!((e - 200e9).abs() / 200e9 < 1e-5, "{e}");
    assert!((nu - 0.29).abs() < 1e-5, "{nu}");
}

#[test]
fn two_different_parts_keep_their_tags() {
    let s = steel();
    let al = rec(
        "al",
        &[
            (P::YoungsModulus, 69e9),
            (P::PoissonsRatio, 0.33),
            (P::Density, 2700.0),
        ],
    );
    let entries = [(&s, 0.5), (&al, 0.5)];
    assert_eq!(
        mix_tagged(&entries, P::YoungsModulus).unwrap().1,
        Source::Estimated
    );
    assert_eq!(mix_tagged(&entries, P::Density).unwrap().1, Source::Derived);
}

#[test]
fn a_part_with_no_nu_uses_its_own_k_and_g() {
    let props = [(P::BulkModulus, 160e9), (P::ShearModulus, 79e9)];
    let dens = (P::Density, 7850.0);
    let a = rec("a", &[props[0], props[1], dens]);
    let b = rec("b", &[props[0], props[1], dens]);
    let (e, _) = mix_tagged(&[(&a, 0.5), (&b, 0.5)], P::YoungsModulus).unwrap();
    let want = 9.0 * 160e9 * 79e9 / (3.0 * 160e9 + 79e9);
    assert!((e - want).abs() / want < 1e-12, "{e} vs {want}");
}

#[test]
fn elastic_mismatch_measures_and_does_not_judge() {
    let s = steel();
    let m = elastic_mismatch(&s.bulk);
    let (e, nu, g): (f64, f64, f64) = (200e9, 0.29, 79.3e9);
    let want = (g - e / (2.0 * (1.0 + nu))).abs() / g;
    assert_eq!(m.shear, Some(want));
    assert!((want - 0.0225).abs() < 5e-4, "{want}");

    let k = e / (3.0 * (1.0 - 2.0 * nu));
    let ok = rec(
        "ok",
        &[
            (P::YoungsModulus, e),
            (P::PoissonsRatio, nu),
            (P::ShearModulus, e / (2.0 * (1.0 + nu))),
            (P::BulkModulus, k),
        ],
    );
    let m = elastic_mismatch(&ok.bulk);
    assert!(m.shear.unwrap() < 1e-12 && m.bulk.unwrap() < 1e-12, "{m:?}");

    let no_g = rec("no_g", &[(P::YoungsModulus, e), (P::PoissonsRatio, nu)]);
    assert_eq!(elastic_mismatch(&no_g.bulk).shear, None);

    let nu_half = rec(
        "nu_half",
        &[
            (P::YoungsModulus, e),
            (P::PoissonsRatio, 0.5),
            (P::ShearModulus, g),
            (P::BulkModulus, k),
        ],
    );
    let m = elastic_mismatch(&nu_half.bulk);
    assert_eq!((m.shear, m.bulk), (None, None));
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

fn pct(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_string(), |x| format!("{:.2}", x * 100.0))
}

fn corpus_lines() -> BTreeSet<String> {
    let root = data_dir();
    let mut files = Vec::new();
    for r in ["active", "derived", "defaults", "reference"] {
        collect_json(&root.join(r), &mut files);
    }
    let mut lines = BTreeSet::new();
    for path in files {
        let Some(rec) = load_stated(&path) else {
            continue;
        };
        let m = elastic_mismatch(&rec.bulk);
        if m.shear.is_none() && m.bulk.is_none() {
            continue;
        }
        let rel = path.strip_prefix(&root).unwrap().to_string_lossy();
        let rel = rel.replace('\\', "/");
        lines.insert(format!("{rel} {} {}", pct(m.shear), pct(m.bulk)));
    }
    lines
}

#[test]
fn stored_elastic_mismatch_of_the_corpus_is_on_record() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracles/elastic-mismatch.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let recorded: BTreeSet<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect();
    let computed = corpus_lines();
    let paste: Vec<&str> = computed.iter().map(String::as_str).collect();
    assert_eq!(
        recorded,
        computed,
        "elastic-mismatch.txt differs from the corpus; computed lines:\n{}\n",
        paste.join("\n")
    );
}

const STEEL: &str = "derived/alloys/Steel-1018.json";
const CONCRETE: &str = "derived/composites/Concrete-C30.json";

#[test]
fn ingest_stated_holds_only_what_the_sheet_states() {
    let path = data_dir().join(STEEL);
    let stated = load_stated(&path).unwrap();
    assert_eq!(stated.bulk.get(P::YoungsModulus), Some(200e9));
    assert_eq!(stated.bulk.get(P::PoissonsRatio), Some(0.29));
    assert_eq!(stated.bulk.get(P::ShearModulus), Some(79.3e9));
    assert_eq!(stated.bulk.get(P::BulkModulus), None);
    assert!(load(&path).unwrap().bulk.get(P::BulkModulus).is_some());
}

#[test]
fn ingest_keeps_every_stated_value_and_source() {
    let path = data_dir().join(STEEL);
    let (stated, full) = (load_stated(&path).unwrap(), load(&path).unwrap());
    for (p, v, s) in stated.bulk.iter() {
        assert_eq!(full.bulk.get(p), Some(v), "{p:?}");
        assert_eq!(full.bulk.source(p), s, "{p:?}");
    }
}

#[test]
fn concrete_c30_states_what_the_mismatch_sees() {
    let stated = load_stated(&data_dir().join(CONCRETE)).unwrap();
    let t = &stated.bulk;
    let got = [
        P::YoungsModulus,
        P::PoissonsRatio,
        P::ShearModulus,
        P::BulkModulus,
    ]
    .map(|p| (t.get(p), t.source(p)));
    println!("Concrete-C30 stated E, nu, G, K: {got:?}");
    let m = elastic_mismatch(t);
    println!("{m:?}");
    assert!(m.shear.is_none() && m.bulk.is_none(), "{m:?}");
}
