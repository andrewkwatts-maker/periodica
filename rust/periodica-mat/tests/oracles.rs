//! Published interface tables are test answers, not engine inputs.

use std::path::Path;

fn oracle_rows() -> Vec<serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracles/pairs.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    v["rows"].as_array().unwrap().clone()
}

#[test]
fn oracle_rows_cite_sources_and_keep_the_moved_tables() {
    let rows = oracle_rows();
    assert_eq!(rows.len(), 8);
    for r in &rows {
        assert!(!r["source"].as_str().unwrap_or("").trim().is_empty(), "{r}");
    }
    let steel = rows
        .iter()
        .find(|r| r["a"] == "Steel" && r["b"] == "Steel")
        .unwrap();
    assert_eq!(steel["mu_k"], 0.42);
}

fn unordered<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[test]
fn bundled_rows_are_cited_record_ids() {
    use periodica_mat::export::load_corpus;
    use periodica_mat::Law;
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
    let corpus = load_corpus(&data, |_| true);
    let names: Vec<&str> = corpus.records.iter().map(|(_, r)| r.name.as_str()).collect();
    let t = periodica_mat::PairTable::bundled().unwrap();
    assert!(!t.rows().is_empty());
    for r in t.rows() {
        assert!(names.contains(&r.a.as_str()), "not a record id: {}", r.a);
        assert!(names.contains(&r.b.as_str()), "not a record id: {}", r.b);
        assert!(!r.source.trim().is_empty(), "{r:?}");
    }
    let row = t.lookup("Concrete-C30", "Concrete-C30").unwrap();
    assert!(matches!(
        row.law,
        Law::Coulomb { mu_s, mu_k: None, .. } if mu_s == 1.4
    ));
    for o in oracle_rows() {
        let (oa, ob) = (o["a"].as_str().unwrap(), o["b"].as_str().unwrap());
        for r in t.rows() {
            let same_key = unordered(oa, ob) == unordered(&r.a, &r.b);
            let same_law = o["law"] == "Coulomb" && matches!(r.law, Law::Coulomb { .. });
            assert!(!(same_key && same_law), "duplicates oracle row {o}");
        }
    }
}
