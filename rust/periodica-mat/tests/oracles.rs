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

#[test]
fn bundled_overrides_are_empty_until_a_measured_pair_is_added() {
    let t = periodica_mat::PairTable::bundled().unwrap();
    assert!(
        t.rows().is_empty(),
        "family rows must not return to the bundled table"
    );
}
