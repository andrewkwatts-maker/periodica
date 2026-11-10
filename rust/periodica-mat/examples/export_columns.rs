//! Export every sheet of the corpus as property columns.
//!
//! Run: `cargo run -p periodica-mat --example export_columns -- <out_dir>`
//! Exports every property that has at least one value in some row.

use std::path::{Path, PathBuf};

use periodica_mat::export::{columns, has_density, load_corpus, write_dir};
use periodica_mat::PropertyId;

fn main() {
    let Some(out) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: export_columns <out_dir>");
        std::process::exit(2);
    };
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data");
    let corpus = load_corpus(&data, has_density);
    assert!(!corpus.records.is_empty(), "no sheets ingested");
    let props: Vec<PropertyId> = PropertyId::ALL
        .iter()
        .copied()
        .filter(|&p| corpus.records.iter().any(|(_, r)| r.bulk.has(p)))
        .collect();
    let mut set = columns(&corpus.records, &props).expect("columns");
    set.shadowed = corpus.shadowed.clone();
    let m = write_dir(&set, &out).expect("write export");
    println!(
        "{} rows, {} properties, {} shadowed, {} skipped",
        m.rows,
        m.properties.len(),
        corpus.shadowed.len(),
        corpus.skipped.len()
    );
    let mut added: std::collections::BTreeMap<&str, usize> = Default::default();
    for ((_, r), f) in corpus.records.iter().zip(&corpus.folders) {
        if !periodica_mat::is_material_like(r) {
            *added.entry(f.as_str()).or_default() += 1;
        }
    }
    println!("rows beyond is_material_like: {added:?}");
    for s in &corpus.skipped {
        println!("skipped {}: {}", s.path, s.reason);
    }
}
