//! Sheet coverage census by PlayTow domain.
//!
//! Run: `cargo run -p periodica-mat --example domain_census`
//! Writes `docs/dev/domain-census.json` (full per-sheet data, sorted keys) and
//! `docs/dev/domain-census.md` (the summary). Read-only for every sheet.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use periodica_mat::census::volume_readiness;
use periodica_mat::domains::DOMAINS;
use periodica_mat::record::MaterialRecord;
use periodica_mat::{ingest, is_material_like, jsonc, PropertyId, Source};
use serde::Serialize;
use serde_json::Value;

/// The eleven sheets PlayTow loads, most used first.
const PLAYTOW: &[&str] = &[
    "Steel-1018",
    "Concrete-C30",
    "Steel-B500B",
    "Concrete-C40",
    "Concrete-C25",
    "Aluminum-6061",
    "Magnet-N42",
    "Timber-C24",
    "Rubber-Natural",
    "Mortar-CementLime",
    "Brick-Clay",
];
/// Values that change with temperature, so a sheet should state the temperature they hold at.
const TEMP_DEPENDENT: &[PropertyId] = &[
    PropertyId::ThermalConductivity,
    PropertyId::ElectricalConductivity,
    PropertyId::ElectricalResistivity,
    PropertyId::SpecificHeat,
    PropertyId::YoungsModulus,
    PropertyId::ShearModulus,
    PropertyId::BulkModulus,
];
/// Lower-case key fragments that count as stating a reference temperature.
const REF_TEMP_KEYS: &[&str] = &[
    "referencetemp",
    "reference_temp",
    "reftemp",
    "ref_temp",
    "temperature_ref",
    "testtemp",
    "test_temp",
    "measurementtemp",
    "_at_",
    "at_20c",
    "at20c",
    "measuredat",
];
const WOOD_WORDS: &[&str] = &[
    "wood", "timber", "laminate", "fiber", "fibre", "cfrp", "gfrp",
];

#[derive(Serialize, Default)]
struct Sheet {
    root: String,
    folder: String,
    name: String,
    category: String,
    material_like: bool,
    /// Per domain: [datasheet, derived, estimated, user override] counts.
    cells: BTreeMap<String, [u32; 4]>,
    /// Property name -> source.
    props: BTreeMap<String, String>,
    unknown_keys: Vec<String>,
    volume_missing: Vec<String>,
    orthotropy_none_but_wood_like: bool,
    temp_dependent_without_reference_temp: u32,
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data")
}

fn collect_json(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for p in entries.flatten().map(|e| e.path()) {
        if p.is_dir() {
            collect_json(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(p);
        }
    }
}

fn has_ref_temp(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| {
            let k = k.to_lowercase();
            REF_TEMP_KEYS.iter().any(|f| k.contains(f)) || has_ref_temp(x)
        }),
        Value::Array(a) => a.iter().any(has_ref_temp),
        _ => false,
    }
}

fn source_slot(s: Source) -> Option<usize> {
    match s {
        Source::Datasheet => Some(0),
        Source::Derived => Some(1),
        Source::Estimated => Some(2),
        Source::UserOverride => Some(3),
        Source::Absent => None,
    }
}

fn cells_for(rec: &MaterialRecord) -> BTreeMap<String, [u32; 4]> {
    DOMAINS
        .iter()
        .map(|d| {
            let mut c = [0u32; 4];
            for &id in d.ids {
                if let Some(i) = source_slot(rec.bulk.source(id)) {
                    c[i] += 1;
                }
            }
            (d.name.to_string(), c)
        })
        .collect()
}

fn looks_woody(rec: &MaterialRecord) -> bool {
    let text = format!(
        "{} {} {}",
        rec.name,
        rec.category.as_deref().unwrap_or(""),
        rec.class.as_deref().unwrap_or("")
    )
    .to_lowercase();
    WOOD_WORDS.iter().any(|w| text.contains(w))
}

fn build_sheet(path: &Path, rec: &MaterialRecord, raw: &Value) -> Sheet {
    let rel = path.strip_prefix(data_dir()).unwrap_or(path);
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let readiness = volume_readiness(rec);
    let no_ref = !has_ref_temp(raw);
    let temp_n = TEMP_DEPENDENT.iter().filter(|&&p| rec.bulk.has(p)).count() as u32;
    Sheet {
        root: parts[0].clone(),
        folder: parts[..parts.len() - 1].join("/"),
        name: rec.name.clone(),
        category: rec.category.clone().unwrap_or_default(),
        material_like: is_material_like(rec),
        cells: cells_for(rec),
        props: rec
            .bulk
            .iter()
            .map(|(id, _, s)| (id.name().to_string(), format!("{s:?}")))
            .collect(),
        unknown_keys: rec.provenance.unknown_keys.clone(),
        volume_missing: readiness.missing,
        orthotropy_none_but_wood_like: rec.orthotropy.is_none() && looks_woody(rec),
        temp_dependent_without_reference_temp: if no_ref { temp_n } else { 0 },
    }
}

fn load_all() -> Vec<Sheet> {
    let mut files = Vec::new();
    for r in ["active", "derived", "defaults", "reference"] {
        collect_json(&data_dir().join(r), &mut files);
    }
    files.sort();
    let mut sheets = Vec::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(raw) = jsonc::from_str(&text) else {
            continue;
        };
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let Ok(rec) = ingest(&raw, stem) else {
            continue;
        };
        let mut s = build_sheet(&path, &rec, &raw);
        s.name = stem.to_string();
        if s.material_like || PLAYTOW.contains(&stem) {
            sheets.push(s);
        }
    }
    sheets
}

fn write_json(sheets: &[Sheet], out: &Path) {
    let text = serde_json::to_string_pretty(sheets).expect("serialise census");
    std::fs::write(out, text + "\n").expect("write json");
}

fn main() {
    let sheets = load_all();
    assert!(!sheets.is_empty(), "no sheets ingested");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/dev");
    std::fs::create_dir_all(&dir).expect("docs/dev");
    write_json(&sheets, &dir.join("domain-census.json"));
    std::fs::write(
        dir.join("domain-census.md"),
        domain_census_md::render(&sheets),
    )
    .expect("write md");
    println!("{} sheets", sheets.len());
}

mod domain_census_md {
    //! Markdown rendering for the census.
    use super::*;

    fn cell(c: &[u32; 4]) -> String {
        format!("{}/{}/{}", c[0] + c[3], c[1], c[2])
    }

    fn group_table(sheets: &[Sheet]) -> String {
        let mut groups: BTreeMap<(String, String), Vec<&Sheet>> = BTreeMap::new();
        for s in sheets {
            groups
                .entry((s.folder.clone(), s.category.clone()))
                .or_default()
                .push(s);
        }
        let names: Vec<&str> = DOMAINS.iter().map(|d| d.name).collect();
        let mut out = format!(
            "| folder | category | sheets | {} |\n|---|---|---|{}\n",
            names.join(" | "),
            "---|".repeat(names.len())
        );
        for ((folder, cat), g) in &groups {
            let cols: Vec<String> = names.iter().map(|n| domain_count(g, n)).collect();
            let cat = if cat.is_empty() { "-" } else { cat };
            out += &format!(
                "| {folder} | {cat} | {} | {} |\n",
                g.len(),
                cols.join(" | ")
            );
        }
        out
    }

    /// "with a datasheet value / only derived or estimated" for one domain.
    fn domain_count(g: &[&Sheet], dom: &str) -> String {
        let (mut ds, mut other) = (0, 0);
        for s in g {
            let c = s.cells[dom];
            if c[0] + c[3] > 0 {
                ds += 1;
            } else if c[1] + c[2] > 0 {
                other += 1;
            }
        }
        format!("{ds}/{other}")
    }

    fn playtow_table(sheets: &[Sheet]) -> String {
        let names: Vec<&str> = DOMAINS.iter().map(|d| d.name).collect();
        let mut out = format!(
            "| sheet | folder | {} |\n|---|---|{}\n",
            names.join(" | "),
            "---|".repeat(names.len())
        );
        for want in PLAYTOW {
            for s in sheets.iter().filter(|s| s.name == *want) {
                let cols: Vec<String> = names.iter().map(|n| cell(&s.cells[*n])).collect();
                out += &format!("| {want} | {} | {} |\n", s.folder, cols.join(" | "));
            }
        }
        out
    }

    fn gaps(sheets: &[Sheet]) -> Vec<String> {
        let mut g = vec![
            "acoustic: no PropertyId (speed of sound, loss) in any sheet; MP-PER-3".to_string(),
            "contact: no PropertyId for friction or restitution; only surface descriptors"
                .to_string(),
            "optical: none of the eleven PlayTow sheets carries n, k or emissivity".to_string(),
        ];
        for want in PLAYTOW {
            // The derived/ copy wins a lookup by name, so rank that one.
            let Some(s) = sheets.iter().find(|s| s.name == *want) else {
                g.push(format!("{want}: not found"));
                continue;
            };
            if s.orthotropy_none_but_wood_like {
                g.push(format!(
                    "{want}: wood-like, orthotropy None, isotropic rules run"
                ));
            }
            for d in DOMAINS.iter().filter(|d| !d.ids.is_empty()) {
                let c = s.cells[d.name];
                let needs = match d.name {
                    "magnetic" => *want == "Magnet-N42",
                    "contact" | "optical" => false,
                    _ => true,
                };
                if needs && c[0] + c[3] == 0 {
                    g.push(format!(
                        "{want}: {} has no datasheet value ({})",
                        d.name,
                        cell(&c)
                    ));
                }
            }
            if s.temp_dependent_without_reference_temp > 0 {
                g.push(format!(
                    "{want}: {} temperature-dependent values, no reference temperature",
                    s.temp_dependent_without_reference_temp
                ));
            }
        }
        g.truncate(20);
        g
    }

    fn lists(sheets: &[Sheet]) -> String {
        let mut unknown: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        let mut not_ready: BTreeMap<String, usize> = BTreeMap::new();
        let (mut wood, mut tdep) = (Vec::new(), 0u32);
        for s in sheets {
            for k in &s.unknown_keys {
                unknown.entry(k).or_default().push(&s.name);
            }
            for k in &s.volume_missing {
                *not_ready.entry(k.clone()).or_default() += 1;
            }
            if s.orthotropy_none_but_wood_like {
                wood.push(s.name.as_str());
            }
            tdep += s.temp_dependent_without_reference_temp;
        }
        let mut out = format!("Sheets not volume-ready, by missing key: {not_ready:?}\n\n");
        out += &format!("Keys ingest drops: {} distinct\n", unknown.len());
        for (k, v) in &unknown {
            out += &format!("- `{k}`: {}\n", v.join(", "));
        }
        out += &format!(
            "\nWood-like with orthotropy None ({}): {}\n",
            wood.len(),
            wood.join(", ")
        );
        out += &format!("\nTemperature-dependent values with no reference temperature: {tdep}\n");
        out
    }

    pub fn render(sheets: &[Sheet]) -> String {
        let mut out = String::from("# Domain census\n\nGenerated by `cargo run -p periodica-mat --example domain_census`. Do not edit. Full data: `domain-census.json`.\n\n");
        out += "## 1. Totals per folder and category\nCells: sheets with a datasheet value / sheets with only derived or estimated values.\n\n";
        out += &group_table(sheets);
        out += "\n## 2. The eleven sheets PlayTow loads\nCells: datasheet/derived/estimated property counts. A name in two folders gets two rows; the lookup by name returns `derived/`, which beats `reference/` (`periodica_core/src/data_loader.rs:67` lists the roots in precedence order, earlier wins; `:294` uses `or_insert`).\n\n";
        out += &playtow_table(sheets);
        out += "\n## 3. Worst gaps for PlayTow, most used sheets first\n";
        for (i, g) in gaps(sheets).iter().enumerate() {
            out += &format!("{}. {g}\n", i + 1);
        }
        out += "\n## 4. Other lists\n";
        out += &lists(sheets);
        out
    }
}
