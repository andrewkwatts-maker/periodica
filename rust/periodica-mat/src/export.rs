//! Column export of scalar material properties (`record.bulk`) for engines that bake
//! per-property arrays: one row per sheet, one `.f64` and one `.src` file per property.
//!
//! Layout of an export directory:
//! - `<Property>.f64`: one little-endian f64 per row; a missing value is NaN.
//! - `<Property>.src`: one byte per row, the [`Source`] code. 0 means no data, so `.src` is the
//!   authority for presence; the NaN is a convenience.
//! - `manifest.json`: see [`Manifest`]. Later versions only add fields and files; [`read_dir`]
//!   accepts a higher `format_version` and ignores what it does not know.
//!
//! Sheet ids are sheet names (file stems). Rows are sorted by id, so two exports of one corpus
//! are byte-identical.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::record::MaterialRecord;
use crate::table::Source;
use crate::{ingest, is_material_like, jsonc, PropertyId};

/// Current manifest version written by [`write_dir`].
pub const FORMAT_VERSION: u32 = 1;
/// Name of the hash used for the manifest's content hashes.
pub const HASH_ALGORITHM: &str = "fnv1a-64";
/// Data roots in precedence order; the first root holding a stem wins.
pub const ROOTS: &[&str] = &["active", "derived", "reference", "defaults"];
/// Sheets always exported, though `is_material_like` may reject them (PlayTow loads them).
pub const ALWAYS_INCLUDE: &[&str] = &[
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
const SOURCES: [Source; 5] = [
    Source::Absent,
    Source::Datasheet,
    Source::Derived,
    Source::Estimated,
    Source::UserOverride,
];

/// Everything that can go wrong exporting or importing columns.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("duplicate sheet id {0:?}")]
    DuplicateSheetId(String),
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("bad manifest: {0}")]
    Manifest(String),
    #[error("unsupported format_version {0}")]
    Version(u32),
    #[error("unknown property {0:?} in manifest")]
    UnknownProperty(String),
    #[error("{file}: expected {expected} bytes, found {found}")]
    Length {
        file: String,
        expected: usize,
        found: usize,
    },
    #[error("{file}: hash mismatch (manifest {expected}, file {found})")]
    Hash {
        file: String,
        expected: String,
        found: String,
    },
    #[error("{0}: not a valid source code")]
    SourceCode(String),
}

pub type Result<T> = std::result::Result<T, ExportError>;

/// A sheet that lost to another of the same name in a higher-precedence folder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shadowed {
    pub name: String,
    pub folder: String,
}

/// One property across all rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub prop: PropertyId,
    /// One value per row; NaN where missing.
    pub values: Vec<f64>,
    /// One [`Source`] code per row; 0 where missing.
    pub sources: Vec<u8>,
}

/// The columns plus the row ids. Built by [`columns`] or [`read_dir`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColumnSet {
    pub sheet_ids: Vec<String>,
    pub shadowed: Vec<Shadowed>,
    pub columns: Vec<Column>,
}

/// One property's entry in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyEntry {
    pub name: String,
    pub id: u32,
    pub unit: String,
    pub values_file: String,
    pub source_file: String,
    pub values_hash: String,
    pub source_hash: String,
    pub present: usize,
}

/// The manifest as written to `manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub endianness: String,
    pub hash_algorithm: String,
    /// Says which file is authoritative for presence.
    pub presence_authority: String,
    pub rows: usize,
    pub sheet_ids: Vec<String>,
    #[serde(default)]
    pub shadowed: Vec<Shadowed>,
    /// Source code (as a decimal string) to name.
    pub source_legend: BTreeMap<String, String>,
    pub properties: Vec<PropertyEntry>,
}

/// FNV-1a 64-bit over `bytes`, as 16 lowercase hex digits.
pub fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn column_for(rows: &[&(String, MaterialRecord)], prop: PropertyId) -> Column {
    let mut col = Column {
        prop,
        values: Vec::with_capacity(rows.len()),
        sources: Vec::with_capacity(rows.len()),
    };
    for (_, rec) in rows {
        let src = rec.bulk.source(prop);
        match rec.bulk.get(prop) {
            Some(v) if src != Source::Absent => {
                col.values.push(v);
                col.sources.push(src as u8);
            }
            _ => {
                col.values.push(f64::NAN);
                col.sources.push(0);
            }
        }
    }
    col
}

/// Build the columns for `props`, one row per record, rows sorted by sheet id.
/// Fails early on a duplicate id.
pub fn columns(records: &[(String, MaterialRecord)], props: &[PropertyId]) -> Result<ColumnSet> {
    let mut rows: Vec<&(String, MaterialRecord)> = records.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(w) = rows.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(ExportError::DuplicateSheetId(w[0].0.clone()));
    }
    Ok(ColumnSet {
        sheet_ids: rows.iter().map(|r| r.0.clone()).collect(),
        shadowed: Vec::new(),
        columns: props.iter().map(|&p| column_for(&rows, p)).collect(),
    })
}

fn io_err(path: &Path, source: std::io::Error) -> ExportError {
    ExportError::Io {
        path: path.display().to_string(),
        source,
    }
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let p = dir.join(name);
    std::fs::write(&p, bytes).map_err(|e| io_err(&p, e))
}

fn entry_for(dir: &Path, col: &Column) -> Result<PropertyEntry> {
    let name = col.prop.name();
    let values: Vec<u8> = col.values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (vf, sf) = (format!("{name}.f64"), format!("{name}.src"));
    write_file(dir, &vf, &values)?;
    write_file(dir, &sf, &col.sources)?;
    Ok(PropertyEntry {
        name: name.to_string(),
        id: col.prop.index() as u32,
        unit: col.prop.si_unit().to_string(),
        values_file: vf,
        source_file: sf,
        values_hash: fnv1a_hex(&values),
        source_hash: fnv1a_hex(&col.sources),
        present: col.sources.iter().filter(|&&s| s != 0).count(),
    })
}

fn legend() -> BTreeMap<String, String> {
    SOURCES
        .iter()
        .map(|&s| ((s as u8).to_string(), format!("{s:?}")))
        .collect()
}

/// Write `set` into `dir` (created if missing) and return the manifest written.
pub fn write_dir(set: &ColumnSet, dir: &Path) -> Result<Manifest> {
    let rows = set.sheet_ids.len();
    if let Some(c) = set
        .columns
        .iter()
        .find(|c| c.values.len() != rows || c.sources.len() != rows)
    {
        return Err(ExportError::Length {
            file: c.prop.name().to_string(),
            expected: rows,
            found: c.values.len().min(c.sources.len()),
        });
    }
    std::fs::create_dir_all(dir).map_err(|e| io_err(dir, e))?;
    let properties = set
        .columns
        .iter()
        .map(|c| entry_for(dir, c))
        .collect::<Result<Vec<_>>>()?;
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        endianness: "little".into(),
        hash_algorithm: HASH_ALGORITHM.into(),
        presence_authority: "src".into(),
        rows,
        sheet_ids: set.sheet_ids.clone(),
        shadowed: set.shadowed.clone(),
        source_legend: legend(),
        properties,
    };
    let text = serde_json::to_string_pretty(&manifest)
        .map_err(|e| ExportError::Manifest(e.to_string()))?;
    write_file(dir, "manifest.json", (text + "\n").as_bytes())?;
    Ok(manifest)
}

fn read_checked(dir: &Path, file: &str, want_len: usize, want_hash: &str) -> Result<Vec<u8>> {
    let p = dir.join(file);
    let bytes = std::fs::read(&p).map_err(|e| io_err(&p, e))?;
    if bytes.len() != want_len {
        return Err(ExportError::Length {
            file: file.to_string(),
            expected: want_len,
            found: bytes.len(),
        });
    }
    let found = fnv1a_hex(&bytes);
    if found != want_hash {
        return Err(ExportError::Hash {
            file: file.to_string(),
            expected: want_hash.to_string(),
            found,
        });
    }
    Ok(bytes)
}

fn read_column(dir: &Path, rows: usize, e: &PropertyEntry) -> Result<Column> {
    let prop = PropertyId::ALL
        .iter()
        .copied()
        .find(|p| p.name() == e.name)
        .ok_or_else(|| ExportError::UnknownProperty(e.name.clone()))?;
    let raw = read_checked(dir, &e.values_file, rows * 8, &e.values_hash)?;
    let sources = read_checked(dir, &e.source_file, rows, &e.source_hash)?;
    if let Some(&bad) = sources.iter().find(|&&s| s as usize >= SOURCES.len()) {
        return Err(ExportError::SourceCode(format!("{}: {bad}", e.source_file)));
    }
    let values = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("chunk of 8")))
        .collect();
    Ok(Column {
        prop,
        values,
        sources,
    })
}

/// Read an export back. Accepts a `format_version` of 1 or higher and ignores manifest
/// fields and files it does not know.
pub fn read_dir(dir: &Path) -> Result<ColumnSet> {
    let p = dir.join("manifest.json");
    let text = std::fs::read_to_string(&p).map_err(|e| io_err(&p, e))?;
    let m: Manifest =
        serde_json::from_str(&text).map_err(|e| ExportError::Manifest(e.to_string()))?;
    if m.format_version < 1 {
        return Err(ExportError::Version(m.format_version));
    }
    if m.endianness != "little" {
        return Err(ExportError::Manifest(format!(
            "endianness {:?}",
            m.endianness
        )));
    }
    if m.sheet_ids.len() != m.rows {
        return Err(ExportError::Length {
            file: "sheet_ids".into(),
            expected: m.rows,
            found: m.sheet_ids.len(),
        });
    }
    let columns = m
        .properties
        .iter()
        .map(|e| read_column(dir, m.rows, e))
        .collect::<Result<Vec<_>>>()?;
    Ok(ColumnSet {
        sheet_ids: m.sheet_ids,
        shadowed: m.shadowed,
        columns,
    })
}

/// A file the corpus walk did not turn into a row, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub path: String,
    pub reason: String,
}

/// The corpus as rows: records keyed by sheet id, the copies shadowed by name, the rest skipped.
#[derive(Debug, Default)]
pub struct Corpus {
    pub records: Vec<(String, MaterialRecord)>,
    pub shadowed: Vec<Shadowed>,
    pub skipped: Vec<Skipped>,
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

fn folder_of(data_dir: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(data_dir).unwrap_or(path);
    let dir = rel.parent().unwrap_or(Path::new(""));
    dir.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Ingest one file (the same `jsonc` + `ingest` path as the census, which fills derived and
/// estimated values). `Err` carries the reason it is not a row.
fn load_sheet(path: &Path, stem: &str) -> std::result::Result<MaterialRecord, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    let raw = jsonc::from_str(&text).map_err(|e| format!("not JSON: {e}"))?;
    let rec = ingest(&raw, stem).map_err(|e| format!("ingest failed: {e}"))?;
    if is_material_like(&rec) || ALWAYS_INCLUDE.contains(&stem) {
        Ok(rec)
    } else {
        Err("not material-like".to_string())
    }
}

/// Walk the four data roots under `data_dir` (precedence order [`ROOTS`], files sorted within a
/// root). The first copy of a name wins, as in the engine's loader; later copies are listed in
/// `shadowed`.
pub fn load_corpus(data_dir: &Path) -> Corpus {
    let mut corpus = Corpus::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for root in ROOTS {
        let mut files = Vec::new();
        collect_json(&data_dir.join(root), &mut files);
        files.sort();
        for path in files {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let folder = folder_of(data_dir, &path);
            match load_sheet(&path, &stem) {
                Err(reason) => corpus.skipped.push(Skipped {
                    path: format!("{folder}/{stem}"),
                    reason,
                }),
                Ok(_) if seen.contains(&stem) => {
                    corpus.shadowed.push(Shadowed { name: stem, folder })
                }
                Ok(rec) => {
                    seen.insert(stem.clone());
                    corpus.records.push((stem, rec));
                }
            }
        }
    }
    corpus
}
