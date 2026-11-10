//! Flattening of a datasheet's property groups: a port of
//! `periodica.properties.flatten`.
//!
//! The group and exclusion lists come from `config/property_schema.json`
//! (the same file Python reads); they are not copied here. The alias and unit
//! vocabulary is not ported: only literal keys of the flattened sheet are
//! answered natively.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use serde_json::{Map, Value};

/// Deepest nesting `flatten` descends into (`properties.MAX_GROUP_DEPTH`).
pub const MAX_GROUP_DEPTH: usize = 3;

/// The two key lists `flatten` needs from the schema.
#[derive(Debug, Default, Clone)]
pub struct GroupSchema {
    /// `property_group_keys`, in schema order.
    pub group_keys: Vec<String>,
    /// `excluded_group_keys`.
    pub excluded: HashSet<String>,
}

/// Drop `//` comments to end of line, exactly as `properties._strip_comments`.
fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

fn string_list(raw: &Value, key: &str) -> Vec<String> {
    raw.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse the schema text.
pub fn parse_schema(text: &str) -> Result<GroupSchema, serde_json::Error> {
    let raw: Value = serde_json::from_str(&strip_comments(text))?;
    Ok(GroupSchema {
        group_keys: string_list(&raw, "property_group_keys"),
        excluded: string_list(&raw, "excluded_group_keys")
            .into_iter()
            .collect(),
    })
}

const SCHEMA_FILE: &str = "config/property_schema.json";

/// The loaded data root's schema when it has one (an installed package),
/// otherwise the default data root's. A fixture root without the file must not
/// make the real schema unreachable.
fn schema_path() -> PathBuf {
    let loaded = crate::data_loader::DATA.read().root.clone();
    if let Some(p) = loaded.map(|r| r.join(SCHEMA_FILE)).filter(|p| p.is_file()) {
        return p;
    }
    crate::data_loader::default_data_root().join(SCHEMA_FILE)
}

/// Read and parse the schema at `path`; the error names the path.
pub fn load_schema(path: &Path) -> Result<GroupSchema, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("property schema: cannot read {}: {e}", path.display()))?;
    parse_schema(&text)
        .map_err(|e| format!("property schema: cannot parse {}: {e}", path.display()))
}

static SCHEMA: Lazy<Result<GroupSchema, String>> = Lazy::new(|| load_schema(&schema_path()));

/// The property schema could not be read (a bug or a broken install, never a
/// decline to Python).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct SchemaError(pub String);

/// The schema read once from the data directory, or the reason it is unusable.
pub fn schema() -> Result<&'static GroupSchema, &'static str> {
    SCHEMA.as_ref().map_err(String::as_str)
}

fn is_scalar(v: &Value) -> bool {
    v.is_number() || v.is_string()
}

fn skipped(key: &str, schema: &GroupSchema) -> bool {
    schema.excluded.contains(key) || key.starts_with('_')
}

fn merge(
    source: &Map<String, Value>,
    depth: usize,
    schema: &GroupSchema,
    out: &mut Map<String, Value>,
) {
    for (key, value) in source {
        if skipped(key, schema) {
            continue;
        }
        if is_scalar(value) {
            if !out.contains_key(key) {
                out.insert(key.clone(), value.clone());
            }
        } else if let Some(obj) = value.as_object() {
            if depth < MAX_GROUP_DEPTH {
                merge(obj, depth + 1, schema, out);
            }
        }
    }
}

/// Collapse an entry's property groups into one flat map; first key wins.
pub fn flatten_with(entry: &Value, schema: &GroupSchema) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(root) = entry.as_object() else {
        return out;
    };
    for (key, value) in root {
        if !skipped(key, schema) && is_scalar(value) && !out.contains_key(key) {
            out.insert(key.clone(), value.clone());
        }
    }
    for key in ["Properties", "properties"] {
        if let Some(block) = root.get(key).and_then(Value::as_object) {
            merge(block, 1, schema, &mut out);
        }
    }
    for key in &schema.group_keys {
        if key == "Properties" || key == "properties" {
            continue;
        }
        if let Some(block) = root.get(key).and_then(Value::as_object) {
            merge(block, 1, schema, &mut out);
        }
    }
    for (key, value) in root {
        let named = schema.group_keys.iter().any(|g| g == key);
        if skipped(key, schema) || named {
            continue;
        }
        if let Some(block) = value.as_object() {
            merge(block, 1, schema, &mut out);
        }
    }
    out
}

/// [`flatten_with`] using the schema from the data directory; an error when
/// that schema cannot be read.
pub fn flatten(entry: &Value) -> anyhow::Result<Map<String, Value>> {
    let schema = schema().map_err(|e| anyhow::Error::new(SchemaError(e.to_string())))?;
    Ok(flatten_with(entry, schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sch() -> GroupSchema {
        GroupSchema {
            group_keys: vec![
                "Properties".into(),
                "PhysicalProperties".into(),
                "ElasticProperties".into(),
            ],
            excluded: ["Composition".to_string()].into_iter().collect(),
        }
    }

    #[test]
    fn nested_sheet_flattens_and_drops_group_names() {
        let e = json!({"PhysicalProperties": {"Density": 7.0}, "ElasticProperties": {"E": 2.0}});
        let f = flatten_with(&e, &sch());
        assert_eq!(f.get("Density"), Some(&json!(7.0)));
        assert_eq!(f.get("E"), Some(&json!(2.0)));
        assert!(!f.contains_key("PhysicalProperties"));
    }

    #[test]
    fn root_scalar_beats_group_and_properties_beats_named_group() {
        let e =
            json!({"A": 1, "Properties": {"A": 2, "B": 3}, "PhysicalProperties": {"A": 9, "B": 8}});
        let f = flatten_with(&e, &sch());
        assert_eq!(f["A"], json!(1));
        assert_eq!(f["B"], json!(3));
    }

    #[test]
    fn booleans_and_excluded_keys_dropped_strings_kept() {
        let e = json!({"flag": true, "Name": "x", "Composition": {"Fe": 1}, "_hidden": 1,
                       "Properties": {"ok": false, "Fe": 2}});
        let f = flatten_with(&e, &sch());
        assert!(!f.contains_key("flag") && !f.contains_key("ok"));
        assert!(!f.contains_key("_hidden"));
        assert_eq!(f["Name"], json!("x"));
    }

    #[test]
    fn depth_four_is_not_reached() {
        let e = json!({"Properties": {"l1": 1, "g": {"l2": 2, "g": {"l3": 3, "g": {"l4": 4}}}}});
        let f = flatten_with(&e, &sch());
        assert!(f.contains_key("l1") && f.contains_key("l2") && f.contains_key("l3"));
        assert!(!f.contains_key("l4"));
    }

    #[test]
    fn malformed_schema_is_an_error() {
        assert!(parse_schema("{ not json").is_err());
    }

    #[test]
    fn missing_schema_file_is_an_error_naming_the_path() {
        let p = Path::new("H:/definitely/not/here/property_schema.json");
        let err = load_schema(p).unwrap_err();
        assert!(err.contains("not/here/property_schema.json"), "{err}");
    }

    #[test]
    fn bundled_schema_parses() {
        let text = std::fs::read_to_string(schema_path()).expect("schema file");
        let s = parse_schema(&text).expect("parse");
        assert!(s.group_keys.iter().any(|k| k == "PhysicalProperties"));
        assert!(s.excluded.contains("Composition"));
    }
}
