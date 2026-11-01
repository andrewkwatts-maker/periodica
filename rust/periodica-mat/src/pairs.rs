//====== periodica/rust/periodica-mat/src/pairs.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # pairs
//!
//! Measured interface values for two specific records, as plain JSON with a
//! citation on every row. A row overrides the law derived from the two sheets.
//!
//! The key is the two exact record ids (`MaterialRecord::name`), unordered.
//! There are no family keys and no surface-finish rows; published family
//! tables (EC2 roughness, handbook friction) live in `tests/oracles/` as test
//! answers. A miss is a typed [`PairMiss`]; there is no default in periodica,
//! so the caller decides and logs what to do.

use serde::Deserialize;
use thiserror::Error;

const BUNDLED: &str = include_str!("../../../src/periodica/data/interfaces/pairs.json");

/// Law kind and its parameters, SI (dimensionless unless stated).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "law")]
pub enum Law {
    /// Dry friction: static `mu_s` (mid of `mu_s_range` when a range is
    /// published) and optional kinetic `mu_k`.
    Coulomb {
        mu_s: f64,
        #[serde(default)]
        mu_k: Option<f64>,
        #[serde(default)]
        mu_s_range: Option<[f64; 2]>,
    },
    /// Concrete shear interface: `cohesion_factor` multiplies `f_ctd`.
    Cohesive { cohesion_factor: f64, mu: f64 },
    /// MC2010 mean bond stress-slip coefficients, not evaluated numbers.
    /// `tau_max = coeff * sqrt(f_cm)` in Pa with `f_cm` in Pa; slips in m.
    BondSlip {
        tau_max_coeff_good: f64,
        tau_max_coeff_other: f64,
        s1_good: f64,
        s1_other: f64,
        s2_good: f64,
        s2_other: f64,
        s3_rule: String,
        alpha: f64,
    },
}

/// One pair row.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PairRow {
    /// Exact record id (`MaterialRecord::name`).
    pub a: String,
    /// Exact record id (`MaterialRecord::name`).
    pub b: String,
    #[serde(flatten)]
    pub law: Law,
    /// Citation, required and non-empty.
    pub source: String,
}

/// A lookup that found no row. Names the pair so the caller can log it.
#[derive(Debug, Clone, PartialEq, Error)]
#[error("no interface pair row for records {a} and {b}")]
pub struct PairMiss {
    pub a: String,
    pub b: String,
}

/// Load failures.
#[derive(Debug, Error)]
pub enum PairError {
    #[error("pairs JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("row {row} ({a} / {b}) has no source citation")]
    NoSource { row: usize, a: String, b: String },
    #[error("rows {first} and {second} give the same pair {a} / {b}")]
    Conflict {
        first: usize,
        second: usize,
        a: String,
        b: String,
    },
}

#[derive(Deserialize)]
struct File {
    rows: Vec<PairRow>,
}

/// The loaded, validated table.
#[derive(Debug, Clone, Default)]
pub struct PairTable {
    rows: Vec<PairRow>,
}

/// Orientation-independent key: the smaller id first.
fn norm<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl PairTable {
    /// The table bundled with periodica.
    pub fn bundled() -> Result<Self, PairError> {
        Self::from_json(BUNDLED)
    }

    /// Parse and validate. Missing sources and conflicting rows fail.
    pub fn from_json(text: &str) -> Result<Self, PairError> {
        let file: File = serde_json::from_str(text)?;
        let rows = file.rows;
        for (i, r) in rows.iter().enumerate() {
            if r.source.trim().is_empty() {
                return Err(PairError::NoSource { row: i, a: r.a.clone(), b: r.b.clone() });
            }
        }
        check_conflicts(&rows)?;
        Ok(Self { rows })
    }

    pub fn rows(&self) -> &[PairRow] {
        &self.rows
    }

    /// Unordered lookup by the two record ids.
    pub fn lookup(&self, a: &str, b: &str) -> Result<&PairRow, PairMiss> {
        let k = norm(a, b);
        self.rows
            .iter()
            .find(|r| norm(&r.a, &r.b) == k)
            .ok_or_else(|| PairMiss { a: a.to_string(), b: b.to_string() })
    }
}

fn check_conflicts(rows: &[PairRow]) -> Result<(), PairError> {
    for (j, r) in rows.iter().enumerate() {
        let k = norm(&r.a, &r.b);
        for (i, p) in rows.iter().enumerate().take(j) {
            if norm(&p.a, &p.b) == k {
                return Err(PairError::Conflict {
                    first: i,
                    second: j,
                    a: r.a.clone(),
                    b: r.b.clone(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coulomb(a: &str, b: &str, mu: f64) -> String {
        format!(r#"{{"a":"{a}","b":"{b}","law":"Coulomb","mu_s":{mu},"source":"s"}}"#)
    }

    #[test]
    fn bundled_table_loads_and_cites_sources() {
        let t = PairTable::bundled().unwrap();
        assert!(t.rows().iter().all(|r| !r.source.trim().is_empty()));
    }

    #[test]
    fn lookup_is_symmetric_in_record_ids() {
        let json = format!(r#"{{"rows":[{}]}}"#, coulomb("Steel-1018", "Glass_Borosilicate", 0.4));
        let t = PairTable::from_json(&json).unwrap();
        let fwd = t.lookup("Steel-1018", "Glass_Borosilicate").unwrap();
        let rev = t.lookup("Glass_Borosilicate", "Steel-1018").unwrap();
        assert_eq!(fwd, rev);
    }

    #[test]
    fn missing_pair_notice_names_both_records() {
        let t = PairTable::bundled().unwrap();
        let msg = t.lookup("Unobtainium", "Steel-1018").unwrap_err().to_string();
        assert!(msg.contains("Unobtainium") && msg.contains("Steel-1018"), "{msg}");
    }

    #[test]
    fn blank_or_absent_source_fails() {
        let blank = coulomb("A", "B", 0.5).replace(r#""source":"s""#, r#""source":"  ""#);
        let json = format!(r#"{{"rows":[{blank}]}}"#);
        assert!(matches!(PairTable::from_json(&json), Err(PairError::NoSource { .. })));
        let absent = coulomb("A", "B", 0.5).replace(r#","source":"s""#, "");
        let json = format!(r#"{{"rows":[{absent}]}}"#);
        assert!(matches!(PairTable::from_json(&json), Err(PairError::Json(_))));
    }

    #[test]
    fn conflicting_rows_fail_naming_both() {
        let json = format!(r#"{{"rows":[{},{}]}}"#, coulomb("A", "B", 0.5), coulomb("B", "A", 0.6));
        match PairTable::from_json(&json) {
            Err(PairError::Conflict { first: 0, second: 1, .. }) => {}
            other => panic!("{other:?}"),
        }
    }
}
