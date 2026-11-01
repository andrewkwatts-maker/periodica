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
//! Interface pair data: friction, cohesion and bond parameters per material
//! pair and surface finish, as plain JSON with a citation on every row.
//!
//! The key is `(a, b, finish_a, finish_b)`, unordered. An exact finish beats
//! [`Finish::Any`]. A miss is a typed [`PairMiss`]; there is no default in
//! periodica, so the caller decides and logs what to do.

use serde::Deserialize;
use thiserror::Error;

const BUNDLED: &str = include_str!("../../../src/periodica/data/interfaces/pairs.json");

/// Surface finish of one side of an interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
pub enum Finish {
    Smooth,
    Rough,
    Indented,
    VerySmooth,
    Ribbed,
    PlainHotRolled,
    PlainColdDrawn,
    /// Matches any finish.
    Any,
}

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
    pub a: String,
    pub b: String,
    pub finish_a: Finish,
    pub finish_b: Finish,
    #[serde(flatten)]
    pub law: Law,
    /// Citation, required and non-empty.
    pub source: String,
}

/// A lookup that found no row. Names the pair so the caller can log it.
#[derive(Debug, Clone, PartialEq, Error)]
#[error("no interface pair row for {a} ({finish_a:?}) with {b} ({finish_b:?})")]
pub struct PairMiss {
    pub a: String,
    pub b: String,
    pub finish_a: Finish,
    pub finish_b: Finish,
}

/// Load failures.
#[derive(Debug, Error)]
pub enum PairError {
    #[error("pairs JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("row {row} ({a} / {b}) has no source citation")]
    NoSource { row: usize, a: String, b: String },
    #[error("rows {first} and {second} give the same pair {a} ({fa:?}) / {b} ({fb:?})")]
    Conflict {
        first: usize,
        second: usize,
        a: String,
        b: String,
        fa: Finish,
        fb: Finish,
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

type Key<'a> = (&'a str, Finish, &'a str, Finish);

/// Orientation-independent key: the smaller side first.
fn norm<'a>(a: &'a str, fa: Finish, b: &'a str, fb: Finish) -> Key<'a> {
    if (a, fa) <= (b, fb) {
        (a, fa, b, fb)
    } else {
        (b, fb, a, fa)
    }
}

fn side_score(row: Finish, want: Finish) -> Option<u8> {
    if row == want {
        Some(1)
    } else if row == Finish::Any {
        Some(0)
    } else {
        None
    }
}

fn score(r: &PairRow, a: &str, b: &str, fa: Finish, fb: Finish) -> Option<u8> {
    if r.a != a || r.b != b {
        return None;
    }
    Some(side_score(r.finish_a, fa)? + side_score(r.finish_b, fb)?)
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

    /// Unordered lookup. An exact finish beats `Any`; on equal score the
    /// earlier row in the file wins.
    pub fn lookup(&self, a: &str, b: &str, fa: Finish, fb: Finish) -> Result<&PairRow, PairMiss> {
        let mut best: Option<(u8, &PairRow)> = None;
        for r in &self.rows {
            let s = score(r, a, b, fa, fb).max(score(r, b, a, fb, fa));
            if let Some(s) = s {
                if best.is_none_or(|(bs, _)| s > bs) {
                    best = Some((s, r));
                }
            }
        }
        best.map(|(_, r)| r).ok_or_else(|| PairMiss {
            a: a.to_string(),
            b: b.to_string(),
            finish_a: fa,
            finish_b: fb,
        })
    }
}

fn check_conflicts(rows: &[PairRow]) -> Result<(), PairError> {
    for (j, r) in rows.iter().enumerate() {
        let k = norm(&r.a, r.finish_a, &r.b, r.finish_b);
        for (i, p) in rows.iter().enumerate().take(j) {
            if norm(&p.a, p.finish_a, &p.b, p.finish_b) == k {
                return Err(PairError::Conflict {
                    first: i,
                    second: j,
                    a: r.a.clone(),
                    b: r.b.clone(),
                    fa: r.finish_a,
                    fb: r.finish_b,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coulomb(a: &str, b: &str, fa: &str, fb: &str, mu: f64) -> String {
        format!(
            r#"{{"a":"{a}","b":"{b}","finish_a":"{fa}","finish_b":"{fb}","law":"Coulomb","mu_s":{mu},"source":"s"}}"#
        )
    }

    #[test]
    fn pair_lookup_symmetric() {
        let t = PairTable::bundled().unwrap();
        assert!(!t.rows().is_empty());
        for r in t.rows() {
            let fwd = t.lookup(&r.a, &r.b, r.finish_a, r.finish_b).unwrap();
            let rev = t.lookup(&r.b, &r.a, r.finish_b, r.finish_a).unwrap();
            assert_eq!(fwd, rev);
        }
    }

    #[test]
    fn missing_pair_notice() {
        let t = PairTable::bundled().unwrap();
        let miss = t.lookup("Unobtainium", "Steel", Finish::Any, Finish::Any).unwrap_err();
        let msg = miss.to_string();
        assert!(msg.contains("Unobtainium") && msg.contains("Steel"), "{msg}");
    }

    #[test]
    fn pair_rows_cite_sources() {
        let t = PairTable::bundled().unwrap();
        assert!(t.rows().iter().all(|r| !r.source.trim().is_empty()));
        let blank = coulomb("A", "B", "Any", "Any", 0.5).replace(r#""source":"s""#, r#""source":"  ""#);
        let json = format!(r#"{{"rows":[{blank}]}}"#);
        assert!(matches!(PairTable::from_json(&json), Err(PairError::NoSource { .. })));
        let absent = coulomb("A", "B", "Any", "Any", 0.5).replace(r#","source":"s""#, "");
        let json = format!(r#"{{"rows":[{absent}]}}"#);
        assert!(matches!(PairTable::from_json(&json), Err(PairError::Json(_))));
    }

    #[test]
    fn exact_finish_beats_any() {
        let json = format!(
            r#"{{"rows":[{},{}]}}"#,
            coulomb("A", "B", "Any", "Any", 0.1),
            coulomb("A", "B", "Rough", "Any", 0.9)
        );
        let t = PairTable::from_json(&json).unwrap();
        let hit = t.lookup("B", "A", Finish::Any, Finish::Rough).unwrap();
        assert!(matches!(hit.law, Law::Coulomb { mu_s, .. } if mu_s == 0.9));
        let other = t.lookup("A", "B", Finish::Smooth, Finish::Any).unwrap();
        assert!(matches!(other.law, Law::Coulomb { mu_s, .. } if mu_s == 0.1));
    }

    #[test]
    fn conflicting_rows_fail_naming_both() {
        let json = format!(
            r#"{{"rows":[{},{}]}}"#,
            coulomb("A", "B", "Any", "Any", 0.5),
            coulomb("B", "A", "Any", "Any", 0.6)
        );
        match PairTable::from_json(&json) {
            Err(PairError::Conflict { first: 0, second: 1, .. }) => {}
            other => panic!("{other:?}"),
        }
    }
}
