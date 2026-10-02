//====== periodica/rust/periodica_core/src/sample_models.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # sample_models
//!
//! The five built-in field models, and the Python-compatible primitives they
//! rest on.
//!
//! Split out of [`crate::sample`] so the sampler's dispatch logic stays
//! readable and the parity-critical primitives sit together with the tests
//! that pin them.

use anyhow::Result;
use serde_json::Value;

use crate::sample::{bulk_lookup, no_value, FieldModel, SampleError};

// ── Python-compatible primitives ─────────────────────────────────────────

/// Python's `round()`: half-to-even ("banker's rounding").
///
/// Rust's `f64::round` rounds half *away from zero*, so `round(2.5)` is 3
/// there and 2 in Python. The result quantises a sample point into a grain
/// cell and is then hashed, so a one-cell difference selects a different
/// phase outright -- this is not a rounding nicety, it changes the answer.
#[inline]
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// Python's two-argument `round(x, ndigits)`.
pub fn round_to(x: f64, ndigits: i32) -> f64 {
    let f = 10f64.powi(ndigits);
    let scaled = x * f;
    if scaled.is_finite() {
        scaled.round_ties_even() / f
    } else {
        x
    }
}

/// Python's `repr()` for a float, which the `scale_dependent` phase picker
/// hashes.
///
/// Differences from Rust's `{}` that matter here:
/// - Python always shows a decimal point: `0.0`, where Rust prints `0`.
/// - Python switches to scientific notation below `1e-4` and at/above `1e16`:
///   `1e-05`, where Rust prints `0.00001`. The exponent is signed and padded
///   to at least two digits.
pub fn py_float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }

    let mag = x.abs();
    if mag < 1e-4 || mag >= 1e16 {
        // Rust's `{:e}` gives `1e-5`; Python wants `1e-05`.
        let s = format!("{x:e}");
        let (mantissa, exp) = s
            .split_once('e')
            .expect("`{:e}` always emits an exponent marker");
        let (sign, digits) = match exp.strip_prefix('-') {
            Some(d) => ('-', d),
            None => ('+', exp),
        };
        return format!("{mantissa}e{sign}{digits:0>2}");
    }

    let s = format!("{x}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Grain edge length in metres for a declared grain density.
///
/// Mirrors `sample.py` exactly, **including its two clamps**:
/// `max(1e-12, 1.0 / max(1e-12, g_density)) ** (1/3)`.
///
/// The outer clamp is easy to miss and changes answers: a density of 1e15
/// gives `1/1e15 = 1e-15`, which is clamped *up* to `1e-12`, so the grain is
/// 1e-4 m and not the 1e-5 m the raw cube root would suggest. Sampling on the
/// wrong pitch collapses a test grid onto a handful of cells and makes the
/// phase mix look badly skewed.
pub fn grain_size_from_density(g_density: f64) -> f64 {
    (1.0f64 / g_density.max(1e-12)).max(1e-12).cbrt()
}

/// Uniform draw in `[0, 1)` from the SHA-1 of `key`, matching
/// `int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / 2**64`.
///
/// SHA-1 is slow for a sub-microsecond budget. It is used because agreeing
/// with Python comes first; swapping in a cheap mixer is a deliberate,
/// versioned change that must land in *both* implementations together, never
/// as a unilateral Rust optimisation.
pub fn sha1_unit_interval(key: &str) -> f64 {
    use sha1::{Digest, Sha1};
    let digest = Sha1::digest(key.as_bytes());
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(first8) as f64 / 18_446_744_073_709_551_616.0
}

/// Walk `entries` in order, returning the first key whose cumulative fraction
/// exceeds `u`. `None` when the fractions never reach `u`, which Python treats
/// as "no phase here".
pub fn pick_by_cumulative(entries: &[(String, f64)], u: f64) -> Option<String> {
    let mut cum = 0.0;
    // Bounded by the phase count -- CLAUDE.md bounded-loop rule.
    for (name, frac) in entries {
        cum += *frac;
        if u < cum {
            return Some(name.clone());
        }
    }
    None
}

/// `sample.py::_maybe_phase_at` -- hash a 3D point to a phase name.
pub fn maybe_phase_at(
    point: Option<(f64, f64, f64)>,
    fractions: &[(String, f64)],
) -> Option<String> {
    let (x, y, z) = point?;
    let key = format!(
        "({}, {}, {})",
        py_float_repr(round_to(x, 9)),
        py_float_repr(round_to(y, 9)),
        py_float_repr(round_to(z, 9))
    );
    pick_by_cumulative(fractions, sha1_unit_interval(&key))
}

/// Ordered `(name, fraction)` pairs from a JSON object of phase records.
///
/// Order is the file's key order, preserved by `serde_json`'s `preserve_order`
/// feature. Without it the map is sorted and the cumulative walk above picks a
/// different phase whenever the keys were not already in sorted order.
pub fn ordered_fractions(obj: &serde_json::Map<String, Value>) -> Vec<(String, f64)> {
    obj.iter()
        .filter_map(|(k, v)| {
            v.as_object().map(|o| {
                (
                    k.clone(),
                    o.get("fraction").and_then(Value::as_f64).unwrap_or(0.0),
                )
            })
        })
        .collect()
}

// ── Built-in field models ────────────────────────────────────────────────

/// `homogeneous`: the bulk value, with optional small-scale phase dispatch.
#[derive(Debug)]
pub struct HomogeneousModel;

impl FieldModel for HomogeneousModel {
    fn evaluate(
        &self,
        field: &Value,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        scale_m: Option<f64>,
    ) -> Result<f64> {
        let sd = field.get("scale_dependent").and_then(Value::as_object);
        let macro_scale = field.get("macro_scale_m").and_then(Value::as_f64);
        if let (Some(sd), Some(s), Some(macro_scale)) = (sd, scale_m, macro_scale) {
            if s < macro_scale {
                let fractions = ordered_fractions(sd);
                if let Some(phase) = maybe_phase_at(at, &fractions) {
                    if let Some(v) = sd
                        .get(&phase)
                        .and_then(|p| p.get(property))
                        .and_then(Value::as_f64)
                    {
                        return Ok(v);
                    }
                }
            }
        }
        bulk_lookup(entry, property).ok_or_else(|| no_value(entry, property))
    }
}

/// `anisotropic_axial`: blend axial and transverse values by the cosine
/// between the sample direction and the fibre axis.
#[derive(Debug)]
pub struct AnisotropicAxialModel;

impl FieldModel for AnisotropicAxialModel {
    fn evaluate(
        &self,
        field: &Value,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        _scale_m: Option<f64>,
    ) -> Result<f64> {
        let axial = field
            .get("axial")
            .and_then(|v| v.get(property))
            .and_then(Value::as_f64);
        let transverse = field
            .get("transverse")
            .and_then(|v| v.get(property))
            .and_then(Value::as_f64);
        let bulk = bulk_lookup(entry, property);
        let missing = || no_value(entry, property);

        // No sample point: Python returns the axial value, falling back to bulk.
        let Some((px, py, pz)) = at else {
            return axial.or(bulk).ok_or_else(missing);
        };

        let (mut nx, mut ny, mut nz) = match field.get("fiber_direction").and_then(Value::as_array)
        {
            Some(d) if d.len() == 3 => (
                d[0].as_f64().unwrap_or(0.0),
                d[1].as_f64().unwrap_or(0.0),
                d[2].as_f64().unwrap_or(0.0),
            ),
            _ => (1.0, 0.0, 0.0),
        };
        // Python uses `norm or 1.0`, so an all-zero direction stays unnormalised
        // rather than producing NaN.
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        let norm = if norm == 0.0 { 1.0 } else { norm };
        nx /= norm;
        ny /= norm;
        nz /= norm;

        let pnorm = (px * px + py * py + pz * pz).sqrt();
        if pnorm == 0.0 {
            return axial.or(bulk).ok_or_else(missing);
        }
        let cos = (nx * px / pnorm + ny * py / pnorm + nz * pz / pnorm).abs();

        match (axial, transverse) {
            (None, None) => bulk.ok_or_else(missing),
            (None, Some(t)) => Ok(t),
            (Some(a), None) => Ok(a),
            (Some(a), Some(t)) => Ok(a * cos + t * (1.0 - cos)),
        }
    }
}

/// `mixture`: rule-of-mixtures over the entry's constituents.
#[derive(Debug)]
pub struct MixtureModel;

impl FieldModel for MixtureModel {
    fn evaluate(
        &self,
        field: &Value,
        entry: &Value,
        property: &str,
        _at: Option<(f64, f64, f64)>,
        _scale_m: Option<f64>,
    ) -> Result<f64> {
        let composition = entry
            .get("Composition")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let weights = field
            .get("weights")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(|| composition.clone());

        let mut total = 0.0;
        let mut total_w = 0.0;
        for symbol in composition.keys() {
            // Python swallows every resolution failure and skips the term.
            let Ok(c_entry) = crate::get::Get(symbol, None) else {
                continue;
            };
            let Some(v) = bulk_lookup(&c_entry, property) else {
                continue;
            };
            let w = weights.get(symbol).and_then(Value::as_f64).unwrap_or(1.0);
            total += v * w;
            total_w += w;
        }
        if total_w == 0.0 {
            // Python returns None when no constituent carries the property.
            return Err(no_value(entry, property));
        }
        Ok(total / total_w)
    }
}

/// `microstructure_voronoi`: pick a grain, return that phase's property.
#[derive(Debug)]
pub struct MicrostructureVoronoiModel;

impl FieldModel for MicrostructureVoronoiModel {
    fn evaluate(
        &self,
        field: &Value,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        scale_m: Option<f64>,
    ) -> Result<f64> {
        let bulk = bulk_lookup(entry, property);
        let bulk_or_err = || bulk.ok_or_else(|| no_value(entry, property));

        let macro_scale = field.get("macro_scale_m").and_then(Value::as_f64);
        let (Some(pos), Some(s), Some(macro_scale)) = (at, scale_m, macro_scale) else {
            return bulk_or_err();
        };
        if s >= macro_scale {
            return bulk_or_err();
        }
        let Some(phases) = field.get("phases").and_then(Value::as_object) else {
            return bulk_or_err();
        };
        if phases.is_empty() {
            return bulk_or_err();
        }

        // Quantise the point into a grain cell, mirroring sample.py exactly,
        // including both `max()` guards against a zero grain density.
        let g_density = field
            .get("grain_density")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let grain_size = grain_size_from_density(g_density);
        // Python's `round()` returns an int, so the tuple repr is integral:
        // "(1, -2, 3)".
        let key = format!(
            "({}, {}, {})",
            py_round(pos.0 / grain_size) as i64,
            py_round(pos.1 / grain_size) as i64,
            py_round(pos.2 / grain_size) as i64
        );
        let u = sha1_unit_interval(&key);

        let fractions = ordered_fractions(phases);
        match pick_by_cumulative(&fractions, u) {
            Some(name) => phases
                .get(&name)
                .and_then(|p| p.get(property))
                .and_then(Value::as_f64)
                .or(bulk)
                .ok_or_else(|| no_value(entry, property)),
            None => bulk_or_err(),
        }
    }
}

/// `backbone_path`: per-residue property along a protein backbone.
///
/// The no-residue fallback is handled here. The nearest-alpha-carbon lookup
/// for a 3D point needs the folded backbone, so it defers to Python rather
/// than guessing: returning a *wrong* value would be far worse than deferring.
#[derive(Debug)]
pub struct BackbonePathModel;

impl FieldModel for BackbonePathModel {
    fn evaluate(
        &self,
        _field: &Value,
        entry: &Value,
        property: &str,
        at: Option<(f64, f64, f64)>,
        _scale_m: Option<f64>,
    ) -> Result<f64> {
        let residues = entry
            .get("residues")
            .or_else(|| entry.get("Residues"))
            .and_then(Value::as_array)
            .filter(|r| !r.is_empty());

        let Some(_residues) = residues else {
            return bulk_lookup(entry, property).ok_or_else(|| no_value(entry, property));
        };
        if at.is_some() {
            return Err(anyhow::Error::new(SampleError::Unsupported(
                "backbone_path with a 3D point needs the folded backbone, which the \
                 Rust backend does not build"
                    .to_string(),
            )));
        }
        bulk_lookup(entry, property).ok_or_else(|| no_value(entry, property))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Python-compatibility primitives ──────────────────────────────────
    //
    // Expected values below were produced by running the equivalent
    // expression in CPython 3.13; see the session notes in the module docs.

    #[test]
    fn py_round_is_half_to_even() {
        // CPython: [round(x) for x in (0.5,1.5,2.5,-0.5,-1.5,3.5)]
        //          -> [0, 2, 2, 0, -2, 4]
        let got: Vec<f64> = [0.5, 1.5, 2.5, -0.5, -1.5, 3.5]
            .iter()
            .map(|x| py_round(*x))
            .collect();
        assert_eq!(got, vec![0.0, 2.0, 2.0, 0.0, -2.0, 4.0]);

        // Rust's own `round` disagrees on exactly these inputs, which is the
        // whole reason this helper exists.
        assert_eq!(2.5f64.round(), 3.0);
        assert_eq!(py_round(2.5), 2.0);
    }

    #[test]
    fn py_float_repr_matches_cpython() {
        // Each pair is (value, CPython repr(value)).
        for (v, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-2.0, "-2.0"),
            (0.5, "0.5"),
            (0.0123, "0.0123"),
            (123.456, "123.456"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (1e-9, "1e-09"),
            (1e16, "1e+16"),
            (1e17, "1e+17"),
        ] {
            assert_eq!(py_float_repr(v), want, "repr({v})");
        }
    }

    #[test]
    fn py_float_repr_always_round_trips() {
        // Whatever formatting we choose, it must not lose precision.
        for v in [0.1, 1.0 / 3.0, 123.456_789, 2.5e-7, 9.87e15] {
            let s = py_float_repr(v);
            let back: f64 = s.parse().unwrap();
            assert_eq!(back, v, "{s}");
        }
    }

    #[test]
    fn sha1_draw_matches_cpython() {
        // CPython:
        //   h = hashlib.sha1(repr((1,-2,3)).encode()).digest()
        //   int.from_bytes(h[:8],'big') / 2**64  -> 0.2840237670897698
        let u = sha1_unit_interval("(1, -2, 3)");
        assert!(
            (u - 0.284_023_767_089_769_8).abs() < 1e-15,
            "got {u}, expected 0.2840237670897698"
        );
        assert!((0.0..1.0).contains(&u));
    }

    #[test]
    fn sha1_draw_is_deterministic_and_spreads() {
        assert_eq!(
            sha1_unit_interval("(0, 0, 0)"),
            sha1_unit_interval("(0, 0, 0)")
        );
        assert_ne!(
            sha1_unit_interval("(0, 0, 0)"),
            sha1_unit_interval("(0, 0, 1)")
        );

        // Rough uniformity: 1000 distinct cells should land either side of 0.5
        // in roughly equal numbers.
        let below = (0..1000)
            .filter(|i| sha1_unit_interval(&format!("({i}, 0, 0)")) < 0.5)
            .count();
        assert!((400..600).contains(&below), "{below}/1000 below 0.5");
    }

    #[test]
    fn grain_size_applies_pythons_lower_clamp() {
        // 1/1e15 = 1e-15 is clamped up to 1e-12, so the grain is 1e-4 m.
        // Dropping the clamp yields 1e-5 and silently changes which cell -- and
        // therefore which phase -- every sample point lands in.
        assert!((grain_size_from_density(1e15) - 1e-4).abs() < 1e-18);
        // Below the clamp threshold the cube root is taken directly.
        assert!((grain_size_from_density(1e9) - 1e-3).abs() < 1e-17);
        assert!((grain_size_from_density(1e12) - 1e-4).abs() < 1e-18);
        // Degenerate densities must not produce zero, inf or NaN.
        for gd in [0.0, -1.0, f64::INFINITY] {
            let g = grain_size_from_density(gd);
            assert!(g.is_finite() && g > 0.0, "grain_size({gd}) = {g}");
        }
    }

    #[test]
    fn cumulative_pick_respects_declared_order() {
        let f = vec![("a".to_string(), 0.3), ("b".to_string(), 0.7)];
        assert_eq!(pick_by_cumulative(&f, 0.0).as_deref(), Some("a"));
        assert_eq!(pick_by_cumulative(&f, 0.29).as_deref(), Some("a"));
        assert_eq!(pick_by_cumulative(&f, 0.3).as_deref(), Some("b"));
        assert_eq!(pick_by_cumulative(&f, 0.99).as_deref(), Some("b"));
        // Fractions that do not reach u yield no phase, as in Python.
        assert_eq!(pick_by_cumulative(&f, 1.0), None);
    }

    #[test]
    fn phase_order_follows_the_file_not_the_alphabet() {
        // The parity trap: serde_json's default Map is sorted. With
        // `preserve_order` the declared order survives, and the cumulative walk
        // therefore selects the same phase Python would.
        let v: Value =
            serde_json::from_str(r#"{"pearlite":{"fraction":0.15},"ferrite":{"fraction":0.85}}"#)
                .unwrap();
        let names: Vec<String> = ordered_fractions(v.as_object().unwrap())
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            names,
            vec!["pearlite", "ferrite"],
            "serde_json is sorting keys -- the `preserve_order` feature is missing, \
             and voronoi phase selection will silently disagree with Python"
        );
    }
}
