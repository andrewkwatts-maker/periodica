//====== periodica/rust/periodica-mat/src/table.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # table
//!
//! [`PropertyTable`] -- a dense, SI-normalised bag of material properties with
//! per-entry provenance.
//!
//! ## Layout rationale
//!
//! A `HashMap<String, f64>` would cost a hash and a pointer chase per lookup.
//! Because [`PropertyId`] is dense and `#[repr(u16)]`, this is instead a plain
//! array indexed by the discriminant: lookup is one bounds check and one load.
//!
//! Presence is tracked in a side bitset rather than by a NaN sentinel. A
//! sentinel would be one word smaller but makes "absent" and "computed a NaN"
//! indistinguishable, which is precisely the bug class this table exists to
//! prevent.
//!
//! ## Precision
//!
//! Values are `f64` here even though the runtime evaluates in `f32`. This is
//! the *authoring* representation: it must round-trip datasheet values exactly
//! so that the Rust/Python parity suite can assert agreement to 1e-9. The
//! narrowing to `f32` happens once, when a volume is baked or a value crosses
//! into the shader.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::property::PropertyId;

const MASK_WORDS: usize = PropertyId::COUNT.div_ceil(64);

/// Where a property value came from. Drives the provenance badges in the app's
/// inspector and lets consumers decide how far to trust a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Source {
    /// No value present.
    #[default]
    Absent,
    /// Read directly from a curated datasheet.
    Datasheet,
    /// Computed from other present properties by an exact relation
    /// (see [`crate::derive`]).
    Derived,
    /// Produced by a semi-empirical model or a correlation.
    Estimated,
    /// Set by a designer in the app; never overwritten by ingest or derivation.
    UserOverride,
}

impl Source {
    /// Whether a new value from `incoming` may replace an existing `self`.
    ///
    /// A designer's explicit choice outranks everything; a measured datasheet
    /// value outranks anything we computed. This is what stops
    /// [`crate::derive`] clobbering real data with a correlation.
    #[inline]
    pub fn outranks(self, incoming: Source) -> bool {
        self.rank() > incoming.rank()
    }

    #[inline]
    const fn rank(self) -> u8 {
        match self {
            Source::Absent => 0,
            Source::Estimated => 1,
            Source::Derived => 2,
            Source::Datasheet => 3,
            Source::UserOverride => 4,
        }
    }
}

/// A dense, SI-normalised property table.
#[derive(Clone)]
pub struct PropertyTable {
    vals: [f64; PropertyId::COUNT],
    present: [u64; MASK_WORDS],
    source: [Source; PropertyId::COUNT],
}

impl Default for PropertyTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyTable {
    /// An empty table: every property absent.
    pub fn new() -> Self {
        Self {
            vals: [0.0; PropertyId::COUNT],
            present: [0; MASK_WORDS],
            source: [Source::Absent; PropertyId::COUNT],
        }
    }

    #[inline]
    fn mask_of(i: usize) -> (usize, u64) {
        (i / 64, 1u64 << (i % 64))
    }

    /// Whether `prop` has a value.
    #[inline]
    pub fn has(&self, prop: PropertyId) -> bool {
        let (w, b) = Self::mask_of(prop.index());
        self.present[w] & b != 0
    }

    /// The SI value of `prop`, if present.
    #[inline]
    pub fn get(&self, prop: PropertyId) -> Option<f64> {
        if self.has(prop) {
            Some(self.vals[prop.index()])
        } else {
            None
        }
    }

    /// The SI value of `prop`, or `fallback` when absent.
    ///
    /// This is the hot-path accessor: no branch on `Option`, no panic path.
    #[inline]
    pub fn get_or(&self, prop: PropertyId, fallback: f64) -> f64 {
        if self.has(prop) {
            self.vals[prop.index()]
        } else {
            fallback
        }
    }

    /// Provenance of `prop` ([`Source::Absent`] when it has no value).
    #[inline]
    pub fn source(&self, prop: PropertyId) -> Source {
        self.source[prop.index()]
    }

    /// Unconditionally set `prop`, overwriting any existing value.
    ///
    /// Non-finite values are rejected rather than stored: a NaN reaching the
    /// runtime would propagate silently through every downstream average.
    pub fn set(&mut self, prop: PropertyId, value: f64, source: Source) {
        if !value.is_finite() || source == Source::Absent {
            return;
        }
        let i = prop.index();
        let (w, b) = Self::mask_of(i);
        self.vals[i] = value;
        self.present[w] |= b;
        self.source[i] = source;
    }

    /// Fill `prop` if it is absent, or replace it if `source` is strictly more
    /// trustworthy than what is stored.
    ///
    /// Returns whether the write happened.
    ///
    /// The "strictly more trustworthy" part matters: an equal-rank write is
    /// refused, which makes this operation **monotone and therefore
    /// idempotent**. [`crate::derive`] iterates its rules to a fixpoint by
    /// counting writes, so an equal-rank overwrite returning `true` would spin
    /// that loop forever. It also gives ingest a deterministic first-wins rule
    /// when two datasheet spellings of one property appear in the same file.
    pub fn set_if_better(&mut self, prop: PropertyId, value: f64, source: Source) -> bool {
        let existing = self.source(prop);
        if existing != Source::Absent && !source.outranks(existing) {
            return false;
        }
        let had = self.has(prop);
        self.set(prop, value, source);
        // `set` rejects non-finite values, so confirm rather than assume.
        self.has(prop) && (!had || self.source(prop) == source)
    }

    /// Remove `prop`.
    pub fn clear(&mut self, prop: PropertyId) {
        let i = prop.index();
        let (w, b) = Self::mask_of(i);
        self.present[w] &= !b;
        self.source[i] = Source::Absent;
        self.vals[i] = 0.0;
    }

    /// Number of properties present.
    pub fn len(&self) -> usize {
        self.present.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether no property is present.
    pub fn is_empty(&self) -> bool {
        self.present.iter().all(|w| *w == 0)
    }

    /// Iterate every present property in `PropertyId` order.
    pub fn iter(&self) -> impl Iterator<Item = (PropertyId, f64, Source)> + '_ {
        PropertyId::ALL
            .iter()
            .copied()
            .filter(move |p| self.has(*p))
            .map(move |p| (p, self.vals[p.index()], self.source[p.index()]))
    }
}

impl std::fmt::Debug for PropertyTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut m = f.debug_map();
        for (p, v, s) in self.iter() {
            m.entry(&p.name(), &format_args!("{v} {} ({s:?})", p.si_unit()));
        }
        m.finish()
    }
}

impl PartialEq for PropertyTable {
    fn eq(&self, other: &Self) -> bool {
        // Only compare live entries; the value slot behind an absent property
        // is meaningless padding and must not affect equality.
        self.present == other.present
            && PropertyId::ALL.iter().all(|p| {
                !self.has(*p)
                    || (self.vals[p.index()] == other.vals[p.index()]
                        && self.source[p.index()] == other.source[p.index()])
            })
    }
}

// ── Serialisation ─────────────────────────────────────────────────────────
//
// Serialised as a name-keyed map rather than a positional array. Two reasons:
// a `.pmat` header stays readable and diffable, and adding a PropertyId
// variant cannot shift the meaning of previously written files.

/// One serialised entry: SI value plus provenance.
#[derive(Serialize, Deserialize)]
struct Entry {
    v: f64,
    #[serde(default, skip_serializing_if = "is_datasheet")]
    src: Source,
}

fn is_datasheet(s: &Source) -> bool {
    *s == Source::Datasheet
}

impl Default for Entry {
    fn default() -> Self {
        Entry {
            v: 0.0,
            src: Source::Datasheet,
        }
    }
}

impl Serialize for PropertyTable {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let map: BTreeMap<&'static str, Entry> = self
            .iter()
            .map(|(p, v, src)| (p.name(), Entry { v, src }))
            .collect();
        map.serialize(s)
    }
}

impl<'de> Deserialize<'de> for PropertyTable {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = BTreeMap::<String, Entry>::deserialize(d)?;
        let mut t = PropertyTable::new();
        for (k, e) in map {
            // Unknown property names are skipped, not rejected: a file written
            // by a newer periodica must stay loadable by an older one.
            if let Some(p) = PropertyId::from_name(&k) {
                let src = if e.src == Source::Absent {
                    Source::Datasheet
                } else {
                    e.src
                };
                t.set(p, e.v, src);
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_has_nothing() {
        let t = PropertyTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.get(PropertyId::Density), None);
        assert_eq!(t.get_or(PropertyId::Density, 1.5), 1.5);
        assert_eq!(t.source(PropertyId::Density), Source::Absent);
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::Density, 7870.0, Source::Datasheet);
        assert!(t.has(PropertyId::Density));
        assert_eq!(t.get(PropertyId::Density), Some(7870.0));
        assert_eq!(t.source(PropertyId::Density), Source::Datasheet);
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
    }

    #[test]
    fn mask_covers_every_property() {
        // Guards against MASK_WORDS falling behind PropertyId::COUNT.
        let mut t = PropertyTable::new();
        for (i, p) in PropertyId::ALL.iter().enumerate() {
            t.set(*p, i as f64 + 1.0, Source::Datasheet);
        }
        assert_eq!(t.len(), PropertyId::COUNT);
        for (i, p) in PropertyId::ALL.iter().enumerate() {
            assert_eq!(t.get(*p), Some(i as f64 + 1.0), "{p:?}");
        }
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::Density, f64::NAN, Source::Datasheet);
        t.set(PropertyId::YoungsModulus, f64::INFINITY, Source::Datasheet);
        assert!(!t.has(PropertyId::Density));
        assert!(!t.has(PropertyId::YoungsModulus));
    }

    #[test]
    fn derived_never_overwrites_datasheet() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::BulkModulus, 67.6e9, Source::Datasheet);
        let wrote = t.set_if_better(PropertyId::BulkModulus, 1.0e9, Source::Derived);
        assert!(!wrote);
        assert_eq!(t.get(PropertyId::BulkModulus), Some(67.6e9));
        assert_eq!(t.source(PropertyId::BulkModulus), Source::Datasheet);
    }

    #[test]
    fn user_override_outranks_everything() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::Density, 7870.0, Source::Datasheet);
        assert!(t.set_if_better(PropertyId::Density, 8000.0, Source::UserOverride));
        assert_eq!(t.get(PropertyId::Density), Some(8000.0));
        // And nothing may then take it back.
        assert!(!t.set_if_better(PropertyId::Density, 7870.0, Source::Datasheet));
        assert_eq!(t.get(PropertyId::Density), Some(8000.0));
    }

    #[test]
    fn derived_fills_only_absent_slots() {
        let mut t = PropertyTable::new();
        assert!(t.set_if_better(PropertyId::ShearModulus, 26.0e9, Source::Derived));
        assert_eq!(t.source(PropertyId::ShearModulus), Source::Derived);
    }

    #[test]
    fn clear_removes_the_entry() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::Density, 7870.0, Source::Datasheet);
        t.clear(PropertyId::Density);
        assert!(!t.has(PropertyId::Density));
        assert_eq!(t.len(), 0);
        assert_eq!(t.source(PropertyId::Density), Source::Absent);
    }

    #[test]
    fn iter_yields_present_entries_in_order() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::YoungsModulus, 2.0e11, Source::Datasheet);
        t.set(PropertyId::Density, 7870.0, Source::Datasheet);
        let got: Vec<_> = t.iter().map(|(p, _, _)| p).collect();
        // Density is declared before YoungsModulus, so it must come first.
        assert_eq!(got, vec![PropertyId::Density, PropertyId::YoungsModulus]);
    }

    #[test]
    fn json_round_trip_preserves_values_and_provenance() {
        let mut t = PropertyTable::new();
        t.set(PropertyId::Density, 7870.0, Source::Datasheet);
        t.set(PropertyId::ShearModulus, 79.3e9, Source::Derived);
        t.set(PropertyId::YieldStrength, 3.7e8, Source::UserOverride);

        let s = serde_json::to_string(&t).unwrap();
        let back: PropertyTable = serde_json::from_str(&s).unwrap();
        assert_eq!(t, back);
        assert_eq!(back.source(PropertyId::ShearModulus), Source::Derived);
        assert_eq!(back.source(PropertyId::YieldStrength), Source::UserOverride);
        // Readable, name-keyed output.
        assert!(s.contains("YoungsModulus") || s.contains("Density"));
    }

    #[test]
    fn unknown_property_names_are_skipped_not_rejected() {
        // Forward compatibility: a file from a newer periodica must load.
        let json = r#"{"Density":{"v":7870.0},"SomeFutureProperty":{"v":1.0}}"#;
        let t: PropertyTable = serde_json::from_str(json).unwrap();
        assert_eq!(t.get(PropertyId::Density), Some(7870.0));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn equality_ignores_padding_behind_absent_slots() {
        let mut a = PropertyTable::new();
        let mut b = PropertyTable::new();
        a.set(PropertyId::Density, 1.0, Source::Datasheet);
        b.set(PropertyId::Density, 1.0, Source::Datasheet);
        // Write and clear on `b` only: the value slot keeps stale bytes.
        b.set(PropertyId::YoungsModulus, 42.0, Source::Datasheet);
        b.clear(PropertyId::YoungsModulus);
        assert_eq!(a, b);
    }
}
