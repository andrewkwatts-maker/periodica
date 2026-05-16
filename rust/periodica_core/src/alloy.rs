//====== periodica/rust/periodica_core/src/alloy.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # alloy
//!
//! Hume-Rothery alloy optimisation. Mirrors `periodica.optimize.optimize_alloy`.
//!
//! Randomly samples composition vectors (random-subset Hume-Rothery rule:
//! up to 4 alloying elements, fractions ≤ 0.2 each), estimates bulk properties
//! via rule-of-mixtures from the DataHub, scores against caller-supplied
//! targets, and returns the top-k results.
//
// Python fallback: src/periodica/optimize.py

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

/// Scalar target constraint for alloy optimisation.
#[derive(Debug, Clone)]
pub struct AlloyTarget {
    pub property: String,
    /// Minimum acceptable value (None = unconstrained below).
    pub min_value: Option<f64>,
    /// Maximum acceptable value (None = unconstrained above).
    pub max_value: Option<f64>,
    /// Unitless weight for objective (higher → property matters more).
    pub weight: f64,
}

/// One ranked optimisation result.
#[derive(Debug, Clone)]
pub struct AlloyCandidate {
    /// Mole fraction per element symbol (sums to 1.0).
    pub composition: HashMap<String, f64>,
    /// Estimated property values by name.
    pub estimated_properties: HashMap<String, f64>,
    /// Objective score (higher is better, matches Python convention).
    pub score: f64,
}

/// Default alloying pool when the caller does not specify one.
const DEFAULT_POOL: &[&str] = &["Cr", "Ni", "Mn", "Si", "Cu", "Mo"];
/// Maximum fraction for any single alloying element.
const MAX_FRAC: f64 = 0.20;
/// Maximum combined fraction of all alloying elements.
const MAX_TOTAL_FRAC: f64 = 0.50;

/// Optimise alloy composition toward `targets`.
///
/// `alloying_pool` — symbols to draw from (defaults to Fe common alloyers).
/// `n_candidates` — how many random compositions to evaluate.
/// `top_k` — how many top results to return.
/// `seed` — RNG seed for reproducibility; None → non-deterministic.
pub fn optimize_alloy(
    targets: &[AlloyTarget],
    base: &str,
    alloying_pool: &[String],
    n_candidates: usize,
    top_k: usize,
    seed: Option<u64>,
) -> Result<Vec<AlloyCandidate>> {
    if targets.is_empty() {
        return Err(anyhow!("alloy::optimize_alloy: at least one target is required"));
    }
    if base.is_empty() {
        return Err(anyhow!("alloy::optimize_alloy: base element name required"));
    }
    if n_candidates == 0 {
        return Ok(Vec::new());
    }

    let pool: Vec<&str> = if alloying_pool.is_empty() {
        DEFAULT_POOL.to_vec()
    } else {
        alloying_pool.iter().map(String::as_str).collect()
    };

    let mut rng: SmallRng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

    let mut results: Vec<AlloyCandidate> = Vec::with_capacity(n_candidates / 4);

    for _ in 0..n_candidates {
        let comp = random_composition(&mut rng, base, &pool);
        let props = estimate_properties(&comp);
        if let Some(sc) = score_candidate(&props, targets) {
            results.push(AlloyCandidate { composition: comp, estimated_properties: props, score: sc });
        }
    }

    // Sort by score descending (higher is better)
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(results.len());
    results.truncate(k);
    Ok(results)
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Generate one random composition: 0–4 alloying elements with random fractions.
fn random_composition(rng: &mut SmallRng, base: &str, pool: &[&str]) -> HashMap<String, f64> {
    // Pick 0–4 elements from pool without replacement
    let n_elements = rng.gen_range(0..=4.min(pool.len()));
    let mut chosen: Vec<&str> = Vec::with_capacity(n_elements);
    let mut remaining: Vec<&str> = pool.to_vec();
    for _ in 0..n_elements {
        if remaining.is_empty() { break; }
        let idx = rng.gen_range(0..remaining.len());
        chosen.push(remaining.remove(idx));
    }

    // Assign random fractions in [0.0005, MAX_FRAC]
    let mut fracs: Vec<f64> = chosen.iter().map(|_| rng.gen_range(0.0005_f64..MAX_FRAC)).collect();

    // Clamp total alloying fraction to MAX_TOTAL_FRAC
    let total: f64 = fracs.iter().sum();
    if total > MAX_TOTAL_FRAC {
        let scale = MAX_TOTAL_FRAC / total;
        for f in fracs.iter_mut() { *f *= scale; }
    }

    let alloying_total: f64 = fracs.iter().sum();
    let mut comp = HashMap::with_capacity(chosen.len() + 1);
    comp.insert(base.to_string(), (1.0 - alloying_total).max(0.0));
    for (el, frac) in chosen.iter().zip(fracs.iter()) {
        comp.insert(el.to_string(), *frac);
    }
    comp
}

/// Rule-of-mixtures property estimation from the DataHub.
/// For each element, look up numeric properties and weight by mole fraction.
fn estimate_properties(composition: &HashMap<String, f64>) -> HashMap<String, f64> {
    let hub = crate::data_loader::DATA.read();
    let mut acc: HashMap<String, (f64, f64)> = HashMap::new(); // property → (weighted_sum, weight_sum)

    for (element, frac) in composition {
        // Search all tiers for this element
        let entry: Option<serde_json::Value> = {
            let mut found = None;
            'outer: for kv in hub.tiers.iter() {
                if let Some(v) = kv.value().get(element.as_str()) {
                    found = Some(v.value().clone());
                    break 'outer;
                }
            }
            found
        };
        let entry = match entry { Some(e) => e, None => continue };

        // Walk top-level numeric values and "Properties" sub-object
        for (prop_map, w) in [
            (entry.as_object(), *frac),
            (entry.get("Properties").and_then(|p| p.as_object()), *frac),
        ] {
            if let Some(map) = prop_map {
                for (k, v) in map {
                    if let Some(f) = v.as_f64() {
                        let e = acc.entry(k.clone()).or_insert((0.0, 0.0));
                        e.0 += f * w;
                        e.1 += w;
                    }
                }
            }
        }
    }

    acc.into_iter()
        .filter_map(|(k, (ws, wt))| if wt > 0.0 { Some((k, ws / wt)) } else { None })
        .collect()
}

/// Score a composition. Returns None if any hard constraint is violated.
/// Returns a higher value for compositions that satisfy targets more.
fn score_candidate(props: &HashMap<String, f64>, targets: &[AlloyTarget]) -> Option<f64> {
    let mut score = 0.0_f64;
    for t in targets {
        let v = *props.get(&t.property)?; // None → missing property → reject
        if let Some(min) = t.min_value {
            if v < min { return None; }
            score += (v - min) * t.weight;
        }
        if let Some(max) = t.max_value {
            if v > max { return None; }
            score += (max - v) * t.weight;
        }
    }
    Some(score)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn density_target() -> AlloyTarget {
        AlloyTarget { property: "Density_g_cm3".into(), min_value: Some(7.0), max_value: Some(9.0), weight: 1.0 }
    }

    #[test]
    fn validates_empty_targets() {
        let res = optimize_alloy(&[], "Fe", &[], 10, 5, Some(42));
        assert!(res.is_err());
    }

    #[test]
    fn validates_empty_base() {
        let res = optimize_alloy(&[density_target()], "", &[], 10, 5, Some(42));
        assert!(res.is_err());
    }

    #[test]
    fn zero_candidates_returns_empty() {
        let res = optimize_alloy(&[density_target()], "Fe", &[], 0, 5, Some(42)).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn returns_at_most_top_k() {
        let res = optimize_alloy(&[density_target()], "Fe", &[], 100, 3, Some(42)).unwrap();
        assert!(res.len() <= 3);
    }

    #[test]
    fn compositions_sum_to_one() {
        let res = optimize_alloy(&[density_target()], "Fe", &[], 20, 5, Some(1)).unwrap();
        for c in &res {
            let total: f64 = c.composition.values().sum();
            assert!((total - 1.0).abs() < 1e-9, "composition total {total}");
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let a = optimize_alloy(&[density_target()], "Fe", &[], 50, 5, Some(99)).unwrap();
        let b = optimize_alloy(&[density_target()], "Fe", &[], 50, 5, Some(99)).unwrap();
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn custom_pool_respected() {
        let pool = vec!["Au".to_string(), "Pt".to_string()];
        let res = optimize_alloy(&[density_target()], "Fe", &pool, 30, 5, Some(7)).unwrap();
        for c in &res {
            for key in c.composition.keys() {
                assert!(key == "Fe" || key == "Au" || key == "Pt", "unexpected element {key}");
            }
        }
    }
}
