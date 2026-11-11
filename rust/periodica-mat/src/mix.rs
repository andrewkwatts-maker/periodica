//! Mixing rules: the property of a blend of materials from the properties of
//! its parts.
//!
//! Each [`PropertyId`] has one rule, chosen by the physics of the property:
//!
//! | Property | Rule |
//! |---|---|
//! | density, porosity | volume-fraction average |
//! | thermal expansion | Turner, `sum(alpha K V) / sum(K V)`, when every part has a bulk modulus (given or from E and nu); else the volume average |
//! | refractive index n | Lorentz-Lorenz over volume fractions, when no part has a positive extinction coefficient k |
//! | surface energy | fraction-weighted average; the caller supplies atom fractions, as the surface is counted in atoms |
//! | specific heat (per mass) | mass-weighted average (the volume average of heat capacity per volume, divided by the mixed density) |
//! | shear, bulk modulus | Hill average, the mean of the Voigt (arithmetic) and Reuss (harmonic) bounds |
//! | Young modulus, Poisson ratio | from the Hill K and G, `E = 9KG / (3K + G)`, `nu = (3K - 2G) / (2 (3K + G))`, so the isotropic identities hold exactly; a part's K and G come from its E and nu when it has both (`-1 < nu < 0.5`, the primary pair), else from its own K and G. When a part has neither: Hill on E, volume average of nu |
//! | thermal and electrical conductivity | harmonic (series) average, a lower bound; a zero part gives zero |
//! | strengths, toughness, fracture energy, fatigue | volume average. A documented estimate, not a bound: real strength depends on microstructure |
//!
//! [`mix_tagged`] tags density and porosity `Derived` (exact); every other rule is `Estimated`.
//!
//! A mix of one part is the part: when every entry with a positive fraction is the same
//! record (the same reference, or records equal under `==`), [`mix_tagged`] returns that
//! record's stored value and source for any property, with or without a mixing rule, and
//! [`MixError::Missing`] when it lacks the property. E and nu are the primary pair, so the
//! mix is continuous as a part's fraction tends to 1.
//!
//! Any other property is not mixable ([`MixError::NotMixable`]).
//! Fractions are volume fractions and must sum to 1 within [`FRACTION_TOLERANCE`].
//!
//! direct-float: periodica-mat sits below Arithma and has no dependency on it,
//! so these rules are plain `f64`.

use crate::property::PropertyId;
use crate::record::MaterialRecord;
use crate::table::Source;

/// How far from 1 the fractions may sum.
pub const FRACTION_TOLERANCE: f64 = 1.0e-9;

/// Why a mix could not be computed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MixError {
    /// No entries were given.
    #[error("mix has no entries")]
    Empty,
    /// A fraction is negative or not finite.
    #[error("fraction {0} of `{1}` is not a finite value in 0..=1")]
    BadFraction(f64, String),
    /// The fractions do not sum to 1.
    #[error("fractions sum to {0}, not 1")]
    NotNormalised(f64),
    /// A material with a non-zero fraction lacks the property.
    #[error("material `{material}` has no {property:?}")]
    Missing {
        /// The material's name.
        material: String,
        /// The property that is absent.
        property: PropertyId,
    },
    /// No mixing rule exists for the property.
    #[error("{0:?} has no mixing rule")]
    NotMixable(PropertyId),
}

/// Result of a mix.
pub type MixResult<T> = Result<T, MixError>;

/// Materials with their volume fractions.
pub type Entries<'a> = &'a [(&'a MaterialRecord, f64)];

/// The `(Voigt, Reuss)` bounds of `prop` over `entries`: the arithmetic and
/// harmonic volume averages.
///
/// # Errors
/// An empty list, a bad fraction, fractions not summing to 1, or a missing property.
pub fn bounds(entries: Entries<'_>, prop: PropertyId) -> MixResult<(f64, f64)> {
    let parts = gather(entries, prop)?;
    Ok((voigt(&parts), reuss(&parts)))
}

/// The property `prop` of the mixture of `entries`, by the rule in the module docs.
///
/// # Errors
/// An empty list, a bad fraction, fractions not summing to 1, a missing
/// property, or a property with no mixing rule.
pub fn mix(entries: Entries<'_>, prop: PropertyId) -> MixResult<f64> {
    mix_tagged(entries, prop).map(|(v, _)| v)
}

/// Like [`mix`], with the provenance of the result. `Derived` only for the
/// exact rules, density and porosity (ideal volume-fraction sums); every other
/// rule (Hill, Turner, Lorentz-Lorenz, bounds, volume-average strengths, the
/// volume-average fallbacks) is a model and is tagged `Estimated`.
///
/// # Errors
/// As for [`mix`].
pub fn mix_tagged(entries: Entries<'_>, prop: PropertyId) -> MixResult<(f64, Source)> {
    use PropertyId as P;
    if let Some(rec) = single_record(entries)? {
        return stored(rec, prop);
    }
    if prop == P::ThermalExpansion {
        return thermal_expansion(entries).map(|v| (v, rule_source(prop)));
    }
    if prop == P::YoungsModulus || prop == P::PoissonsRatio {
        if let Some(v) = elastic_from_hill(entries, prop)? {
            return Ok((v, rule_source(prop)));
        }
    }
    mix_plain(entries, prop).map(|v| (v, rule_source(prop)))
}

/// The one record every positive-fraction entry shares (the same reference, or
/// records that compare equal), after the fractions are checked.
fn single_record<'a>(entries: Entries<'a>) -> MixResult<Option<&'a MaterialRecord>> {
    check_fractions(entries)?;
    let mut parts = entries.iter().filter(|(_, f)| *f > 0.0).map(|(r, _)| *r);
    let Some(first) = parts.next() else {
        return Ok(None);
    };
    let same = parts.all(|r| std::ptr::eq(r, first) || r == first);
    Ok(same.then_some(first))
}

/// A record's stored value and source for `prop`.
fn stored(rec: &MaterialRecord, prop: PropertyId) -> MixResult<(f64, Source)> {
    let v = rec.bulk.get(prop).ok_or_else(|| MixError::Missing {
        material: rec.name.clone(),
        property: prop,
    })?;
    Ok((v, rec.bulk.source(prop)))
}

/// The provenance of a property's mixing rule. Exhaustive on the exact rules:
/// a new exact rule must be listed here, anything else is a model.
fn rule_source(prop: PropertyId) -> Source {
    match prop {
        PropertyId::Density | PropertyId::Porosity => Source::Derived,
        _ => Source::Estimated,
    }
}

fn mix_plain(entries: Entries<'_>, prop: PropertyId) -> MixResult<f64> {
    use PropertyId as P;
    match prop {
        P::RefractiveIndexN => lorentz_lorenz(entries),
        P::Density
        | P::Porosity
        | P::ThermalExpansion
        | P::StrainHardeningExponent
        | P::SurfaceEnergy => volume_average(entries, prop),
        P::SpecificHeat => mass_weighted(entries, prop),
        P::YoungsModulus | P::ShearModulus | P::BulkModulus => hill(entries, prop),
        P::PoissonsRatio => poisson(entries),
        P::ThermalConductivity | P::ElectricalConductivity => {
            let parts = gather(entries, prop)?;
            Ok(reuss(&parts))
        }
        p if is_strength_like(p) => volume_average(entries, prop),
        p => Err(MixError::NotMixable(p)),
    }
}

fn is_strength_like(p: PropertyId) -> bool {
    use PropertyId as P;
    matches!(
        p,
        P::YieldStrength
            | P::UltimateTensileStrength
            | P::CompressiveStrength
            | P::CompressiveYieldStrength
            | P::ShearStrength
            | P::FlexuralStrength
            | P::FractureToughnessKic
            | P::FractureEnergy
            | P::FatigueStrength
            | P::FatigueLimit
    )
}

/// Fractions checked and values read, zero-fraction entries dropped.
fn gather(entries: Entries<'_>, prop: PropertyId) -> MixResult<Vec<(f64, f64)>> {
    check_fractions(entries)?;
    let mut out = Vec::with_capacity(entries.len());
    for (rec, f) in entries.iter().filter(|(_, f)| *f > 0.0) {
        let v = rec.bulk.get(prop).ok_or_else(|| MixError::Missing {
            material: rec.name.clone(),
            property: prop,
        })?;
        out.push((*f, v));
    }
    Ok(out)
}

fn check_fractions(entries: Entries<'_>) -> MixResult<()> {
    if entries.is_empty() {
        return Err(MixError::Empty);
    }
    let mut sum = 0.0;
    for (rec, f) in entries {
        if !f.is_finite() || *f < 0.0 {
            return Err(MixError::BadFraction(*f, rec.name.clone()));
        }
        sum += f;
    }
    if (sum - 1.0).abs() > FRACTION_TOLERANCE {
        return Err(MixError::NotNormalised(sum));
    }
    Ok(())
}

fn voigt(parts: &[(f64, f64)]) -> f64 {
    parts.iter().map(|(f, v)| f * v).sum()
}

fn reuss(parts: &[(f64, f64)]) -> f64 {
    if parts.iter().any(|(_, v)| *v <= 0.0) {
        return 0.0;
    }
    1.0 / parts.iter().map(|(f, v)| f / v).sum::<f64>()
}

fn volume_average(entries: Entries<'_>, prop: PropertyId) -> MixResult<f64> {
    Ok(voigt(&gather(entries, prop)?))
}

fn hill(entries: Entries<'_>, prop: PropertyId) -> MixResult<f64> {
    let parts = gather(entries, prop)?;
    Ok(0.5 * (voigt(&parts) + reuss(&parts)))
}

fn mass_weighted(entries: Entries<'_>, prop: PropertyId) -> MixResult<f64> {
    let cp = gather(entries, prop)?;
    let rho = gather(entries, PropertyId::Density)?;
    let mass: f64 = rho.iter().map(|(f, r)| f * r).sum();
    let heat: f64 = cp.iter().zip(&rho).map(|((f, c), (_, r))| f * r * c).sum();
    Ok(heat / mass)
}

/// `(K, G)` of a part: from E and nu when both are present and `-1 < nu < 0.5`
/// (the primary pair), else its own K and G, else `None`.
fn kg_of(rec: &MaterialRecord) -> Option<(f64, f64)> {
    use PropertyId as P;
    let b = &rec.bulk;
    if let (Some(e), Some(nu)) = (b.get(P::YoungsModulus), b.get(P::PoissonsRatio)) {
        if nu > -1.0 && nu < 0.5 {
            return Some((e / (3.0 * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu))));
        }
    }
    Some((b.get(P::BulkModulus)?, b.get(P::ShearModulus)?))
}

/// Bulk modulus of a part, given or derived.
fn k_of(rec: &MaterialRecord) -> Option<f64> {
    rec.bulk
        .get(PropertyId::BulkModulus)
        .or_else(|| kg_of(rec).map(|(k, _)| k))
}

/// Hill-averaged `(K, G)` when every part has both.
fn hill_kg(entries: Entries<'_>) -> MixResult<Option<(f64, f64)>> {
    check_fractions(entries)?;
    let mut kp = Vec::new();
    let mut gp = Vec::new();
    for (rec, f) in entries.iter().filter(|(_, f)| *f > 0.0) {
        let Some((k, g)) = kg_of(rec) else {
            return Ok(None);
        };
        kp.push((*f, k));
        gp.push((*f, g));
    }
    let h = |p: &[(f64, f64)]| 0.5 * (voigt(p) + reuss(p));
    Ok(Some((h(&kp), h(&gp))))
}

/// E or nu from the Hill K and G; `None` when a part lacks the moduli.
fn elastic_from_hill(entries: Entries<'_>, prop: PropertyId) -> MixResult<Option<f64>> {
    Ok(hill_kg(entries)?.map(|(k, g)| {
        if prop == PropertyId::YoungsModulus {
            9.0 * k * g / (3.0 * k + g)
        } else {
            (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))
        }
    }))
}

fn poisson(entries: Entries<'_>) -> MixResult<f64> {
    match elastic_from_hill(entries, PropertyId::PoissonsRatio)? {
        Some(nu) => Ok(nu),
        None => volume_average(entries, PropertyId::PoissonsRatio),
    }
}

/// Turner's thermal expansion when every part has a bulk modulus, else the
/// volume average.
fn thermal_expansion(entries: Entries<'_>) -> MixResult<f64> {
    let alpha = gather(entries, PropertyId::ThermalExpansion)?;
    let ks: Option<Vec<f64>> = entries
        .iter()
        .filter(|(_, f)| *f > 0.0)
        .map(|(r, _)| k_of(r))
        .collect();
    match ks {
        Some(ks) => {
            let num: f64 = alpha.iter().zip(&ks).map(|((f, a), k)| a * k * f).sum();
            let den: f64 = alpha.iter().zip(&ks).map(|((f, _), k)| k * f).sum();
            Ok(num / den)
        }
        None => Ok(voigt(&alpha)),
    }
}

/// Lorentz-Lorenz mixing of refractive index for transparent parts.
fn lorentz_lorenz(entries: Entries<'_>) -> MixResult<f64> {
    let n = gather(entries, PropertyId::RefractiveIndexN)?;
    let absorbing = entries
        .iter()
        .filter(|(_, f)| *f > 0.0)
        .any(|(r, _)| r.bulk.get(PropertyId::ExtinctionK).is_some_and(|k| k > 0.0));
    if absorbing {
        return Err(MixError::NotMixable(PropertyId::RefractiveIndexN));
    }
    let ll: f64 = n
        .iter()
        .map(|(f, n)| f * (n * n - 1.0) / (n * n + 2.0))
        .sum();
    Ok(((1.0 + 2.0 * ll) / (1.0 - ll)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Source;

    fn rec(name: &str, props: &[(PropertyId, f64)]) -> MaterialRecord {
        let mut r = MaterialRecord {
            name: name.into(),
            ..Default::default()
        };
        for (p, v) in props {
            r.bulk.set(*p, *v, Source::Datasheet);
        }
        r
    }

    fn pair() -> (MaterialRecord, MaterialRecord) {
        use PropertyId as P;
        let a = rec(
            "a",
            &[
                (P::Density, 8000.0),
                (P::YoungsModulus, 200e9),
                (P::ThermalConductivity, 50.0),
            ],
        );
        let b = rec(
            "b",
            &[
                (P::Density, 2000.0),
                (P::YoungsModulus, 50e9),
                (P::ThermalConductivity, 150.0),
            ],
        );
        (a, b)
    }

    #[test]
    fn density_is_volume_average_and_modulus_is_between_bounds() {
        let (a, b) = pair();
        let e = [(&a, 0.5), (&b, 0.5)];
        assert!((mix(&e, PropertyId::Density).unwrap() - 5000.0).abs() < 1e-9);
        let (v, r) = bounds(&e, PropertyId::YoungsModulus).unwrap();
        let h = mix(&e, PropertyId::YoungsModulus).unwrap();
        assert!(r < h && h < v);
    }

    #[test]
    fn strain_hardening_exponent_is_volume_weighted() {
        use PropertyId as P;
        let a = rec("a", &[(P::StrainHardeningExponent, 0.2)]);
        let b = rec("b", &[(P::StrainHardeningExponent, 0.05)]);
        let n = mix(&[(&a, 0.25), (&b, 0.75)], P::StrainHardeningExponent).unwrap();
        assert!((n - 0.0875).abs() < 1e-12);
        let c = rec("c", &[(P::Density, 1.0)]);
        assert!(mix(&[(&a, 0.5), (&c, 0.5)], P::StrainHardeningExponent).is_err());
    }

    #[test]
    fn conductivity_is_harmonic() {
        let (a, b) = pair();
        let k = mix(&[(&a, 0.5), (&b, 0.5)], PropertyId::ThermalConductivity).unwrap();
        assert!((k - 75.0).abs() < 1e-9);
    }

    #[test]
    fn bad_fractions_and_unmixable_fail() {
        let (a, b) = pair();
        let id = PropertyId::Density;
        assert!(matches!(
            mix(&[(&a, 0.5), (&b, 0.4)], id),
            Err(MixError::NotNormalised(_))
        ));
        assert!(matches!(
            mix(&[(&a, 1.5), (&b, -0.5)], id),
            Err(MixError::BadFraction(..))
        ));
        assert!(matches!(mix(&[], id), Err(MixError::Empty)));
        let melt = PropertyId::MeltingPoint;
        assert!(matches!(
            mix(&[(&a, 0.5), (&b, 0.5)], melt),
            Err(MixError::NotMixable(_))
        ));
        let ys = PropertyId::YieldStrength;
        assert!(matches!(
            mix(&[(&a, 1.0)], ys),
            Err(MixError::Missing { .. })
        ));
    }
}
