//! Mixing rules: the property of a blend of materials from the properties of
//! its parts.
//!
//! Each [`PropertyId`] has one rule, chosen by the physics of the property:
//!
//! | Property | Rule |
//! |---|---|
//! | density, porosity, thermal expansion | volume-fraction average |
//! | specific heat (per mass) | mass-weighted average (the volume average of heat capacity per volume, divided by the mixed density) |
//! | Young, shear, bulk modulus | Hill average, the mean of the Voigt (arithmetic) and Reuss (harmonic) bounds |
//! | Poisson ratio | from the Hill bulk and shear moduli, `(3K - 2G) / (2 (3K + G))`; volume average when either modulus is missing |
//! | thermal and electrical conductivity | harmonic (series) average, a lower bound; a zero part gives zero |
//! | strengths, toughness, fracture energy, fatigue | volume average. A documented estimate, not a bound: real strength depends on microstructure |
//!
//! Any other property is not mixable ([`MixError::NotMixable`]).
//! Fractions are volume fractions and must sum to 1 within [`FRACTION_TOLERANCE`].
//!
//! direct-float: periodica-mat sits below Arithma and has no dependency on it,
//! so these rules are plain `f64`.

use crate::property::PropertyId;
use crate::record::MaterialRecord;

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
    use PropertyId as P;
    match prop {
        P::Density | P::Porosity | P::ThermalExpansion | P::StrainHardeningExponent => {
            volume_average(entries, prop)
        }
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

fn poisson(entries: Entries<'_>) -> MixResult<f64> {
    let km = hill(entries, PropertyId::BulkModulus);
    let gm = hill(entries, PropertyId::ShearModulus);
    match (km, gm) {
        (Ok(k), Ok(g)) => Ok((3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))),
        _ => volume_average(entries, PropertyId::PoissonsRatio),
    }
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
            mix(&[(&a, 1.0)], melt),
            Err(MixError::NotMixable(_))
        ));
        let ys = PropertyId::YieldStrength;
        assert!(matches!(
            mix(&[(&a, 1.0)], ys),
            Err(MixError::Missing { .. })
        ));
    }
}
