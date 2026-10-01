//====== periodica/rust/periodica-mat/src/record.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # record
//!
//! [`MaterialRecord`] -- the canonical, SI-normalised view of one datasheet.
//!
//! Everything here is populated from data that **already exists** in the
//! bundled JSON. In particular [`Microstructure`] mirrors the `Microstructure`
//! block authored in 68 alloy/material datasheets that no code previously
//! read; it carries exactly the grain, noise, defect and inclusion parameters
//! the runtime's field models and fracture graph need.
//!
//! This type does not replace `serde_json::Value` in `periodica_core`. The
//! registry still stores raw JSON so that non-material tiers (proteins, cells,
//! quarks) keep working unchanged; `MaterialRecord` is an *ingest view* over
//! that, produced on demand by [`crate::ingest`].

use serde::{Deserialize, Serialize};

use crate::table::PropertyTable;

/// Which periodica tier a record came from. Mirrors the on-disk folder names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Element,
    Molecule,
    Alloy,
    Ceramic,
    Composite,
    Polymer,
    Material,
    BiologicalMaterial,
    #[default]
    Unknown,
}

/// Crystal system / packing of the primary phase.
///
/// This is what makes fracture come out *planar*: [`CrystalSystem::cleavage_planes`]
/// supplies the low-index planes a brittle crack prefers to follow, which the
/// runtime's fracture edge cost weights against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CrystalSystem {
    /// Body-centred cubic. Cleaves readily on {100} -- the classic brittle
    /// transgranular fracture of ferritic steel below its DBTT.
    Bcc,
    /// Face-centred cubic. Has no cleavage plane: FCC metals (Al, Cu,
    /// austenitic steel) deform plastically instead, so brittle cracks are
    /// forced along grain boundaries.
    Fcc,
    /// Hexagonal close-packed. Cleaves on the {0001} basal plane.
    Hcp,
    /// Simple cubic.
    SimpleCubic,
    /// Diamond cubic. Cleaves on {111} (silicon, germanium).
    DiamondCubic,
    Tetragonal,
    Orthorhombic,
    Rhombohedral,
    Monoclinic,
    Triclinic,
    /// Glass, most polymers -- no lattice, therefore no preferred plane.
    Amorphous,
    #[default]
    Unknown,
}

impl CrystalSystem {
    /// Parse the `LatticeProperties.PrimaryStructure` string used on disk.
    pub fn parse(s: &str) -> Self {
        let c = crate::canon::canon(s);
        match c.as_str() {
            "bcc" | "bodycenteredcubic" | "bodycentredcubic" => Self::Bcc,
            "fcc" | "facecenteredcubic" | "facecentredcubic" | "ccp" => Self::Fcc,
            "hcp" | "hexagonalclosepacked" | "hexagonal" => Self::Hcp,
            // Double-HCP (ABAC stacking; La, Pr, Nd, Am...) is a close-packed
            // hexagonal polytype with the same {0001} basal cleavage, so it
            // maps onto Hcp rather than adding a variant every consumer of
            // this enum would have to learn.
            "dhcp" | "doublehexagonalclosepacked" => Self::Hcp,
            "sc" | "simplecubic" | "cubic" | "primitivecubic" => Self::SimpleCubic,
            "diamond" | "diamondcubic" => Self::DiamondCubic,
            "tetragonal" | "bct" | "fct" => Self::Tetragonal,
            "orthorhombic" => Self::Orthorhombic,
            "rhombohedral" | "trigonal" => Self::Rhombohedral,
            "monoclinic" => Self::Monoclinic,
            "triclinic" => Self::Triclinic,
            "amorphous" | "glass" | "glassy" | "noncrystalline" => Self::Amorphous,
            _ => Self::Unknown,
        }
    }

    /// Unit normals of the preferred cleavage planes, in lattice axes.
    ///
    /// An empty slice means "no cleavage": the material has no brittle
    /// transgranular path, so cracks must run intergranularly. That is the
    /// physically correct answer for FCC metals and for amorphous solids, not
    /// a missing-data placeholder.
    pub fn cleavage_planes(self) -> &'static [[f32; 3]] {
        const SQRT3_INV: f32 = 0.577_350_3;
        match self {
            // {100} family.
            Self::Bcc | Self::SimpleCubic => &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            // {0001} basal only.
            Self::Hcp => &[[0.0, 0.0, 1.0]],
            // {111} family.
            Self::DiamondCubic => &[
                [SQRT3_INV, SQRT3_INV, SQRT3_INV],
                [-SQRT3_INV, SQRT3_INV, SQRT3_INV],
                [SQRT3_INV, -SQRT3_INV, SQRT3_INV],
                [SQRT3_INV, SQRT3_INV, -SQRT3_INV],
            ],
            // Ductile or structureless: no preferred plane.
            Self::Fcc | Self::Amorphous | Self::Unknown => &[],
            // Low-symmetry systems do cleave, but the plane depends on the
            // specific compound; without per-material data, decline to guess.
            _ => &[],
        }
    }

    /// Whether brittle transgranular cleavage is available at all.
    #[inline]
    pub fn is_cleavable(self) -> bool {
        !self.cleavage_planes().is_empty()
    }
}

/// Unit-cell geometry. Lengths in metres, angles in radians (SI, like
/// everything else) even though the datasheets store pm and degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Lattice {
    pub system: CrystalSystem,
    /// Cell edge lengths `a`, `b`, `c` in metres.
    pub abc_m: [f64; 3],
    /// Cell angles alpha, beta, gamma in radians.
    pub angles_rad: [f64; 3],
    pub packing_factor: Option<f64>,
    pub coordination_number: Option<u32>,
    pub atoms_per_cell: Option<u32>,
    pub space_group: Option<u32>,
    /// Burgers vector magnitude in metres -- sets the dislocation length scale.
    pub burgers_vector_m: Option<f64>,
}

impl Default for Lattice {
    fn default() -> Self {
        Self {
            system: CrystalSystem::Unknown,
            abc_m: [0.0; 3],
            angles_rad: [std::f64::consts::FRAC_PI_2; 3],
            packing_factor: None,
            coordination_number: None,
            atoms_per_cell: None,
            space_group: None,
            burgers_vector_m: None,
        }
    }
}

/// One phase of a multi-phase material (ferrite, pearlite, cementite, ...).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(default)]
    pub structure: CrystalSystem,
    /// Volume fraction in `0..=1` (the datasheets store percent).
    pub volume_fraction: f64,
    #[serde(default)]
    pub magnetic: bool,
    /// Per-phase property overrides. Anything absent falls back to bulk.
    #[serde(default, skip_serializing_if = "PropertyTable::is_empty")]
    pub properties: PropertyTable,
}

/// Noise basis used to distribute phases in space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NoiseKind {
    #[default]
    Perlin,
    Simplex,
    Worley,
    Value,
}

impl NoiseKind {
    pub fn parse(s: &str) -> Self {
        match crate::canon::canon(s).as_str() {
            "simplex" | "opensimplex" => Self::Simplex,
            "worley" | "cellular" | "voronoi" => Self::Worley,
            "value" => Self::Value,
            _ => Self::Perlin,
        }
    }
}

/// Grain geometry, straight from `Microstructure.GrainStructure`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GrainStructure {
    /// Mean grain diameter in metres (datasheets store micrometres).
    pub average_size_m: f64,
    /// Log-normal sigma of the grain-size distribution.
    pub size_std_dev: f64,
    pub astm_number: Option<f64>,
    /// Voronoi seeds per cubic metre.
    ///
    /// The datasheets quote an *areal* density (`VoronoiSeedDensity_per_mm2`);
    /// [`crate::ingest`] raises it to a volumetric one by `n_v = n_a^(3/2)`,
    /// which is the standard stereological relation for equiaxed grains.
    pub seed_density_per_m3: f64,
    /// Grain-boundary width in metres.
    pub boundary_width_m: f64,
    /// Elongation of grains along the rolling direction; 1.0 is equiaxed.
    pub aspect_ratio: f64,
    /// Twin boundaries per metre.
    pub twin_density_per_m: Option<f64>,
}

impl Default for GrainStructure {
    fn default() -> Self {
        Self {
            average_size_m: 50.0e-6,
            size_std_dev: 0.35,
            astm_number: None,
            seed_density_per_m3: 8.0e12,
            boundary_width_m: 0.5e-9,
            aspect_ratio: 1.0,
            twin_density_per_m: None,
        }
    }
}

/// Spatial distribution of phases, from `Microstructure.PhaseDistribution`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseDistribution {
    pub noise: NoiseKind,
    /// Noise feature size **relative to the mean grain diameter**.
    ///
    /// The datasheets store `NoiseScale` as a bare number (0.06, 0.08) with no
    /// unit and no stated reference length, so converting it to metres at
    /// ingest would mean inventing one. It is carried through dimensionless
    /// instead; use [`PhaseDistribution::scale_m`] to resolve it against a
    /// grain size, and let the field editor expose it for tuning.
    pub scale_rel: f64,
    pub octaves: u8,
    pub persistence: f64,
    /// Noise value above which the secondary phase appears.
    pub threshold: f64,
}

impl PhaseDistribution {
    /// Resolve the dimensionless [`Self::scale_rel`] to a feature size in
    /// metres against a reference length (normally the mean grain diameter).
    #[inline]
    pub fn scale_m(&self, reference_m: f64) -> f64 {
        // Guard the degenerate case so a zero in the data cannot produce a
        // zero-wavelength noise field that aliases at every sample point.
        if self.scale_rel > 0.0 {
            self.scale_rel * reference_m
        } else {
            reference_m
        }
    }
}

impl Default for PhaseDistribution {
    fn default() -> Self {
        Self {
            noise: NoiseKind::Perlin,
            scale_rel: 1.0,
            octaves: 3,
            persistence: 0.5,
            threshold: 0.9,
        }
    }
}

/// Lattice defects, from `Microstructure.Defects`. These lower the local
/// fracture resistance and raise local resistivity.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Defects {
    /// Vacancies per lattice site.
    pub vacancy_concentration: f64,
    pub interstitial_concentration: f64,
    /// Dislocation line length per unit volume, 1/m^2.
    pub dislocation_density_per_m2: f64,
    pub low_angle_boundary_fraction: f64,
    pub high_angle_boundary_fraction: f64,
    pub twin_boundary_fraction: f64,
    /// Residual stress magnitude in Pa, if the datasheet declares one.
    pub residual_stress_pa: Option<f64>,
}

/// Second-phase inclusions (oxides, sulphides). Classic crack initiators.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Inclusions {
    /// Inclusions per cubic metre.
    pub density_per_m3: f64,
    /// Mean inclusion diameter in metres.
    pub average_size_m: f64,
}

/// The full internal-structure description.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Microstructure {
    pub grains: GrainStructure,
    pub phase_distribution: PhaseDistribution,
    pub defects: Defects,
    pub inclusions: Inclusions,
}

/// Preferred grain orientation, from `CrystallographicOrientation`.
///
/// `strength_mrd` is in multiples of a random distribution: 1.0 means grains
/// are randomly oriented, higher means a sharper texture. The fracture model
/// uses it to decide how much grain orientations spread around the texture
/// axis, which controls whether cleavage produces one flat sheet or a stepped,
/// faceted surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Texture {
    pub preferred: bool,
    pub strength_mrd: f64,
    /// Texture axis in material-local coordinates (typically the rolling
    /// direction). `None` means untextured.
    pub axis: Option<[f32; 3]>,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            preferred: false,
            strength_mrd: 1.0,
            axis: None,
        }
    }
}

/// Where a record came from and how completely it ingested.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source file stem, e.g. `"Aluminum_6061_T6"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Data root the file came from (`active`, `derived`, `defaults`, `reference`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    /// Canonical keys seen during ingest that are neither mapped nor ignored.
    /// Empty for every bundled datasheet; non-empty means new data introduced
    /// a spelling the alias table does not know.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unknown_keys: Vec<String>,
}

/// A material, normalised to SI and ready for the runtime.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialRecord {
    pub name: String,
    #[serde(default)]
    pub tier: Tier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Element symbol -> mass fraction in `0..=1`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub composition: Vec<(String, f64)>,
    /// Bulk properties. Always SI (see [`crate::property`]).
    pub bulk: PropertyTable,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lattice: Option<Lattice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phases: Vec<Phase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub micro: Option<Microstructure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<Texture>,
    /// Base colour, linear sRGB in `0..=1`.
    #[serde(default = "default_color")]
    pub color_srgb: [f32; 3],
    #[serde(default)]
    pub provenance: Provenance,
}

fn default_color() -> [f32; 3] {
    [0.75, 0.75, 0.78]
}

impl MaterialRecord {
    /// Crystal system of the primary phase, or [`CrystalSystem::Unknown`].
    pub fn crystal_system(&self) -> CrystalSystem {
        self.lattice
            .map(|l| l.system)
            .filter(|s| *s != CrystalSystem::Unknown)
            .or_else(|| self.phases.first().map(|p| p.structure))
            .unwrap_or(CrystalSystem::Unknown)
    }

    /// The dominant phase by volume fraction, if any phases are declared.
    pub fn dominant_phase(&self) -> Option<&Phase> {
        self.phases.iter().max_by(|a, b| {
            a.volume_fraction
                .partial_cmp(&b.volume_fraction)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Whether ingest understood every key it saw.
    pub fn is_fully_ingested(&self) -> bool {
        self.provenance.unknown_keys.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::PropertyId;
    use crate::table::Source;

    #[test]
    fn crystal_system_parses_datasheet_spellings() {
        assert_eq!(CrystalSystem::parse("BCC"), CrystalSystem::Bcc);
        assert_eq!(CrystalSystem::parse("bcc"), CrystalSystem::Bcc);
        assert_eq!(
            CrystalSystem::parse("Body-Centered Cubic"),
            CrystalSystem::Bcc
        );
        assert_eq!(CrystalSystem::parse("FCC"), CrystalSystem::Fcc);
        assert_eq!(CrystalSystem::parse("HCP"), CrystalSystem::Hcp);
        // The element datasheets use these for the lanthanides / tin / polonium.
        assert_eq!(CrystalSystem::parse("DHCP"), CrystalSystem::Hcp);
        assert_eq!(CrystalSystem::parse("BCT"), CrystalSystem::Tetragonal);
        assert_eq!(CrystalSystem::parse("Diamond"), CrystalSystem::DiamondCubic);
        assert_eq!(
            CrystalSystem::parse("SimpleCubic"),
            CrystalSystem::SimpleCubic
        );
        assert_eq!(CrystalSystem::parse("Amorphous"), CrystalSystem::Amorphous);
        assert_eq!(CrystalSystem::parse("wurtzite"), CrystalSystem::Unknown);
    }

    #[test]
    fn fcc_has_no_cleavage_but_bcc_and_hcp_do() {
        // This is the physics the planar-fracture model rests on, so pin it.
        assert!(CrystalSystem::Fcc.cleavage_planes().is_empty());
        assert!(!CrystalSystem::Fcc.is_cleavable());
        assert_eq!(CrystalSystem::Bcc.cleavage_planes().len(), 3);
        assert_eq!(CrystalSystem::Hcp.cleavage_planes().len(), 1);
        assert_eq!(CrystalSystem::DiamondCubic.cleavage_planes().len(), 4);
        assert!(CrystalSystem::Amorphous.cleavage_planes().is_empty());
    }

    #[test]
    fn cleavage_normals_are_unit_length() {
        for sys in [
            CrystalSystem::Bcc,
            CrystalSystem::Hcp,
            CrystalSystem::DiamondCubic,
            CrystalSystem::SimpleCubic,
        ] {
            for n in sys.cleavage_planes() {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-5, "{sys:?} normal {n:?} len {len}");
            }
        }
    }

    #[test]
    fn noise_kind_parses_and_defaults() {
        assert_eq!(NoiseKind::parse("Simplex"), NoiseKind::Simplex);
        assert_eq!(NoiseKind::parse("Worley"), NoiseKind::Worley);
        assert_eq!(NoiseKind::parse("Voronoi"), NoiseKind::Worley);
        // Unrecognised falls back to Perlin rather than failing ingest.
        assert_eq!(NoiseKind::parse("whatever"), NoiseKind::Perlin);
    }

    #[test]
    fn crystal_system_falls_back_to_dominant_phase() {
        let mut r = MaterialRecord {
            name: "Test".into(),
            phases: vec![Phase {
                name: "Ferrite".into(),
                structure: CrystalSystem::Bcc,
                volume_fraction: 1.0,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(r.crystal_system(), CrystalSystem::Bcc);
        // An explicit lattice wins over the phase.
        r.lattice = Some(Lattice {
            system: CrystalSystem::Fcc,
            ..Default::default()
        });
        assert_eq!(r.crystal_system(), CrystalSystem::Fcc);
    }

    #[test]
    fn dominant_phase_picks_the_largest_fraction() {
        let r = MaterialRecord {
            name: "Steel".into(),
            phases: vec![
                Phase {
                    name: "pearlite".into(),
                    volume_fraction: 0.15,
                    ..Default::default()
                },
                Phase {
                    name: "ferrite".into(),
                    volume_fraction: 0.85,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(r.dominant_phase().unwrap().name, "ferrite");
    }

    #[test]
    fn record_json_round_trips() {
        let mut bulk = PropertyTable::new();
        bulk.set(PropertyId::Density, 7870.0, Source::Datasheet);
        bulk.set(PropertyId::YoungsModulus, 2.0e11, Source::Datasheet);

        let r = MaterialRecord {
            name: "Steel-1018".into(),
            tier: Tier::Alloy,
            category: Some("Carbon Steel".into()),
            composition: vec![("Fe".into(), 0.99), ("C".into(), 0.01)],
            bulk,
            lattice: Some(Lattice {
                system: CrystalSystem::Bcc,
                ..Default::default()
            }),
            micro: Some(Microstructure::default()),
            texture: Some(Texture::default()),
            ..Default::default()
        };

        let s = serde_json::to_string(&r).unwrap();
        let back: MaterialRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
        assert_eq!(back.crystal_system(), CrystalSystem::Bcc);
    }

    #[test]
    fn defaults_are_physically_sane() {
        let g = GrainStructure::default();
        assert!(g.average_size_m > 0.0 && g.average_size_m < 1e-3);
        assert!(g.boundary_width_m > 0.0 && g.boundary_width_m < g.average_size_m);
        assert_eq!(g.aspect_ratio, 1.0, "default grains must be equiaxed");

        let p = PhaseDistribution::default();
        assert!(p.octaves >= 1);
        assert!(p.persistence > 0.0 && p.persistence <= 1.0);
    }
}
