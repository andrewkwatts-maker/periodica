//! Volumetric material: Poisson-Voronoi grains, a volume probe calibrated to
//! the record's bulk values, and a voxel bake.

pub mod bake;
pub mod error;
pub mod grains;
pub mod probe;

pub use bake::{bake, PropertyVolume, MAX_VOXELS};
pub use error::VolumeError;
pub use grains::{Grain, GrainField};
pub use probe::{MaterialVolume, Probe, PROBE_SAMPLES, RVE_GRAIN_DIAMETERS};
