//! Errors of the material volume.

use periodica_mat::{MixError, PropertyId};

/// Why a probe or bake failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VolumeError {
    /// The probe or bake geometry is not finite, not positive, or not a valid box.
    #[error("bad probe: {0}")]
    BadProbe(&'static str),
    /// The bake would exceed the voxel cap.
    #[error("bake of {count} voxels exceeds the cap of {cap}")]
    TooManyVoxels {
        /// Requested voxel count.
        count: u128,
        /// Largest allowed voxel count.
        cap: usize,
    },
    /// The mixing rule failed.
    #[error("mix failed: {0}")]
    Mix(#[from] MixError),
    /// The record has no bulk value for the property.
    #[error("record has no {0:?}")]
    MissingProperty(PropertyId),
}
