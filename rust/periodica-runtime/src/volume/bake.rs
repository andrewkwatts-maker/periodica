//! Voxel bake of the material volume: the probe sampled at voxel centres.

use super::error::VolumeError;
use super::probe::{MaterialVolume, Probe};
use periodica_mat::PropertyId;

/// Largest voxel count a bake accepts.
pub const MAX_VOXELS: usize = 1 << 24;

/// Baked properties, one `f32` array per property, x fastest.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyVolume {
    /// Voxels per axis.
    pub dims: [usize; 3],
    /// Minimum corner in metres.
    pub origin: [f64; 3],
    /// Voxel edge in metres.
    pub voxel_m: f64,
    /// Baked properties, in array order.
    pub props: Vec<PropertyId>,
    /// `data[i][x + dims[0] * (y + dims[1] * z)]` is property `i`.
    pub data: Vec<Vec<f32>>,
}

/// Bakes `props` over the box `bounds = (min, max)` at `voxel_m` resolution.
pub fn bake(
    volume: &MaterialVolume,
    bounds: ([f64; 3], [f64; 3]),
    voxel_m: f64,
    props: &[PropertyId],
) -> Result<PropertyVolume, VolumeError> {
    let (lo, hi) = bounds;
    if !(voxel_m.is_finite() && voxel_m > 0.0) {
        return Err(VolumeError::BadProbe("voxel size must be finite and positive"));
    }
    let dims = dims_of(lo, hi, voxel_m)?;
    let mut data = vec![Vec::with_capacity(dims[0] * dims[1] * dims[2]); props.len()];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let c = [x, y, z].map(|i| i as f64);
                let centre = [0, 1, 2].map(|k| lo[k] + (c[k] + 0.5) * voxel_m);
                let probe = Probe { point: centre, half_extent_m: 0.5 * voxel_m, direction: None };
                for (column, v) in data.iter_mut().zip(volume.probe(&probe, props)?) {
                    column.push(v as f32);
                }
            }
        }
    }
    Ok(PropertyVolume { dims, origin: lo, voxel_m, props: props.to_vec(), data })
}

fn dims_of(lo: [f64; 3], hi: [f64; 3], voxel_m: f64) -> Result<[usize; 3], VolumeError> {
    let mut dims = [0usize; 3];
    let mut count: u128 = 1;
    for k in 0..3 {
        let n = ((hi[k] - lo[k]) / voxel_m).ceil();
        if !n.is_finite() || n < 1.0 {
            return Err(VolumeError::BadProbe("bounds must be finite with max above min"));
        }
        if n > MAX_VOXELS as f64 {
            return Err(VolumeError::TooManyVoxels { count: u128::MAX, cap: MAX_VOXELS });
        }
        dims[k] = n as usize;
        count *= dims[k] as u128;
    }
    if count > MAX_VOXELS as u128 {
        return Err(VolumeError::TooManyVoxels { count, cap: MAX_VOXELS });
    }
    Ok(dims)
}
