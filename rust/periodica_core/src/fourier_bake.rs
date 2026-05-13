//====== periodica/rust/periodica_core/src/fourier_bake.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Fourier-coefficient baking for material properties.
//!
//! Bakes a periodica material property (density, IOR, SSS coefficients,
//! caustic profile, ...) onto a 3D voxel grid, performs a 3D FFT, threshold-
//! prunes coefficients below a noise floor, and emits a `FourierFieldConfig`
//! that the engine sampler reconstructs at ray-step time.
//!
//! Wave-2 stub. The real FFT path lands in Wave 3 once the periodica side of
//! `voxel_sample` is ported.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// One Fourier coefficient: amplitude × cos(2π(n·x + m·y + l·z)/L + phase).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FourierCoefficient {
    pub n: i32,
    pub m: i32,
    pub l: i32,
    pub amplitude: f64,
    pub phase: f64,
}

/// Truncated 3D Fourier expansion of a scalar property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierFieldConfig {
    pub property_name: String,
    pub base_value: f64,
    pub domain_size_m: (f64, f64, f64),
    pub coefficients: Vec<FourierCoefficient>,
    pub boundary_condition: String,
}

impl FourierFieldConfig {
    /// Empty config representing a constant `base_value` field.
    pub fn constant(property_name: impl Into<String>, base_value: f64) -> Self {
        Self {
            property_name: property_name.into(),
            base_value,
            domain_size_m: (1.0, 1.0, 1.0),
            coefficients: Vec::new(),
            boundary_condition: "periodic".to_string(),
        }
    }

    /// Total non-DC coefficient count.
    pub fn coefficient_count(&self) -> usize {
        self.coefficients.len()
    }
}

/// Bake a material property into a Fourier expansion on a voxel grid.
///
/// Wave-2 stub — returns a constant config with `base_value = 0` so callers
/// can wire up the texture-upload pipeline before the FFT is real.
pub fn bake_fourier(
    entry_name: &str,
    property: &str,
    bounds: ((f64, f64, f64), (f64, f64, f64)),
    grid_size: (usize, usize, usize),
    truncate_threshold: f64,
) -> Result<FourierFieldConfig> {
    if entry_name.is_empty() {
        return Err(anyhow!("entry_name must not be empty"));
    }
    if grid_size.0 == 0 || grid_size.1 == 0 || grid_size.2 == 0 {
        return Err(anyhow!("grid_size must be non-zero in all dimensions"));
    }
    if !truncate_threshold.is_finite() || truncate_threshold < 0.0 {
        return Err(anyhow!("truncate_threshold must be a non-negative finite value"));
    }
    let (lo, hi) = bounds;
    Ok(FourierFieldConfig {
        property_name: property.to_string(),
        base_value: 0.0,
        domain_size_m: (hi.0 - lo.0, hi.1 - lo.1, hi.2 - lo.2),
        coefficients: Vec::new(),
        boundary_condition: "periodic".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_name_rejected() {
        let r = bake_fourier("", "density", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (8, 8, 8), 0.001);
        assert!(r.is_err());
    }

    #[test]
    fn zero_grid_rejected() {
        let r = bake_fourier("Fe", "density", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0, 8, 8), 0.001);
        assert!(r.is_err());
    }

    #[test]
    fn negative_threshold_rejected() {
        let r = bake_fourier("Fe", "density", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (8, 8, 8), -0.1);
        assert!(r.is_err());
    }

    #[test]
    fn nominal_bake_returns_config() {
        let cfg = bake_fourier("Fe", "density", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (8, 8, 8), 0.001).unwrap();
        assert_eq!(cfg.property_name, "density");
        assert_eq!(cfg.coefficient_count(), 0); // stub
        assert_eq!(cfg.domain_size_m, (1.0, 1.0, 1.0));
    }

    #[test]
    fn constant_helper_works() {
        let cfg = FourierFieldConfig::constant("density", 7874.0);
        assert_eq!(cfg.base_value, 7874.0);
        assert_eq!(cfg.coefficient_count(), 0);
    }
}
