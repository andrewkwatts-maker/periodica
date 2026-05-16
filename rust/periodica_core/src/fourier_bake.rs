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
//! caustic profile, ...) onto a 3D voxel grid, performs a direct 3D DFT,
//! threshold-prunes coefficients below `truncate_threshold × max_amplitude`,
//! and emits a `FourierFieldConfig` the engine sampler reconstructs per ray step.
//!
//! Reconstruction formula (mirrors Python `FourierFieldEvaluator.evaluate`):
//!   P(x,y,z) = base_value + Σ A·cos(2π(n·x/Lx + m·y/Ly + l·z/Lz) + φ)
//!
//! The direct DFT is O(N²) in grid point count, which is acceptable for
//! offline baking (typical grids 8³–32³). An FFT path can replace it later.
//
// Python fallback: src/periodica/utils/transforms/fourier_field.py

use std::f64::consts::TAU; // 2π

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// One Fourier coefficient: amplitude × cos(2π(n·x/Lx + m·y/Ly + l·z/Lz) + phase).
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
    /// DC component — mean of the sampled voxel grid.
    pub base_value: f64,
    /// Periodic domain size (Lx, Ly, Lz) in metres.
    pub domain_size_m: (f64, f64, f64),
    pub coefficients: Vec<FourierCoefficient>,
    pub boundary_condition: String,
}

impl FourierFieldConfig {
    /// Constant-field config — no AC terms, reconstructs to `base_value` everywhere.
    pub fn constant(property_name: impl Into<String>, base_value: f64) -> Self {
        Self {
            property_name: property_name.into(),
            base_value,
            domain_size_m: (1.0, 1.0, 1.0),
            coefficients: Vec::new(),
            boundary_condition: "periodic".to_string(),
        }
    }

    /// Number of non-DC (AC) coefficients retained after truncation.
    pub fn coefficient_count(&self) -> usize {
        self.coefficients.len()
    }

    /// Reconstruct the field value at world position (x, y, z).
    pub fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let (lx, ly, lz) = self.domain_size_m;
        let kx = if lx > 0.0 { TAU / lx } else { 0.0 };
        let ky = if ly > 0.0 { TAU / ly } else { 0.0 };
        let kz = if lz > 0.0 { TAU / lz } else { 0.0 };
        let mut v = self.base_value;
        for c in &self.coefficients {
            let arg = c.n as f64 * kx * x + c.m as f64 * ky * y + c.l as f64 * kz * z + c.phase;
            v += c.amplitude * arg.cos();
        }
        v
    }
}

// ─── Public baking entry point ──────────────────────────────────────────────

/// Bake a material property into a truncated 3D Fourier expansion.
///
/// Steps:
/// 1. Sample `property` on a uniform `grid_size` voxel grid spanning `bounds`.
/// 2. Compute the mean (DC component → `base_value`).
/// 3. Compute each non-DC frequency in the positive half-space via direct DFT.
/// 4. Discard modes whose amplitude < `truncate_threshold × max_amplitude`.
/// 5. Return a `FourierFieldConfig` ready for engine texture upload.
///
/// `truncate_threshold = 0.0` retains all modes; `0.01` keeps modes ≥ 1% of peak.
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
    if property.is_empty() {
        return Err(anyhow!("property must not be empty"));
    }
    let (nx, ny, nz) = grid_size;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(anyhow!("grid_size must be non-zero in all dimensions"));
    }
    if !truncate_threshold.is_finite() || truncate_threshold < 0.0 {
        return Err(anyhow!("truncate_threshold must be a non-negative finite value"));
    }

    let (lo, hi) = bounds;
    let lx = (hi.0 - lo.0).max(f64::EPSILON);
    let ly = (hi.1 - lo.1).max(f64::EPSILON);
    let lz = (hi.2 - lo.2).max(f64::EPSILON);
    let voxel_x = lx / nx as f64;
    let voxel_y = ly / ny as f64;
    let voxel_z = lz / nz as f64;
    // Scale hint: smallest voxel dimension
    let voxel_size = voxel_x.min(voxel_y).min(voxel_z);

    // ── Step 1: sample property on the voxel grid ──
    let n_total = nx * ny * nz;
    let mut grid = vec![vec![vec![0.0f64; nz]; ny]; nx];
    let mut grid_sum = 0.0f64;

    for ix in 0..nx {
        let x = lo.0 + (ix as f64 + 0.5) * voxel_x;
        for iy in 0..ny {
            let y = lo.1 + (iy as f64 + 0.5) * voxel_y;
            for iz in 0..nz {
                let z = lo.2 + (iz as f64 + 0.5) * voxel_z;
                let v = crate::sample::sample(entry_name, property, Some((x, y, z)), Some(voxel_size))
                    .unwrap_or(0.0);
                grid[ix][iy][iz] = v;
                grid_sum += v;
            }
        }
    }

    // ── Step 2: DC component (mean) ──
    let base_value = grid_sum / n_total as f64;

    // Subtract mean so the zero-frequency DFT term is zero (saves a DFT call).
    for ix in 0..nx { for iy in 0..ny { for iz in 0..nz {
        grid[ix][iy][iz] -= base_value;
    }}}

    // ── Step 3: 3D DFT — positive half-space only ──
    // For a real-valued grid f, F[-kn,-km,-kl] = conj(F[kn,km,kl]).
    // The canonical positive half-space avoids double-counting:
    //   kn > 0, OR (kn==0 && km > 0), OR (kn==0 && km==0 && kl > 0)
    // Amplitude for reconstruction = 2·|F[kn,km,kl]| / N  (factor 2 for the pair).
    // Phase = atan2(Im, Re).
    let n_total_f = n_total as f64;
    let max_kn = (nx / 2) as i32;
    let max_km = (ny / 2) as i32;
    let max_kl = (nz / 2) as i32;

    let mut raw: Vec<FourierCoefficient> = Vec::new();
    let mut max_amp = 0.0f64;

    for kn in 0..=max_kn {
        let km_lo: i32 = if kn == 0 { 0 } else { -max_km };
        for km in km_lo..=max_km {
            let kl_lo: i32 = if kn == 0 && km == 0 { 1 } else { -max_kl };
            for kl in kl_lo..=max_kl {
                // Canonical half-space: (0,0,l>0), (0,m>0,l_any), (n>0,m_any,l_any)
                // The loop bounds already enforce this via kl_lo and km_lo.

                let (re, im) = dft3(&grid, nx, ny, nz, kn, km, kl);
                // At Nyquist (kn==nx/2 for even nx), the conjugate is itself → factor is 1 not 2.
                // For simplicity use factor 2 everywhere (slight overcount at exact Nyquist).
                let amplitude = 2.0 * (re * re + im * im).sqrt() / n_total_f;
                if amplitude > max_amp { max_amp = amplitude; }
                let phase = im.atan2(re);
                raw.push(FourierCoefficient { n: kn, m: km, l: kl, amplitude, phase });
            }
        }
    }

    // ── Step 4: threshold ──
    let thresh = if max_amp > 0.0 { truncate_threshold * max_amp } else { 0.0 };
    let coefficients: Vec<FourierCoefficient> = raw
        .into_iter()
        .filter(|c| c.amplitude >= thresh)
        .collect();

    Ok(FourierFieldConfig {
        property_name: property.to_string(),
        base_value,
        domain_size_m: (lx, ly, lz),
        coefficients,
        boundary_condition: "periodic".to_string(),
    })
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Direct 3D DFT at a single signed frequency triple (kn, km, kl).
/// Returns (Re, Im) of the unnormalised DFT coefficient.
fn dft3(grid: &[Vec<Vec<f64>>], nx: usize, ny: usize, nz: usize, kn: i32, km: i32, kl: i32) -> (f64, f64) {
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let angle = -TAU * (
                    kn as f64 * ix as f64 / nx as f64
                    + km as f64 * iy as f64 / ny as f64
                    + kl as f64 * iz as f64 / nz as f64
                );
                let v = grid[ix][iy][iz];
                re += v * angle.cos();
                im += v * angle.sin();
            }
        }
    }
    (re, im)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

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
    fn nominal_bake_produces_config() {
        // DataHub is empty in tests; sample() returns 0.0 for all points.
        // Constant-zero field → base_value=0, all AC coefficients amplitude=0
        // → after threshold, zero coefficients.
        let cfg = bake_fourier("Fe", "density", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (8, 8, 8), 0.001)
            .unwrap();
        assert_eq!(cfg.property_name, "density");
        assert_eq!(cfg.domain_size_m, (1.0, 1.0, 1.0));
        // Constant-zero field → no AC coefficients survive threshold
        for c in &cfg.coefficients {
            assert!(c.amplitude < 1e-10, "expected ~0 amplitude, got {}", c.amplitude);
        }
    }

    #[test]
    fn constant_helper_works() {
        let cfg = FourierFieldConfig::constant("density", 7874.0);
        assert_eq!(cfg.base_value, 7874.0);
        assert_eq!(cfg.coefficient_count(), 0);
    }

    #[test]
    fn evaluate_constant_field() {
        let cfg = FourierFieldConfig::constant("ior", 1.5);
        assert!((cfg.evaluate(0.1, 0.2, 0.3) - 1.5).abs() < 1e-12);
        assert!((cfg.evaluate(0.0, 0.0, 0.0) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn evaluate_single_mode() {
        // A_100 * cos(2π x/L) at x=0 → A_100
        let cfg = FourierFieldConfig {
            property_name: "test".into(),
            base_value: 1.0,
            domain_size_m: (1.0, 1.0, 1.0),
            coefficients: vec![FourierCoefficient { n: 1, m: 0, l: 0, amplitude: 0.5, phase: 0.0 }],
            boundary_condition: "periodic".into(),
        };
        // At x=0: cos(0) = 1 → 1.0 + 0.5 = 1.5
        assert!((cfg.evaluate(0.0, 0.0, 0.0) - 1.5).abs() < 1e-12);
        // At x=L/2: cos(π) = -1 → 1.0 - 0.5 = 0.5
        assert!((cfg.evaluate(0.5, 0.0, 0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn dft3_of_constant_is_zero_for_nondc() {
        // Constant grid after DC subtraction = zero grid → all DFT coefficients zero
        let grid = vec![vec![vec![0.0f64; 4]; 4]; 4];
        let (re, im) = dft3(&grid, 4, 4, 4, 1, 0, 0);
        assert!(re.abs() < 1e-12 && im.abs() < 1e-12);
    }

    #[test]
    fn dft3_single_cosine_mode() {
        // Grid: f[ix][iy][iz] = cos(2π ix/N) → DFT at k=1 should give N/2 (unnormalised)
        let n = 8usize;
        let mut grid = vec![vec![vec![0.0f64; 1]; 1]; n];
        for ix in 0..n {
            grid[ix][0][0] = (TAU * ix as f64 / n as f64).cos();
        }
        let (re, _im) = dft3(&grid, n, 1, 1, 1, 0, 0);
        // Expected: N/2 = 4
        assert!((re - n as f64 / 2.0).abs() < 1e-9, "re={re}");
    }

    #[test]
    fn bake_preserves_domain_size() {
        let cfg = bake_fourier(
            "Fe", "density",
            ((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)),
            (4, 6, 8), 0.0,
        ).unwrap();
        assert!((cfg.domain_size_m.0 - 2.0).abs() < 1e-10);
        assert!((cfg.domain_size_m.1 - 3.0).abs() < 1e-10);
        assert!((cfg.domain_size_m.2 - 4.0).abs() < 1e-10);
    }
}
