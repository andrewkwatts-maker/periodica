//! Sampled spectra and the single spectrum-to-RGB conversion.

use super::xyz::{self, rgb_of_xyz};

/// A spectrum sampled at `bins.len()` evenly spaced wavelengths over `[lo, hi]` metres.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    pub lo: f64,
    pub hi: f64,
    pub bins: Vec<f64>,
}

impl Spectrum {
    /// Samples `f` (wavelength in metres) over the band; `bins` is raised to at least 2.
    pub fn sample(lo: f64, hi: f64, bins: usize, f: impl Fn(f64) -> f64) -> Self {
        let n = bins.max(2);
        let values = (0..n).map(|i| f(lo + (hi - lo) * i as f64 / (n - 1) as f64)).collect();
        Self { lo, hi, bins: values }
    }

    /// A constant spectrum.
    pub fn flat(lo: f64, hi: f64, bins: usize, value: f64) -> Self {
        Self::sample(lo, hi, bins, |_| value)
    }

    /// Linear interpolation; zero outside the band.
    pub fn at(&self, lambda: f64) -> f64 {
        if lambda < self.lo || lambda > self.hi || self.bins.len() < 2 {
            return 0.0;
        }
        let pos = (lambda - self.lo) / (self.hi - self.lo) * (self.bins.len() - 1) as f64;
        let i = (pos.floor() as usize).min(self.bins.len() - 2);
        let t = pos - i as f64;
        self.bins[i] * (1.0 - t) + self.bins[i + 1] * t
    }

    /// Trapezoid integral over wavelength in metres.
    pub fn integral(&self) -> f64 {
        let n = self.bins.len();
        let h = (self.hi - self.lo) / (n - 1) as f64;
        let inner: f64 = self.bins[1..n - 1].iter().sum();
        h * (inner + 0.5 * (self.bins[0] + self.bins[n - 1]))
    }
}

/// Width in metres of the luminous band: the integral of the CIE y-bar over wavelength.
pub fn luminous_band_m() -> f64 {
    xyz::table().y_sum * 1e-9
}

/// Linear sRGB of a spectrum: CIE 1931 2-degree matching functions, then the sRGB
/// primaries, balanced so a flat spectrum of value `v` gives `[v, v, v]`.
/// This is the only spectrum-to-RGB conversion in the ecosystem.
pub fn spectrum_to_rgb(s: &Spectrum) -> [f64; 3] {
    let t = xyz::table();
    let mut acc = [0.0; 3];
    for (i, c) in t.cmf.iter().enumerate() {
        let v = s.at((xyz::LAMBDA_MIN_NM + i as f64) * 1e-9);
        for (a, w) in acc.iter_mut().zip(c) {
            *a += v * w;
        }
    }
    let rgb = rgb_of_xyz([acc[0] / t.y_sum, acc[1] / t.y_sum, acc[2] / t.y_sum]);
    fit_gamut([rgb[0] / t.white[0], rgb[1] / t.white[1], rgb[2] / t.white[2]])
}

const GAMUT_FLOOR: f64 = 1e-3;

/// Pulls an out-of-gamut colour toward its own mean until every channel is at least
/// `GAMUT_FLOOR` of the mean, keeping the mean fixed. In-gamut colours are unchanged.
fn fit_gamut(rgb: [f64; 3]) -> [f64; 3] {
    let mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let floor = GAMUT_FLOOR * mean;
    if mean <= 0.0 || min >= floor {
        return rgb;
    }
    let s = (floor - min) / (mean - min);
    rgb.map(|c| c + s * (mean - c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_white_is_neutral() {
        let rgb = spectrum_to_rgb(&Spectrum::flat(300e-9, 900e-9, 601, 2.0));
        for c in rgb {
            assert!((c - 2.0).abs() < 1e-3, "{rgb:?}");
        }
    }

    #[test]
    fn integral_of_flat() {
        let s = Spectrum::flat(0.0, 2.0, 11, 3.0);
        assert!((s.integral() - 6.0).abs() < 1e-12);
    }
}
