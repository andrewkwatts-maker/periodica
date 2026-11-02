//! Poisson-Voronoi grains in material coordinates, hashed per cell so there is no global state.

use crate::sampling::{splitmix64, unit_from_bits};
use periodica_mat::GrainStructure;

const MAX_PER_CELL: u32 = 8;
const E_INV: f64 = 0.367_879_441_171_442_33;

/// One grain: a stable id and the index of its phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grain {
    /// Hash of the owning cell and seed index.
    pub id: u64,
    /// Index into the record's phase list.
    pub phase: usize,
}

/// The grain structure of one material volume.
#[derive(Debug, Clone)]
pub struct GrainField {
    cell_m: f64,
    seed: u64,
    scale_par: f64,
    scale_perp: f64,
    axis: [f64; 3],
    cumulative: Vec<f64>,
}

impl GrainField {
    /// Builds the field from grain data, an optional texture axis, phase volume fractions and a seed.
    pub fn new(
        grains: &GrainStructure,
        axis: Option<[f32; 3]>,
        fractions: &[f64],
        seed: u64,
    ) -> Self {
        let aspect = if grains.aspect_ratio.is_finite() && grains.aspect_ratio >= 1.0 {
            grains.aspect_ratio
        } else {
            1.0
        };
        let (axis, aspect) = unit_axis(axis).map_or(([0.0; 3], 1.0), |a| (a, aspect));
        Self {
            cell_m: grains.seed_density_per_m3.max(f64::MIN_POSITIVE).powf(-1.0 / 3.0),
            seed,
            scale_par: aspect.powf(-2.0 / 3.0),
            scale_perp: aspect.powf(1.0 / 3.0),
            axis,
            cumulative: cumulative(fractions),
        }
    }

    /// Edge of one hash cell in metres, `n_v^(-1/3)`.
    pub fn cell_m(&self) -> f64 {
        self.cell_m
    }

    /// Seeds in the cell block `0..n[0] x 0..n[1] x 0..n[2]`.
    pub fn count_seeds(&self, n: [u32; 3]) -> u64 {
        let mut total = 0u64;
        for x in 0..i64::from(n[0]) {
            for y in 0..i64::from(n[1]) {
                for z in 0..i64::from(n[2]) {
                    total += u64::from(self.cell_count(self.cell_hash([x, y, z])));
                }
            }
        }
        total
    }

    /// The grain whose seed is nearest to the material point `p`.
    pub fn grain_at(&self, p: [f64; 3]) -> Grain {
        let q = self.stretch(p);
        let cell = [0, 1, 2].map(|i| (q[i] / self.cell_m).floor() as i64);
        let mut best: Option<(f64, Grain)> = None;
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                for dz in -1..=1i64 {
                    let c = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    self.nearest_in_cell(c, q, &mut best);
                }
            }
        }
        best.map_or_else(|| self.fallback(cell), |(_, g)| g)
    }

    fn nearest_in_cell(&self, c: [i64; 3], q: [f64; 3], best: &mut Option<(f64, Grain)>) {
        let h = self.cell_hash(c);
        for k in 0..self.cell_count(h) {
            let mut s = h.wrapping_add(u64::from(k + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let id = splitmix64(&mut s);
            let mut d2 = 0.0;
            for i in 0..3 {
                let pos = (c[i] as f64 + unit_from_bits(splitmix64(&mut s))) * self.cell_m;
                d2 += (pos - q[i]) * (pos - q[i]);
            }
            if best.is_none_or(|(b, _)| d2 < b) {
                *best = Some((d2, Grain { id, phase: self.phase_of(id) }));
            }
        }
    }

    fn fallback(&self, cell: [i64; 3]) -> Grain {
        let id = self.cell_hash(cell);
        Grain { id, phase: self.phase_of(id) }
    }

    fn phase_of(&self, id: u64) -> usize {
        let mut s = id ^ self.seed.rotate_left(17);
        let u = unit_from_bits(splitmix64(&mut s));
        self.cumulative
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.cumulative.len().saturating_sub(1))
    }

    fn cell_hash(&self, c: [i64; 3]) -> u64 {
        let mut s = self.seed;
        for v in c {
            let mut t = s ^ v as u64;
            s = splitmix64(&mut t);
        }
        s
    }

    fn cell_count(&self, hash: u64) -> u32 {
        let mut s = hash ^ 0xA076_1D64_78BD_642F;
        let u = unit_from_bits(splitmix64(&mut s));
        let (mut term, mut cum) = (E_INV, E_INV);
        for k in 0..MAX_PER_CELL {
            if u < cum {
                return k;
            }
            term /= f64::from(k + 1);
            cum += term;
        }
        MAX_PER_CELL
    }

    fn stretch(&self, p: [f64; 3]) -> [f64; 3] {
        let along = p[0] * self.axis[0] + p[1] * self.axis[1] + p[2] * self.axis[2];
        let extra = (self.scale_par - self.scale_perp) * along;
        [0, 1, 2].map(|i| self.scale_perp * p[i] + extra * self.axis[i])
    }
}

fn unit_axis(axis: Option<[f32; 3]>) -> Option<[f64; 3]> {
    let a = axis?.map(f64::from);
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    (len.is_finite() && len > 0.0).then(|| a.map(|v| v / len))
}

fn cumulative(fractions: &[f64]) -> Vec<f64> {
    let total: f64 = fractions.iter().map(|f| f.max(0.0)).sum();
    if total <= 0.0 {
        return vec![1.0];
    }
    let mut acc = 0.0;
    fractions
        .iter()
        .map(|f| {
            acc += f.max(0.0) / total;
            acc
        })
        .collect()
}
