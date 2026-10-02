//====== periodica/rust/periodica-runtime/src/fracture.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # fracture
//!
//! Weakpoint-following fracture: sample fracture toughness on a grid, then let
//! A* find the cheapest way for a crack to cross the body.
//!
//! ## Model
//!
//! 1. [`WeakpointGrid::build`] samples `K_IC` at every cell centre. A cell's
//!    cost is its toughness, so a minimum-cost path **follows the weakest
//!    material** (defects, weak inclusions, grain boundaries).
//! 2. The plane **normal** `n` is the dominant axis of the principal stress
//!    (the crack opens against the pull). With no stress field it is `+y`, so
//!    a dropped brick cracks horizontally.
//! 3. The crack runs **across** the section, perpendicular to `n`. For each
//!    transverse axis `t != n`, A* finds the cheapest 6-connected path from
//!    the low face of `t` to its high face. Each path is a trace of the crack
//!    front through the section.
//! 4. The plane's `offset` (Hesse normal form, `n . p = offset`) is the mean
//!    `n`-coordinate of the cell centres of both transverse paths, so it sits
//!    where the weak material is. `weakpoint_confidence = 1 / (1 + c)`, where
//!    `c` is the mean cell cost over both paths.
//!
//! ## Search details
//!
//! - **Heuristic**: remaining cells to the goal face along `t`, times the
//!   cheapest cell cost in the grid. Every remaining step enters at least one
//!   cell costing at least that much, so it never overestimates (admissible);
//!   one step changes it by at most the cost of the cell entered
//!   (consistent), so each cell is settled once.
//! - **Goal**: a cell on the high face of `t`, tested when the
//!   cell is popped (not pushed), which is what makes the path optimal.
//! - **Ties**: the open set orders by lowest `f`, then highest `g` (deeper
//!   first, the standard A* tie-break, which walks straight down a uniform
//!   grid instead of flooding it), then nearest the middle of the section
//!   along `n`, then lowest flat index. The whole key is a total order, so the
//!   search is deterministic. The centre preference only decides between
//!   *equal-cost* crack positions; without it a uniform material would always
//!   crack along its lowest face, an artefact of index order.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use thiserror::Error;

use crate::fields::ToughnessFn;

/// Errors produced by the fracture system.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FractureError {
    /// Grid resolution is zero in some axis.
    #[error("weakpoint grid resolution must be > 0 in all axes")]
    ZeroResolution,
    /// Toughness function returned NaN/Inf for some sampled point (or a
    /// hand-built grid holds a non-finite cost; the point is that cell's
    /// centre).
    #[error("non-finite toughness sampled at {0:?}")]
    NonFiniteToughness([f64; 3]),
    /// A hand-built grid whose `cost` length disagrees with its resolution.
    #[error("weakpoint grid holds {actual} cost cells but its resolution implies {expected}")]
    CostLengthMismatch {
        /// `resolution.0 * resolution.1 * resolution.2`.
        expected: usize,
        /// `cost.len()`.
        actual: usize,
    },
}

/// Default grid resolution for the weakpoint search. Tuned so a 1 m^3 object at
/// 64^3 resolution gives ~1.5 cm per cell -- fine enough for visible fracture
/// lines, coarse enough to keep A* runtime bounded.
pub const DEFAULT_FRACTURE_GRID_RESOLUTION: (usize, usize, usize) = (64, 64, 64);

/// Floor on a cell's cost. Keeps every step strictly positive, which both
/// Dijkstra and A* need, even where the toughness closure returns zero.
const MIN_CELL_COST: f64 = 1.0e-9;

/// Pre-computed grid of A* cost values driving the fracture pathfinder.
///
/// `cost[i] = toughness(centre of cell i)` (floored at `1e-9`), directly
/// proportional to the material's resistance to fracture, so the
/// minimum-cost path follows the weakest material.
///
/// Cells are stored x-fastest: `i = (z * res_y + y) * res_x + x`.
#[derive(Debug, Clone, PartialEq)]
pub struct WeakpointGrid {
    /// Cells along x, y and z.
    pub resolution: (usize, usize, usize),
    /// Lower and upper corners of the sampled box, in metres.
    pub bounds: ([f64; 3], [f64; 3]),
    /// Per-cell cost (toughness, Pa*m^0.5 when built from SI data).
    pub cost: Vec<f32>,
}

impl WeakpointGrid {
    /// Sample the toughness function at every cell centre of a regular grid.
    ///
    /// # Errors
    ///
    /// [`FractureError::ZeroResolution`] if any axis has zero cells;
    /// [`FractureError::NonFiniteToughness`] at the first point where
    /// `toughness` returns NaN/Inf.
    pub fn build(
        bounds: ([f64; 3], [f64; 3]),
        resolution: (usize, usize, usize),
        toughness: ToughnessFn<'_>,
    ) -> Result<Self, FractureError> {
        let (rx, ry, rz) = resolution;
        if rx == 0 || ry == 0 || rz == 0 {
            return Err(FractureError::ZeroResolution);
        }
        let (lo, hi) = bounds;
        let mut cost = Vec::with_capacity(rx * ry * rz);
        // Bounded loop: one sample per cell.
        for k in 0..rz {
            let z = axis_centre(lo[2], hi[2], rz, k);
            for j in 0..ry {
                let y = axis_centre(lo[1], hi[1], ry, j);
                for i in 0..rx {
                    let x = axis_centre(lo[0], hi[0], rx, i);
                    let toughness_value = toughness([x, y, z]);
                    if !toughness_value.is_finite() {
                        return Err(FractureError::NonFiniteToughness([x, y, z]));
                    }
                    cost.push(toughness_value.max(MIN_CELL_COST) as f32);
                }
            }
        }
        Ok(Self {
            resolution,
            bounds,
            cost,
        })
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.cost.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.cost.is_empty()
    }

    /// World-space centre of a cell.
    fn cell_centre(&self, cell: CellIndex) -> [f64; 3] {
        let (lo, hi) = self.bounds;
        [0, 1, 2].map(|a| {
            axis_centre(
                lo[a],
                hi[a],
                axis_size(self.resolution, a),
                cell.coord(a) as usize,
            )
        })
    }

    /// Cost of entering a cell, as used by the search.
    #[inline]
    fn step_cost(&self, flat: usize) -> f64 {
        f64::from(self.cost[flat]).max(MIN_CELL_COST)
    }
}

/// Centre of cell `i` of `n` along one axis of `[lo, hi]`.
#[inline]
fn axis_centre(lo: f64, hi: f64, n: usize, i: usize) -> f64 {
    lo + (hi - lo) * (i as f64 + 0.5) / n as f64
}

/// A baked fracture plane: a normal + offset (Hesse normal form,
/// `normal . p = offset`), plus a confidence score derived from the
/// weakpoint cost along the A* paths it was fitted to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FracturePlane {
    /// Unit plane normal (axis-aligned: the dominant stress axis).
    pub normal: [f64; 3],
    /// Signed distance of the plane from the origin along `normal`, metres.
    pub offset: f64,
    /// `1 / (1 + mean cell cost along the crack)`, in `(0, 1]`. Higher means
    /// the crack found weaker material. Depends on the cost units: with SI
    /// toughness (Pa*m^0.5) it is small; compare it between planes of the same
    /// material, not as an absolute probability.
    pub weakpoint_confidence: f64,
}

/// Applied stress, reduced to its principal component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressField {
    /// Principal-stress direction in world space (need not be unit length).
    pub principal_dir: [f64; 3],
    /// Principal-stress magnitude in Pa (SI). Not used by the path search
    /// itself, which only needs the direction.
    pub principal_magnitude: f64,
}

/// Find candidate fracture planes through a weakpoint grid.
///
/// Returns one plane whose normal is the dominant axis of
/// `stress_field.principal_dir` (`+y` when `None`) and whose offset follows
/// the weakest material across the section; see the [module docs](crate::fracture) for
/// the model. An empty grid yields no planes.
///
/// # Errors
///
/// For a hand-built grid: [`FractureError::CostLengthMismatch`] when
/// `cost.len()` disagrees with `resolution`, and
/// [`FractureError::NonFiniteToughness`] for a NaN/Inf cost. A grid from
/// [`WeakpointGrid::build`] never hits either.
pub fn find_fracture_paths(
    grid: &WeakpointGrid,
    stress_field: Option<&StressField>,
) -> Result<Vec<FracturePlane>, FractureError> {
    if grid.is_empty() {
        return Ok(Vec::new());
    }
    let step_lower_bound = validated_min_step_cost(grid)?;
    let normal_axis = principal_axis(stress_field);

    let paths: Vec<PathSearch> = (0..3)
        .filter(|&t| t != normal_axis)
        .filter_map(|t| astar_face_to_face(grid, t, normal_axis, step_lower_bound))
        .collect();
    if paths.iter().all(|p| p.cells.is_empty()) {
        return Ok(Vec::new()); // Disconnected -- no fracture.
    }
    Ok(vec![fit_plane(&paths, grid, normal_axis)])
}

/// Check a grid is well formed and return its cheapest step cost -- the
/// per-step lower bound the A* heuristic scales by. Computed once per call.
fn validated_min_step_cost(grid: &WeakpointGrid) -> Result<f64, FractureError> {
    let (rx, ry, rz) = grid.resolution;
    let expected = rx
        .checked_mul(ry)
        .and_then(|v| v.checked_mul(rz))
        .unwrap_or(usize::MAX);
    if expected != grid.cost.len() {
        return Err(FractureError::CostLengthMismatch {
            expected,
            actual: grid.cost.len(),
        });
    }
    let mut min_cost = f64::INFINITY;
    for (flat, c) in grid.cost.iter().enumerate() {
        if !c.is_finite() {
            let cell = CellIndex::from_flat(flat, grid.resolution);
            return Err(FractureError::NonFiniteToughness(grid.cell_centre(cell)));
        }
        min_cost = min_cost.min(grid.step_cost(flat));
    }
    Ok(min_cost)
}

/// Pick the index `(0=x, 1=y, 2=z)` of the dominant component of the
/// stress field's principal direction (first axis on ties). Defaults to
/// `1` (Y) when no stress field is provided.
fn principal_axis(stress: Option<&StressField>) -> usize {
    match stress {
        None => 1,
        Some(s) => {
            let abs = s.principal_dir.map(f64::abs);
            let mut best = 0;
            for (axis, &v) in abs.iter().enumerate().skip(1) {
                if v > abs[best] {
                    best = axis;
                }
            }
            best
        }
    }
}

/// 3D grid index `(ix, iy, iz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellIndex {
    x: u32,
    y: u32,
    z: u32,
}

impl CellIndex {
    fn flat(self, res: (usize, usize, usize)) -> usize {
        (self.z as usize) * res.0 * res.1 + (self.y as usize) * res.0 + (self.x as usize)
    }

    fn from_flat(flat: usize, res: (usize, usize, usize)) -> Self {
        let layer = res.0 * res.1;
        let rem = flat % layer;
        CellIndex {
            x: (rem % res.0) as u32,
            y: (rem / res.0) as u32,
            z: (flat / layer) as u32,
        }
    }

    fn coord(self, axis: usize) -> u32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

fn axis_size(res: (usize, usize, usize), axis: usize) -> usize {
    match axis {
        0 => res.0,
        1 => res.1,
        _ => res.2,
    }
}

const NEIGHBOUR_OFFSETS: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// The in-bounds 6-connected neighbours of `cell`.
fn neighbours(cell: CellIndex, res: (usize, usize, usize)) -> impl Iterator<Item = CellIndex> {
    let limit = [res.0 as i64, res.1 as i64, res.2 as i64];
    NEIGHBOUR_OFFSETS.iter().filter_map(move |&(dx, dy, dz)| {
        let n = [
            i64::from(cell.x) + dx,
            i64::from(cell.y) + dy,
            i64::from(cell.z) + dz,
        ];
        let inside = n.iter().zip(limit).all(|(&c, l)| (0..l).contains(&c));
        inside.then(|| CellIndex {
            x: n[0] as u32,
            y: n[1] as u32,
            z: n[2] as u32,
        })
    })
}

/// One of the two grid faces perpendicular to an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceSide {
    /// Coordinate 0 along the axis.
    Low,
    /// Coordinate `axis_len - 1` along the axis.
    High,
}

impl FaceSide {
    /// The pinned coordinate of this face on an axis of `axis_len >= 1` cells.
    fn pin(self, axis_len: usize) -> u32 {
        match self {
            FaceSide::Low => 0,
            FaceSide::High => (axis_len - 1) as u32,
        }
    }

    /// Whether `cell` lies on this face of `axis`.
    fn contains(self, cell: CellIndex, res: (usize, usize, usize), axis: usize) -> bool {
        cell.coord(axis) == self.pin(axis_size(res, axis))
    }
}

/// Every cell on one face of the grid, in flat-index order.
fn face_cells(res: (usize, usize, usize), axis: usize, side: FaceSide) -> Vec<CellIndex> {
    let (rx, ry, rz) = res;
    let pin = side.pin(axis_size(res, axis));
    let mut out = Vec::new();
    // Bounded: one entry per face cell.
    match axis {
        0 => {
            for z in 0..rz as u32 {
                for y in 0..ry as u32 {
                    out.push(CellIndex { x: pin, y, z });
                }
            }
        }
        1 => {
            for z in 0..rz as u32 {
                for x in 0..rx as u32 {
                    out.push(CellIndex { x, y: pin, z });
                }
            }
        }
        _ => {
            for y in 0..ry as u32 {
                for x in 0..rx as u32 {
                    out.push(CellIndex { x, y, z: pin });
                }
            }
        }
    }
    out
}

/// A* heuristic: remaining cells to the High face of `axis`, times the
/// cheapest step cost. `step_lower_bound = 0` turns A* into Dijkstra.
fn heuristic(
    cell: CellIndex,
    res: (usize, usize, usize),
    axis: usize,
    step_lower_bound: f64,
) -> f64 {
    let remaining = FaceSide::High.pin(axis_size(res, axis)) - cell.coord(axis);
    f64::from(remaining) * step_lower_bound
}

/// One A* open-set entry. `Ord` is arranged so `BinaryHeap` (a max-heap) pops
/// the entry to expand first; see the module docs for the key.
#[derive(Debug, Clone, Copy)]
struct OpenEntry {
    f: f64,
    g: f64,
    /// `|2 * coord_n - (len_n - 1)|`: distance from the section middle along
    /// the plane normal, in half-cells.
    centre_rank: u32,
    flat: usize,
}

impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .total_cmp(&self.f) // lower f first
            .then_with(|| self.g.total_cmp(&other.g)) // then higher g
            .then_with(|| other.centre_rank.cmp(&self.centre_rank)) // then nearer the middle
            .then_with(|| other.flat.cmp(&self.flat)) // then lower flat index
    }
}

impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OpenEntry {}

/// The outcome of one face-to-face search.
#[derive(Debug, Clone)]
struct PathSearch {
    /// Cells from the Low face to the High face, inclusive.
    cells: Vec<CellIndex>,
    /// Sum of the step costs of every cell on the path.
    cost: f64,
    /// Cells settled (popped and not stale) before the goal was reached.
    /// A stats hook for tests comparing A* against Dijkstra.
    #[cfg_attr(not(test), allow(dead_code))]
    expanded: usize,
}

const NO_PARENT: usize = usize::MAX;

/// Multi-source, multi-goal A* from the Low face of `axis` to its High face.
/// `normal_axis` only feeds the tie-break. Returns `None` if the High face is
/// unreachable.
fn astar_face_to_face(
    grid: &WeakpointGrid,
    axis: usize,
    normal_axis: usize,
    step_lower_bound: f64,
) -> Option<PathSearch> {
    let res = grid.resolution;
    let total = grid.len();
    let normal_len = axis_size(res, normal_axis) as i64;

    let entry_for = |cell: CellIndex, g: f64| OpenEntry {
        f: g + heuristic(cell, res, axis, step_lower_bound),
        g,
        centre_rank: (2 * i64::from(cell.coord(normal_axis)) - (normal_len - 1)).unsigned_abs()
            as u32,
        flat: cell.flat(res),
    };

    let mut g_score = vec![f64::INFINITY; total];
    let mut came_from = vec![NO_PARENT; total];
    let mut open = BinaryHeap::new();

    // Every Low-face cell is a source, at the cost of entering it.
    for cell in face_cells(res, axis, FaceSide::Low) {
        let flat = cell.flat(res);
        let g = grid.step_cost(flat);
        g_score[flat] = g;
        open.push(entry_for(cell, g));
    }

    // Bounded loop. With a consistent heuristic each cell is settled at most
    // once and pushes at most 6 entries, so pops <= face + 6 * total.
    let mut pops_remaining = total.saturating_mul(7).saturating_add(1);
    let mut expanded = 0usize;
    while let Some(entry) = open.pop() {
        if pops_remaining == 0 {
            return None;
        }
        pops_remaining -= 1;
        if entry.g > g_score[entry.flat] {
            continue; // stale: a cheaper route to this cell was found later
        }
        expanded += 1;
        let cell = CellIndex::from_flat(entry.flat, res);
        if FaceSide::High.contains(cell, res, axis) {
            return Some(PathSearch {
                cells: reconstruct_path(&came_from, entry.flat, res),
                cost: entry.g,
                expanded,
            });
        }
        for neighbour in neighbours(cell, res) {
            let n_flat = neighbour.flat(res);
            let tentative = entry.g + grid.step_cost(n_flat);
            if tentative < g_score[n_flat] {
                g_score[n_flat] = tentative;
                came_from[n_flat] = entry.flat;
                open.push(entry_for(neighbour, tentative));
            }
        }
    }
    None
}

fn reconstruct_path(came_from: &[usize], end: usize, res: (usize, usize, usize)) -> Vec<CellIndex> {
    let mut path = Vec::new();
    let mut current = end;
    // Bounded by the total cell count.
    let mut steps_remaining = came_from.len() + 1;
    while current != NO_PARENT && steps_remaining > 0 {
        path.push(CellIndex::from_flat(current, res));
        current = came_from[current];
        steps_remaining -= 1;
    }
    path.reverse();
    path
}

/// Fit the plane to the transverse crack paths: normal along `normal_axis`,
/// offset at the mean `normal_axis` coordinate of every path cell centre,
/// confidence from the mean cell cost over every path cell.
fn fit_plane(paths: &[PathSearch], grid: &WeakpointGrid, normal_axis: usize) -> FracturePlane {
    let (lo, hi) = grid.bounds;
    let n_len = axis_size(grid.resolution, normal_axis);

    let mut sum_offset = 0.0_f64;
    let mut sum_cost = 0.0_f64;
    let mut count = 0usize;
    for path in paths {
        for cell in &path.cells {
            sum_offset += axis_centre(
                lo[normal_axis],
                hi[normal_axis],
                n_len,
                cell.coord(normal_axis) as usize,
            );
        }
        sum_cost += path.cost;
        count += path.cells.len();
    }
    let n = count.max(1) as f64;

    let mut normal = [0.0_f64; 3];
    normal[normal_axis] = 1.0;

    FracturePlane {
        normal,
        offset: sum_offset / n,
        weakpoint_confidence: 1.0 / (1.0 + sum_cost / n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{splitmix64, unit_from_bits};

    // --- Transplanted from the engine (pt_periodica_fracture.rs) -----------

    #[test]
    fn zero_resolution_rejected() {
        let t = |_p: [f64; 3]| 1.0;
        let r = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (0, 1, 1), &t);
        assert!(matches!(r, Err(FractureError::ZeroResolution)));
    }

    #[test]
    fn nan_toughness_rejected() {
        let t = |_p: [f64; 3]| f64::NAN;
        let r = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &t);
        assert!(matches!(r, Err(FractureError::NonFiniteToughness(_))));
    }

    #[test]
    fn build_yields_correct_cell_count() {
        let t = |_p: [f64; 3]| 1.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &t).unwrap();
        assert_eq!(g.len(), 64);
    }

    #[test]
    fn empty_grid_yields_no_paths() {
        let g = WeakpointGrid {
            resolution: (0, 0, 0),
            bounds: ([0.0; 3], [1.0; 3]),
            cost: Vec::new(),
        };
        let paths = find_fracture_paths(&g, None).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn nonempty_grid_yields_at_least_one_plane() {
        let t = |_p: [f64; 3]| 1.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &t).unwrap();
        let paths = find_fracture_paths(&g, None).unwrap();
        assert!(!paths.is_empty());
    }

    #[test]
    fn default_propagation_is_y_axis() {
        // No stress field -> the plane normal is Y by default.
        let t = |_p: [f64; 3]| 1.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &t).unwrap();
        let planes = find_fracture_paths(&g, None).unwrap();
        assert_eq!(planes.len(), 1);
        let p = planes[0];
        assert!((p.normal[0]).abs() < 1.0e-9);
        assert!((p.normal[1] - 1.0).abs() < 1.0e-9);
        assert!((p.normal[2]).abs() < 1.0e-9);
    }

    #[test]
    fn x_aligned_stress_picks_x_axis() {
        let t = |_p: [f64; 3]| 1.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &t).unwrap();
        let stress = StressField {
            principal_dir: [1.0, 0.1, 0.05],
            principal_magnitude: 100.0,
        };
        let planes = find_fracture_paths(&g, Some(&stress)).unwrap();
        assert_eq!(planes.len(), 1);
        let p = planes[0];
        // Stress is mostly +x -> normal should be (1, 0, 0).
        assert!((p.normal[0] - 1.0).abs() < 1.0e-9);
        assert!((p.normal[1]).abs() < 1.0e-9);
        assert!((p.normal[2]).abs() < 1.0e-9);
    }

    #[test]
    fn confidence_is_higher_for_lower_cost_path() {
        let low_cost = |_p: [f64; 3]| 1.0;
        let high_cost = |_p: [f64; 3]| 10.0;
        let g_low = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &low_cost).unwrap();
        let g_high = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &high_cost).unwrap();
        let p_low = find_fracture_paths(&g_low, None).unwrap()[0];
        let p_high = find_fracture_paths(&g_high, None).unwrap()[0];
        assert!(
            p_low.weakpoint_confidence > p_high.weakpoint_confidence,
            "lower cost ({}) should give higher confidence than ({})",
            p_low.weakpoint_confidence,
            p_high.weakpoint_confidence,
        );
    }

    #[test]
    fn weak_band_attracts_path_offset() {
        // Construct a grid where the y=mid layer is much weaker (lower
        // toughness) than the rest. The fracture should pass through
        // that band -- its offset should be near grid centre.
        let mid_y = 0.5;
        let band_width = 0.15;
        let toughness = move |p: [f64; 3]| -> f64 {
            if (p[1] - mid_y).abs() < band_width {
                0.001 // very weak -- easy to fracture
            } else {
                100.0 // strong
            }
        };
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (8, 8, 8), &toughness).unwrap();
        let planes = find_fracture_paths(&g, None).unwrap();
        assert_eq!(planes.len(), 1);
        let p = planes[0];
        // Plane's Y offset should be close to mid_y because A* found
        // the cheap path through the weak band.
        assert!(
            (p.offset - mid_y).abs() < 0.2,
            "offset {} should be near {}",
            p.offset,
            mid_y,
        );
    }

    #[test]
    fn confidence_is_in_unit_interval() {
        let t = |_p: [f64; 3]| 1.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (4, 4, 4), &t).unwrap();
        let p = find_fracture_paths(&g, None).unwrap()[0];
        assert!(p.weakpoint_confidence > 0.0);
        assert!(p.weakpoint_confidence <= 1.0);
    }

    #[test]
    fn principal_axis_selects_dominant_component() {
        let s_x = StressField {
            principal_dir: [1.0, 0.0, 0.0],
            principal_magnitude: 1.0,
        };
        let s_y = StressField {
            principal_dir: [0.0, 1.0, 0.0],
            principal_magnitude: 1.0,
        };
        let s_z = StressField {
            principal_dir: [0.0, 0.0, 1.0],
            principal_magnitude: 1.0,
        };
        let s_neg_x = StressField {
            principal_dir: [-1.0, 0.0, 0.0],
            principal_magnitude: 1.0,
        };
        assert_eq!(principal_axis(Some(&s_x)), 0);
        assert_eq!(principal_axis(Some(&s_y)), 1);
        assert_eq!(principal_axis(Some(&s_z)), 2);
        assert_eq!(principal_axis(Some(&s_neg_x)), 0); // abs picks the magnitude
        assert_eq!(principal_axis(None), 1);
    }

    // --- Crack placement (the offset bug) ----------------------------------

    #[test]
    fn off_centre_weak_band_sets_the_offset() {
        // The engine ran A* *along* the normal, so its offset was always the
        // grid centre. A band at y = 0.3 must now pull the plane to 0.3.
        let res = (8, 20, 8);
        let cell_h = 1.0 / res.1 as f64;
        let toughness = |p: [f64; 3]| if (p[1] - 0.3).abs() < 0.03 { 0.5 } else { 40.0 };
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), res, &toughness).unwrap();
        let p = find_fracture_paths(&g, None).unwrap()[0];
        assert_eq!(p.normal, [0.0, 1.0, 0.0]);
        assert!(
            (p.offset - 0.3).abs() <= cell_h,
            "offset {} not within one cell of 0.3",
            p.offset
        );
    }

    #[test]
    fn x_stress_with_weak_x_band_sets_the_x_offset() {
        let res = (20, 8, 8);
        let cell_w = 1.0 / res.0 as f64;
        let toughness = |p: [f64; 3]| {
            if (p[0] - 0.7).abs() < 0.03 {
                1.0e6
            } else {
                5.0e7
            }
        };
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), res, &toughness).unwrap();
        let stress = StressField {
            principal_dir: [1.0, 0.1, 0.05],
            principal_magnitude: 2.0e8,
        };
        let p = find_fracture_paths(&g, Some(&stress)).unwrap()[0];
        assert_eq!(p.normal, [1.0, 0.0, 0.0]);
        assert!(
            (p.offset - 0.7).abs() <= cell_w,
            "offset {} not within one cell of 0.7",
            p.offset
        );
    }

    #[test]
    fn offset_is_in_world_coordinates() {
        // Same band, translated and scaled bounds.
        let bounds = ([-2.0, 10.0, 0.0], [2.0, 12.0, 1.0]);
        let toughness = |p: [f64; 3]| if (p[1] - 11.5).abs() < 0.06 { 1.0 } else { 9.0 };
        let g = WeakpointGrid::build(bounds, (6, 20, 6), &toughness).unwrap();
        let p = find_fracture_paths(&g, None).unwrap()[0];
        assert!((p.offset - 11.5).abs() <= 0.1, "{}", p.offset);
    }

    #[test]
    fn uniform_grid_cracks_through_the_section_centre_deterministically() {
        let t = |_p: [f64; 3]| 3.0e7;
        for res in [(8, 8, 8), (5, 9, 4), (6, 7, 3)] {
            let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), res, &t).unwrap();
            let a = find_fracture_paths(&g, None).unwrap()[0];
            let b = find_fracture_paths(&g, None).unwrap()[0];
            assert_eq!(a, b, "search must be deterministic");
            let half_cell = 0.5 / res.1 as f64;
            assert!(
                (a.offset - 0.5).abs() <= half_cell + 1e-12,
                "{res:?}: offset {}",
                a.offset
            );
        }
    }

    // --- Search correctness ------------------------------------------------

    /// Deterministic pseudo-random costs in [1, 10).
    fn noisy_grid(res: (usize, usize, usize), seed: u64) -> WeakpointGrid {
        let mut state = seed;
        let cost = (0..res.0 * res.1 * res.2)
            .map(|_| (1.0 + 9.0 * unit_from_bits(splitmix64(&mut state))) as f32)
            .collect();
        WeakpointGrid {
            resolution: res,
            bounds: ([0.0; 3], [1.0; 3]),
            cost,
        }
    }

    #[test]
    fn astar_matches_dijkstra_cost_with_fewer_expansions() {
        for (res, seed) in [((12, 10, 9), 1), ((9, 14, 7), 2), ((16, 6, 11), 3)] {
            let g = noisy_grid(res, seed);
            let min_cost = validated_min_step_cost(&g).unwrap();
            assert!(min_cost >= 1.0);
            for axis in 0..3 {
                let normal = (axis + 1) % 3;
                let astar = astar_face_to_face(&g, axis, normal, min_cost).unwrap();
                let dijkstra = astar_face_to_face(&g, axis, normal, 0.0).unwrap();
                assert!(
                    (astar.cost - dijkstra.cost).abs() <= 1e-9 * dijkstra.cost,
                    "{res:?} axis {axis}: A* {} vs Dijkstra {}",
                    astar.cost,
                    dijkstra.cost
                );
                assert!(
                    astar.expanded < dijkstra.expanded,
                    "{res:?} axis {axis}: A* expanded {} vs Dijkstra {}",
                    astar.expanded,
                    dijkstra.expanded
                );
                // The reported cost is the sum of the path's cells.
                let sum: f64 = astar.cells.iter().map(|c| g.step_cost(c.flat(res))).sum();
                assert!((sum - astar.cost).abs() <= 1e-9 * sum);
            }
        }
    }

    /// Exact cost-to-go to the High face of `axis` (excluding the cell
    /// itself), by Bellman-Ford relaxation on a small grid.
    fn exact_cost_to_go(g: &WeakpointGrid, axis: usize) -> Vec<f64> {
        let res = g.resolution;
        let mut dist = vec![f64::INFINITY; g.len()];
        for cell in face_cells(res, axis, FaceSide::High) {
            dist[cell.flat(res)] = 0.0;
        }
        loop {
            let mut changed = false;
            for flat in 0..g.len() {
                let cell = CellIndex::from_flat(flat, res);
                for n in neighbours(cell, res) {
                    let nf = n.flat(res);
                    let via = g.step_cost(nf) + dist[nf];
                    if via < dist[flat] {
                        dist[flat] = via;
                        changed = true;
                    }
                }
            }
            if !changed {
                return dist;
            }
        }
    }

    #[test]
    fn heuristic_is_admissible_and_consistent() {
        let g = noisy_grid((6, 5, 7), 17);
        let res = g.resolution;
        let min_cost = validated_min_step_cost(&g).unwrap();
        for axis in 0..3 {
            let truth = exact_cost_to_go(&g, axis);
            for (flat, &exact) in truth.iter().enumerate() {
                let cell = CellIndex::from_flat(flat, res);
                let h = heuristic(cell, res, axis, min_cost);
                assert!(h <= exact + 1e-12, "axis {axis} {cell:?}: h {h} > {exact}");
                for n in neighbours(cell, res) {
                    let hn = heuristic(n, res, axis, min_cost);
                    assert!(h <= g.step_cost(n.flat(res)) + hn + 1e-12);
                }
            }
        }
    }

    #[test]
    fn high_face_goal_on_non_cubic_resolution() {
        let res = (3, 7, 5);
        let t = |_p: [f64; 3]| 2.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), res, &t).unwrap();

        for axis in 0..3 {
            let len = axis_size(res, axis);
            // Both face iterators: right size, right pinned coordinate.
            let low = face_cells(res, axis, FaceSide::Low);
            let high = face_cells(res, axis, FaceSide::High);
            assert_eq!(low.len(), g.len() / len);
            assert_eq!(high.len(), g.len() / len);
            assert!(low.iter().all(|c| c.coord(axis) == 0));
            assert!(high.iter().all(|c| c.coord(axis) as usize == len - 1));

            let normal = (axis + 1) % 3;
            let s = astar_face_to_face(&g, axis, normal, 2.0).unwrap();
            assert_eq!(s.cells.len(), len, "axis {axis}: straight path expected");
            assert!(FaceSide::Low.contains(s.cells[0], res, axis));
            assert!(FaceSide::High.contains(*s.cells.last().unwrap(), res, axis));
            for w in s.cells.windows(2) {
                let step: u32 = (0..3).map(|a| w[0].coord(a).abs_diff(w[1].coord(a))).sum();
                assert_eq!(step, 1, "6-connected steps only");
            }
            // Straight down a uniform grid: A* expands only the path itself.
            assert_eq!(s.expanded, len);
        }

        for dir in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let stress = StressField {
                principal_dir: dir,
                principal_magnitude: 1.0,
            };
            let planes = find_fracture_paths(&g, Some(&stress)).unwrap();
            assert_eq!(planes.len(), 1);
            assert_eq!(planes[0].normal, dir);
        }
    }

    #[test]
    fn single_cell_axis_is_its_own_goal() {
        let t = |_p: [f64; 3]| 1.0;
        let g = WeakpointGrid::build(([0.0; 3], [1.0; 3]), (1, 4, 4), &t).unwrap();
        let s = astar_face_to_face(&g, 0, 1, 1.0).unwrap();
        assert_eq!(s.cells.len(), 1);
        assert_eq!(find_fracture_paths(&g, None).unwrap().len(), 1);
    }

    #[test]
    fn malformed_hand_built_grids_are_errors_not_panics() {
        let short = WeakpointGrid {
            resolution: (2, 2, 2),
            bounds: ([0.0; 3], [1.0; 3]),
            cost: vec![1.0; 7],
        };
        assert_eq!(
            find_fracture_paths(&short, None),
            Err(FractureError::CostLengthMismatch {
                expected: 8,
                actual: 7
            })
        );

        let mut nan = WeakpointGrid {
            resolution: (2, 2, 2),
            bounds: ([0.0; 3], [1.0; 3]),
            cost: vec![1.0; 8],
        };
        nan.cost[7] = f32::NAN;
        assert_eq!(
            find_fracture_paths(&nan, None),
            Err(FractureError::NonFiniteToughness([0.75, 0.75, 0.75]))
        );
    }

    #[test]
    fn open_set_order_is_total_and_as_documented() {
        let e = |f, g, centre_rank, flat| OpenEntry {
            f,
            g,
            centre_rank,
            flat,
        };
        let mut heap = BinaryHeap::from(vec![
            e(2.0, 1.0, 0, 0),
            e(1.0, 0.5, 3, 9),
            e(1.0, 0.9, 3, 8),
            e(1.0, 0.9, 1, 7),
            e(1.0, 0.9, 1, 4),
        ]);
        let order: Vec<usize> = std::iter::from_fn(|| heap.pop().map(|x| x.flat)).collect();
        // Lowest f; then highest g; then nearest the middle; then lowest index.
        assert_eq!(order, vec![4, 7, 8, 9, 0]);
    }
}
