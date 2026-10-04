// TEMPORARY benchmark -- deleted before commit.
use periodica_runtime::sampling::{splitmix64, unit_from_bits};
use periodica_runtime::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    // Halton points.
    let s = HaltonSampler::new(42);
    let n = 1u64 << 22;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let mut acc = 0.0;
        for i in 0..n {
            let p = s.point(black_box(i));
            acc += p[0] + p[1] + p[2];
        }
        black_box(acc);
        best = best.min(t.elapsed().as_secs_f64() / n as f64 * 1e9);
    }
    println!("halton point: {best:.2} ns/pt");

    // integrate_mass at 2^20.
    let sdf = |p: [f64; 3]| {
        ((p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2) + (p[2] - 0.5).powi(2)).sqrt() - 0.4
    };
    let rho = |_p: [f64; 3]| 7874.0;
    let steps = 1u64 << 20;
    let mut best = f64::MAX;
    let mut m = 0.0;
    for _ in 0..5 {
        let t = Instant::now();
        let r = integrate_mass(&sdf, &rho, ([0.0; 3], [1.0; 3]), black_box(steps), 7).unwrap();
        best = best.min(t.elapsed().as_secs_f64() / steps as f64 * 1e9);
        m = r.total_mass_kg;
    }
    println!("integrate_mass 2^20: {best:.2} ns/sample (mass {m:.17e})");

    // Fracture 64^3, costs [1,10)*1e7.
    let res = (64, 64, 64);
    let mut state = 5u64;
    let cost: Vec<f32> = (0..res.0 * res.1 * res.2)
        .map(|_| ((1.0 + 9.0 * unit_from_bits(splitmix64(&mut state))) * 1e7) as f32)
        .collect();
    let grid = WeakpointGrid {
        resolution: res,
        bounds: ([0.0; 3], [1.0; 3]),
        cost,
    };
    let mut best = f64::MAX;
    let mut plane = None;
    for _ in 0..5 {
        let t = Instant::now();
        let p = find_fracture_paths(black_box(&grid), None).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        plane = Some(p[0]);
    }
    println!("fracture 64^3 noisy: {best:.2} ms  {plane:?}");
}
