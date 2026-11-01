// TEMPORARY benchmark -- deleted before commit.
use periodica_runtime::HaltonSampler;
use std::hint::black_box;
use std::time::Instant;

const ONE_BELOW: f64 = 1.0 - f64::EPSILON / 2.0;

#[inline(never)]
fn old_ri(base: u32, index: u64) -> f64 {
    let b = u64::from(base);
    let inv_base = 1.0 / f64::from(base);
    let mut remaining = index;
    let mut scale = inv_base;
    let mut acc = 0.0_f64;
    while remaining > 0 {
        acc += (remaining % b) as f64 * scale;
        remaining /= b;
        scale *= inv_base;
    }
    acc.min(ONE_BELOW)
}
#[inline(always)]
fn single<const B: u64>(index: u64) -> f64 {
    let inv_base = 1.0 / B as f64;
    let mut remaining = index;
    let mut scale = inv_base;
    let mut acc = 0.0_f64;
    while remaining > 0 {
        acc += (remaining % B) as f64 * scale;
        remaining /= B;
        scale *= inv_base;
    }
    acc.min(ONE_BELOW)
}
#[inline(always)]
fn pair<const B: u64>(index: u64) -> f64 {
    let inv_base = 1.0 / B as f64;
    let mut remaining = index;
    let mut scale = inv_base;
    let mut acc = 0.0_f64;
    while remaining > 0 {
        let p = remaining % (B * B);
        remaining /= B * B;
        acc += (p % B) as f64 * scale;
        scale *= inv_base;
        acc += (p / B) as f64 * scale;
        scale *= inv_base;
    }
    acc.min(ONE_BELOW)
}
#[inline(always)]
fn b2(index: u64) -> f64 {
    if index < (1 << 53) {
        index.reverse_bits() as f64 * (1.0 / 18_446_744_073_709_551_616.0)
    } else {
        old_ri(2, index)
    }
}

fn time(name: &str, n: u64, f: &dyn Fn(u64) -> [f64; 3], best: &mut f64) {
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        let p = f(black_box(i));
        acc += p[0] + p[1] + p[2];
    }
    black_box(acc);
    let v = t.elapsed().as_secs_f64() / n as f64 * 1e9;
    if v < *best {
        *best = v;
    }
    let _ = name;
}

fn main() {
    let n = 1u64 << 20;
    let s = HaltonSampler::new(42);
    let old = |i: u64| [old_ri(2, i), old_ri(3, i), old_ri(5, i)];
    let sgl = |i: u64| [b2(i), single::<3>(i), single::<5>(i)];
    let pr = |i: u64| [b2(i), pair::<3>(i), pair::<5>(i)];
    let lib = |i: u64| s.point(i);
    let mut b = [f64::MAX; 4];
    for _ in 0..15 {
        time("old", n, &old, &mut b[0]);
        time("single", n, &sgl, &mut b[1]);
        time("pair", n, &pr, &mut b[2]);
        time("lib", n, &lib, &mut b[3]);
    }
    println!("old {:.2}  single {:.2}  pair {:.2}  lib(point) {:.2} ns/pt", b[0], b[1], b[2], b[3]);
}
