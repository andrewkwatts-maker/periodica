use periodica_mat::{
    mix, GrainStructure, MaterialRecord, Microstructure, Phase, PropertyId, PropertyTable, Source,
};
use periodica_runtime::volume::{bake, MaterialVolume, Probe};

const PROPS: [PropertyId; 2] = [PropertyId::YoungsModulus, PropertyId::Density];

fn table(e: f64, rho: f64) -> PropertyTable {
    let mut t = PropertyTable::new();
    t.set(PropertyId::YoungsModulus, e, Source::Datasheet);
    t.set(PropertyId::Density, rho, Source::Datasheet);
    t
}

fn phase(name: &str, f: f64, e: f64, rho: f64) -> Phase {
    Phase { name: name.into(), volume_fraction: f, properties: table(e, rho), ..Phase::default() }
}

fn steel() -> MaterialRecord {
    MaterialRecord {
        name: "Steel-1018 fixture".into(),
        bulk: table(205.0e9, 7870.0),
        phases: vec![phase("ferrite", 0.8, 190.0e9, 7870.0), phase("pearlite", 0.2, 250.0e9, 7800.0)],
        micro: Some(Microstructure {
            grains: GrainStructure { average_size_m: 50.0e-6, seed_density_per_m3: 8.0e12, ..Default::default() },
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn probe_at(i: u64, half: f64) -> Probe {
    let f = |k: u64| ((i * 7919 + k * 104_729) % 10_007) as f64 * 1.0e-5;
    Probe { point: [f(1), f(2), f(3)], half_extent_m: half, direction: None }
}

fn probe_e(v: &MaterialVolume, p: &Probe) -> f64 {
    v.probe(p, &PROPS[..1]).unwrap()[0]
}

#[test]
fn coarse_probe_is_bulk_bit_for_bit() {
    let rec = steel();
    let v = MaterialVolume::new(&rec, 1);
    let out = v.probe(&probe_at(3, v.rve_m), &PROPS).unwrap();
    assert_eq!(out[0].to_bits(), 205.0e9f64.to_bits());
    assert_eq!(out[1].to_bits(), 7870.0f64.to_bits());
    let mut bare = steel();
    bare.micro = None;
    let b = MaterialVolume::new(&bare, 1);
    assert_eq!(probe_e(&b, &probe_at(3, 1.0e-6)).to_bits(), 205.0e9f64.to_bits());
}

#[test]
fn seed_density_and_phase_fractions() {
    let v = MaterialVolume::new(&steel(), 11);
    let g = v.grains().unwrap();
    let n = 30u32;
    let measured = g.count_seeds([n; 3]) as f64 / (f64::from(n).powi(3) * g.cell_m().powi(3));
    let err = (measured / 8.0e12 - 1.0).abs();
    println!("seed density error {err:.4}");
    assert!(err < 0.03);
    let (mut hits, total) = (0u32, 40_000u32);
    for i in 0..total {
        let u = |k: u32| f64::from(i.wrapping_mul(2_654_435_761u32.wrapping_add(k * 40_503)) % 100_003) / 100_003.0;
        let p = [u(1), u(2), u(3)].map(|x| x * 20.0 * g.cell_m());
        hits += u32::from(g.grain_at(p).phase == 1);
    }
    let ferr = (f64::from(hits) / f64::from(total) - 0.2).abs();
    println!("pearlite fraction error {ferr:.4}");
    assert!(ferr < 0.02);
}

fn scale(rec: &MaterialRecord) -> f64 {
    let recs: Vec<MaterialRecord> = rec
        .phases
        .iter()
        .map(|p| MaterialRecord { name: p.name.clone(), bulk: p.properties.clone(), ..Default::default() })
        .collect();
    let entries: Vec<(&MaterialRecord, f64)> = recs.iter().zip([0.8, 0.2]).collect();
    205.0e9 / mix(&entries, PropertyId::YoungsModulus).unwrap()
}

#[test]
fn fine_probes_mean_bounds_and_variance() {
    let rec = steel();
    let v = MaterialVolume::new(&rec, 5);
    let k = scale(&rec);
    let (lo, hi) = (190.0e9 * k, 250.0e9 * k);
    let mut variances = Vec::new();
    for half in [12.5e-6, 25.0e-6, 100.0e-6] {
        let vals: Vec<f64> = (0..400).map(|i| probe_e(&v, &probe_at(i, half))).collect();
        assert!(vals.iter().all(|&x| x >= lo * (1.0 - 1e-12) && x <= hi * (1.0 + 1e-12)));
        let mean = vals.iter().sum::<f64>() / 400.0;
        let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 400.0;
        println!("half {half:e}: mean err {:.5} var {var:e}", (mean / 205.0e9 - 1.0).abs());
        assert!((mean / 205.0e9 - 1.0).abs() < 0.01);
        variances.push(var);
    }
    assert!(variances[0] > variances[1] && variances[1] > variances[2], "{variances:?}");
}

#[test]
fn determinism_and_seed_sensitivity() {
    let rec = steel();
    let (a, b, c) = (MaterialVolume::new(&rec, 1), MaterialVolume::new(&rec, 1), MaterialVolume::new(&rec, 2));
    let ps: Vec<Probe> = (0..50).map(|i| probe_at(i, 20.0e-6)).collect();
    let run = |v: &MaterialVolume| ps.iter().map(|p| probe_e(v, p).to_bits()).collect::<Vec<_>>();
    assert_eq!(run(&a), run(&b));
    assert_ne!(run(&a), run(&c));
}

#[test]
fn bake_equals_probe_at_voxel_centres() {
    let v = MaterialVolume::new(&steel(), 9);
    let voxel = 30.0e-6;
    let vol = bake(&v, ([0.0; 3], [4.0 * voxel, 3.0 * voxel, 2.0 * voxel]), voxel, &PROPS).unwrap();
    assert_eq!(vol.dims, [4, 3, 2]);
    for (x, y, z) in [(0, 0, 0), (3, 2, 1), (1, 2, 0)] {
        let c = [x, y, z].map(|i| (i as f64 + 0.5) * voxel);
        let p = Probe { point: c, half_extent_m: 0.5 * voxel, direction: None };
        let want = v.probe(&p, &PROPS).unwrap();
        let idx = x + 4 * (y + 3 * z);
        assert_eq!(vol.data[0][idx], want[0] as f32);
        assert_eq!(vol.data[1][idx], want[1] as f32);
    }
    assert!(bake(&v, ([0.0; 3], [1.0; 3]), 1.0e-6, &PROPS).is_err());
}
