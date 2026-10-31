//! CIE 1931 2-degree colour matching functions (Wyman, Sloan and Shirley 2013
//! multi-lobe fit) as Arithma expressions, tabulated once at 1 nm.

use std::sync::OnceLock;

use arithma_core::ArithmaExpression as E;

use super::{eval, num, square, sum, var};

pub const LAMBDA_MIN_NM: f64 = 360.0;
pub const LAMBDA_MAX_NM: f64 = 830.0;

const XBAR: [(f64, f64, f64, f64); 3] = [
    (1.056, 599.8, 37.9, 31.0),
    (0.362, 442.0, 16.0, 26.7),
    (-0.065, 501.1, 20.4, 26.2),
];
const YBAR: [(f64, f64, f64, f64); 2] = [(0.821, 568.8, 46.9, 40.5), (0.286, 530.9, 16.3, 31.1)];
const ZBAR: [(f64, f64, f64, f64); 2] = [(1.217, 437.0, 11.8, 36.0), (0.681, 459.0, 26.0, 13.8)];

const XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

fn lobe((amp, mu, s1, s2): (f64, f64, f64, f64)) -> E {
    let side = E::func(arithma_core::ArithmaFunction::Sign, vec![E::sub(num(mu), var("nm"))]);
    let blend = E::mul(num(0.5), E::add(num(1.0), side));
    let sigma = E::add(num(s2), E::mul(num(s1 - s2), blend));
    let z = E::div(E::sub(var("nm"), num(mu)), sigma);
    E::mul(num(amp), E::exp(E::neg(E::mul(num(0.5), square(z)))))
}

fn curve(lobes: &[(f64, f64, f64, f64)]) -> E {
    sum(lobes.iter().map(|l| lobe(*l)).collect())
}

pub struct Table {
    pub cmf: Vec<[f64; 3]>,
    pub white: [f64; 3],
    pub y_sum: f64,
}

pub fn rgb_of_xyz(xyz: [f64; 3]) -> [f64; 3] {
    let row = |r: &[f64; 3]| r[0] * xyz[0] + r[1] * xyz[1] + r[2] * xyz[2];
    [row(&XYZ_TO_RGB[0]), row(&XYZ_TO_RGB[1]), row(&XYZ_TO_RGB[2])]
}

fn build() -> Table {
    let (cx, cy, cz) = (curve(&XBAR), curve(&YBAR), curve(&ZBAR));
    let count = (LAMBDA_MAX_NM - LAMBDA_MIN_NM) as usize + 1;
    let cmf: Vec<[f64; 3]> = (0..count)
        .map(|i| {
            let v = [("nm", LAMBDA_MIN_NM + i as f64)];
            [eval(&cx, &v), eval(&cy, &v), eval(&cz, &v)]
        })
        .collect();
    let sums = cmf.iter().fold([0.0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
    let y_sum = sums[1];
    let flat = rgb_of_xyz([sums[0] / y_sum, 1.0, sums[2] / y_sum]);
    Table { cmf, white: flat, y_sum }
}

pub fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(build)
}
