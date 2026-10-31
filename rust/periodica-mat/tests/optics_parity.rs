//! Values from the Python prototype `optical_properties.py` at five wavelengths.

use periodica_mat::optics::cauchy::Cauchy;
use periodica_mat::optics::drude_lorentz::{DrudeLorentz, Oscillator};
use periodica_mat::optics::sellmeier::Sellmeier;

const WL_NM: [f64; 5] = [400.0, 500.0, 587.6, 700.0, 900.0];
const SELLMEIER_N: [f64; 5] = [1.5308485382492993, 1.5214144757734767, 1.5167984379050088, 1.513063997196311, 1.5089968620388343];
const CAUCHY_N: [f64; 5] = [1.52890625, 1.5176, 1.5124238379792516, 1.508579758433986, 1.5050906873952141];
const DRUDE_N: [f64; 5] = [0.035013498410410175, 0.05342935439610381, 0.07296059836277356, 0.10264148337658309, 0.1682401980199943];
const DRUDE_K: [f64; 5] = [2.735707375077663, 3.5005053731888705, 4.159353844336247, 4.996202211054532, 6.471791021561446];
const DL_N: [f64; 5] = [0.7980509180188352, 0.217202357860447, 0.13979503237340798, 0.15340379074487812, 0.22586433969755723];
const DL_K: [f64; 5] = [2.701698701429443, 2.5431719872668257, 3.3201400745122243, 4.159362810378252, 5.542209831449279];

fn close(got: f64, want: f64, what: &str, nm: f64) {
    assert!((got - want).abs() < 1e-6, "{what} at {nm} nm: {got} vs {want}");
}

#[test]
fn matches_python_prototype() {
    let bk7 = Sellmeier::bk7();
    let cauchy = Cauchy { a: 1.5, b: 0.004, c: 0.0001 };
    let drude = DrudeLorentz { epsilon_inf: 1.0, plasma_ev: 9.03, damping_ev: 0.07, oscillators: vec![] };
    let dl = DrudeLorentz {
        epsilon_inf: 2.0,
        plasma_ev: 8.0,
        damping_ev: 0.1,
        oscillators: vec![Oscillator { strength: 0.7, resonance_ev: 3.0, damping_ev: 0.4 }],
    };
    for (i, nm) in WL_NM.iter().enumerate() {
        let l = nm * 1e-9;
        close(bk7.refractive_index(l), SELLMEIER_N[i], "sellmeier", *nm);
        close(cauchy.refractive_index(l), CAUCHY_N[i], "cauchy", *nm);
        let (n, k) = drude.nk(l);
        close(n, DRUDE_N[i], "drude n", *nm);
        close(k, DRUDE_K[i], "drude k", *nm);
        let (n, k) = dl.nk(l);
        close(n, DL_N[i], "drude-lorentz n", *nm);
        close(k, DL_K[i], "drude-lorentz k", *nm);
    }
}
