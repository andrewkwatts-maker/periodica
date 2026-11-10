//! Direction-dependent stiffness and strength of fibre composites, read from a sheet's `Field` block.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Names the inputs a direction query needs but the record lacks; `"direction"` marks a zero vector.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("orthotropy incomplete, missing: {}", missing.join(", "))]
pub struct OrthotropyIncomplete {
    /// Key names (`E1`, `G12`, `nu12`, `X1`, ...) that are absent.
    pub missing: Vec<&'static str>,
}

/// Orthotropic elastic and strength data in SI (Pa), with unknown entries left as `None`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Orthotropy {
    /// Material axis 1 (the fibre direction), unit length.
    pub axis_1: [f64; 3],
    /// Material axis 2; `None` means transversely isotropic about axis 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis_2: Option<[f64; 3]>,
    /// Young's moduli E1, E2, E3.
    pub e: [Option<f64>; 3],
    /// Shear moduli G12, G13, G23.
    pub g: [Option<f64>; 3],
    /// Poisson ratios nu12, nu13, nu23.
    pub nu: [Option<f64>; 3],
    /// Tensile strengths X1, X2.
    pub tensile: [Option<f64>; 2],
    /// Compressive strengths along 1 and 2.
    pub compressive: [Option<f64>; 2],
    /// In-plane shear strength S.
    pub shear: Option<f64>,
}

impl Orthotropy {
    /// Reads `axial`, `transverse` and `fiber_direction` from a sheet `Field` object, filling the transversely isotropic relations.
    pub fn from_field(field: &Map<String, Value>) -> Option<Self> {
        let axial = field.get("axial")?.as_object()?;
        let transverse = field.get("transverse").and_then(Value::as_object);
        let t = |k: &str, s: f64| transverse.and_then(|m| num(m, k)).map(|v| v * s);
        let a = |k: &str, s: f64| num(axial, k).map(|v| v * s);
        let e2 = t("YoungsModulus_GPa", 1.0e9);
        let g12 = a("ShearModulus_GPa", 1.0e9);
        let nu12 = a("PoissonRatio", 1.0);
        let nu23 = t("PoissonRatio", 1.0);
        let g23 = t("ShearModulus_GPa", 1.0e9).or_else(|| Some(e2? / (2.0 * (1.0 + nu23?))));
        Some(Self {
            axis_1: unit(fiber_direction(field)?)?,
            axis_2: None,
            e: [a("YoungsModulus_GPa", 1.0e9), e2, e2],
            g: [g12, g12, g23],
            nu: [nu12, nu12, nu23],
            tensile: [
                a("TensileStrength_MPa", 1.0e6),
                t("TensileStrength_MPa", 1.0e6),
            ],
            compressive: [
                a("CompressiveStrength_MPa", 1.0e6),
                t("CompressiveStrength_MPa", 1.0e6),
            ],
            shear: a("ShearStrength_MPa", 1.0e6).or_else(|| t("ShearStrength_MPa", 1.0e6)),
        })
    }

    /// Young's modulus along `d` by the compliance transformation, in Pa.
    pub fn modulus_along(&self, d: [f64; 3]) -> Result<f64, OrthotropyIncomplete> {
        let mut missing = Vec::new();
        let dir = self.local(d, &mut missing);
        let e1 = need(self.e[0], "E1", &mut missing);
        let e2 = need(self.e[1], "E2", &mut missing);
        let g12 = need(self.g[0], "G12", &mut missing);
        let nu12 = need(self.nu[0], "nu12", &mut missing);
        let full = self.axis_2.map(|_| self.full_terms(&mut missing));
        fail_if(&missing)?;
        let l = dir.unwrap_or([0.0; 3]);
        // direct-float: periodica has no Arithma dependency
        let mut inv = l[0].powi(4) / e1 + (1.0 / g12 - 2.0 * nu12 / e1) * l[0] * l[0] * l[1] * l[1];
        inv += match full {
            Some([e3, g13, g23, nu13, nu23]) => {
                l[1].powi(4) / e2
                    + l[2].powi(4) / e3
                    + (1.0 / g13 - 2.0 * nu13 / e1) * l[0] * l[0] * l[2] * l[2]
                    + (1.0 / g23 - 2.0 * nu23 / e2) * l[1] * l[1] * l[2] * l[2]
            }
            None => l[1].powi(4) / e2,
        };
        Ok(1.0 / inv)
    }

    /// Uniaxial tensile strength along `d` by Tsai-Hill from X1, X2 and S, in Pa.
    pub fn tensile_along(&self, d: [f64; 3]) -> Result<f64, OrthotropyIncomplete> {
        let mut missing = Vec::new();
        let dir = self.local(d, &mut missing);
        let x1 = need(self.tensile[0], "X1", &mut missing);
        let x2 = need(self.tensile[1], "X2", &mut missing);
        let s = need(self.shear, "S", &mut missing);
        fail_if(&missing)?;
        let c2 = dir.map_or(0.0, |v| v[0] * v[0]);
        let s2 = 1.0 - c2;
        // direct-float: periodica has no Arithma dependency
        let inv =
            c2 * c2 / (x1 * x1) + (1.0 / (s * s) - 1.0 / (x1 * x1)) * c2 * s2 + s2 * s2 / (x2 * x2);
        Ok(1.0 / inv.sqrt())
    }

    fn full_terms(&self, missing: &mut Vec<&'static str>) -> [f64; 5] {
        [
            need(self.e[2], "E3", missing),
            need(self.g[1], "G13", missing),
            need(self.g[2], "G23", missing),
            need(self.nu[1], "nu13", missing),
            need(self.nu[2], "nu23", missing),
        ]
    }

    fn local(&self, d: [f64; 3], missing: &mut Vec<&'static str>) -> Option<[f64; 3]> {
        let Some(d) = unit(d) else {
            missing.push("direction");
            return None;
        };
        let c1 = dot(d, self.axis_1);
        match self.axis_2 {
            Some(a2) => Some([c1, dot(d, a2), dot(d, cross(self.axis_1, a2))]),
            None => Some([c1, (1.0 - c1 * c1).max(0.0).sqrt(), 0.0]),
        }
    }
}

fn need(v: Option<f64>, key: &'static str, missing: &mut Vec<&'static str>) -> f64 {
    v.filter(|x| *x > 0.0 || key.starts_with("nu"))
        .unwrap_or_else(|| {
            missing.push(key);
            1.0
        })
}

fn fail_if(missing: &[&'static str]) -> Result<(), OrthotropyIncomplete> {
    if missing.is_empty() {
        Ok(())
    } else {
        Err(OrthotropyIncomplete {
            missing: missing.to_vec(),
        })
    }
}

fn num(map: &Map<String, Value>, key: &str) -> Option<f64> {
    map.get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
}

fn fiber_direction(field: &Map<String, Value>) -> Option<[f64; 3]> {
    let arr = field.get("fiber_direction")?.as_array()?;
    let v: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
    <[f64; 3]>::try_from(v).ok()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot(v, v).sqrt();
    (n.is_finite() && n > 0.0).then(|| [v[0] / n, v[1] / n, v[2] / n])
}
