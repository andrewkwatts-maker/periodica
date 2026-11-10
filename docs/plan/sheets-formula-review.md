# periodica-mat formula review (lane R1a, 2026-11-11)

Scope: `rust/periodica-mat/src/` at `claude/sheets` 9b79b4a. Read-only; no code changed. Numbers were
checked with a scratch Python script (numpy) under `H:/claude-scratch/periodica-review/check.py`, not
committed. Paths below are relative to `rust/periodica-mat/src/`. Nothing was run in Rust, so claims
about compiled behaviour (NaN, inf) follow from the formula, not from a test run.

## 1. BUGs (ranked)

| # | file:line | As implemented | Reference | Wrong vs right | Test that would catch it |
|---|---|---|---|---|---|
| 1 | derive.rs:235-243 | Normal-incidence `R = ((n-1)/(n+1))^2` from `RefractiveIndexN`, k ignored. Fires for any record with n >= 1, then feeds Absorptance and Emissivity (#2). | Born & Wolf, Principles of Optics, sec. 13.4: `R = ((n-1)^2+k^2)/((n+1)^2+k^2)`. The crate already has this (`optics/dispersion.rs:32 fresnel_expr`). | Aluminium n=1.2, k=7.26 (589 nm): code 0.0083, right 0.917. | derive test: n=1.2, ExtinctionK=7.26 gives Reflectance ~0.92; and no derivation for a metal when ExtinctionK is absent. |
| 2 | derive.rs:228-233 | `Emissivity := Absorptance` (and back), where Absorptance may be `1-R-T` from visible-band R and T. | Kirchhoff holds per wavelength and direction. Total hemispherical emissivity at T needs the spectrum integrated against Planck (Modest, Radiative Heat Transfer, sec. 3.2). | Gold, visible R ~0.9 gives eps 0.1; handbook total emissivity at 300 K is ~0.02. Window glass: visible A~0 gives eps 0, handbook ~0.9. | derive test: R=0.95, T=0 (visible) must not produce an Emissivity unless the source is tagged thermal-band. Fix idea: separate property or band tag. |
| 3 | optics/drude_lorentz.rs:32-35 | `gold()` is Drude only, `epsilon_inf=1`, no interband oscillator; `drude_of` (optics/dispersion.rs:55-66) also defaults `EpsilonInfinity` to 1.0. | Johnson & Christy, PRB 6, 4370 (1972); Etchegoin et al., J. Chem. Phys. 125, 164705 (2006) add interband Lorentz terms. | At 450 nm the model gives n=0.044, k=3.12, R=0.984; J&C gold has R~0.38 at 450 nm. R is flat ~0.984 from 450 to 700 nm, so `base_color` is silver, not gold. The existing test only asserts `r > b` and R(600)>0.9, which pass. | optics test: gold R(450 nm) < 0.6, R(600 nm) > 0.9, base_color r/b > 1.5. Needs two Lorentz terms (interband near 2.4 and 3.8 eV). |
| 4 | derive.rs:250-264 | `HardnessVickers <-> UltimateTensileStrength` with UTS = 3.3 HV for every material. | Tabor, The Hardness of Metals (1951): holds for work-hardening metals only, not ceramics, glass or polymers. | Alumina HV 1500 gives UTS 4950 MPa (brittle ceramic; flexural ~300 MPa). | derive test: ceramic or polymer record with HV gets no UTS. Gate on a metal class. |
| 5 | optics/sellmeier.rs:24-34 (+ optics/dispersion.rs:75-85) | `n^2 = 1 + sum B L^2/(L^2-C)` evaluated at any wavelength. No range, no clamp. | Schott TIE-29 (BK7 valid 0.3-2.5 um). Outside the fitted range the form has no meaning. | BK7 at 9 um: n^2 < 0, NaN; at 10.18 um (pole, sqrt(C3)): n = 38; at 12 um: 2.42. NaN reaches `base_color` and IR callers silently. | optics test: Sellmeier outside its stated range returns an error or the edge value. Store the validity range with the model. |
| 6 | mix.rs:77-82 | ThermalExpansion mixed by plain volume average. | Turner (1946): `alpha = sum(alpha_i K_i V_i)/sum(K_i V_i)`; Kerner (1956) for matrix/particle. | Al (23e-6, K=68.6 GPa) + steel (12e-6, K=167 GPa) 50/50: code 17.5e-6, Turner 15.3e-6 (14% high). | mix test with those two records expecting 15.3e-6 within 1%. |

## 2. UNCLEAR (needs an owner decision)

| # | file:line | Issue |
|---|---|---|
| U1 | mix.rs:83, 159-162 | Hill average is taken on E, G and K separately, so the mixed (E, G, K, nu) are not mutually consistent. Size is small: Al+steel 0.10%, rubber+steel 0.07% (E vs 9KG/(3K+G)). Reference Hill (1952); cheap fix: derive E from Hill K and G. |
| U2 | mix.rs:84-87 | Thermal and electrical conductivity always take the Reuss (series) bound. For a dispersed mix this is far off: Cu + 10% air (400 and 0.026 W/mK): series 0.26, parallel 360, Maxwell-Eucken ~320. Documented as a lower bound, but PlayTow gets the extreme. Needs a microstructure tag or Maxwell-Garnett/Eucken. |
| U3 | derive.rs:174-195 | Irwin `G = K^2/E` is plane stress. For thick sections (plane strain, the K_IC definition) `G = K^2 (1-nu^2)/E`, 11% lower at nu=0.33. J_IC <-> G_F is also mirrored; for concrete G_F (~140 J/m^2) is not K_IC^2/E (~30 J/m^2), so the chained `Kic = sqrt(Jic E)` can be 2x off. Stamped Estimated. |
| U4 | mix.rs:81 vs module doc line 9 | SurfaceEnergy mixed by volume average, while the module doc says atom fractions. Two stated bases for one function. |
| U5 | mix.rs:80 | StrainHardeningExponent by volume average: no reference; the Hollomon n of a composite is not an average of n. |
| U6 | mix.rs:91 | Strengths, toughness and fatigue by volume average. Documented as an estimate (rule of mixtures holds for longitudinal fibre composites only; transverse and particulate are lower). |
| U7 | property.rs:134-135, canon.rs:198 | ParisC stored as unit "1" with no scale. C has units (m/cycle)/(Pa*sqrt(m))^m, so the SI factor depends on m; a MPa*sqrt(m) datasheet value cannot be converted by a linear factor. Check what units the datasheets quote. |
| U8 | optics/xyz.rs:13-19 | CMFs are the Wyman-Sloan-Shirley multi-lobe fit (JCGT 2(2), 2013), not the CIE 1931 table. Abs error at checked points up to ~0.006 (z at 555 nm: 0.0057 vs 0.0109; x at 500 nm: 0.0023 vs 0.0049). Fine for rendering, not for colorimetry; do not call it "CIE table". |
| U9 | concrete.rs:58-60 | `Eci` (MC2010 tangent modulus, alpha_E = 1) is stored as YoungsModulus. The secant `Ec = (0.8+0.2 fcm/88) Eci` is 11% lower at fcm=38 (29.8 vs 33.6 GPa). It is also not EN 1992: `Ecm = 22 (fcm/10)^0.3` gives 32.8 GPa at C30 vs 33.6. No alpha_E for limestone (0.9), sandstone (0.7), basalt (1.2). The file states MC2010, so it is self-consistent; the brief assumed EN 1992. |
| U10 | optics/planck.rs:15-23 | `exp(hc/(l k T))` overflows to inf for small `l*T`; result relies on the evaluator mapping 1/inf to 0. Not tested. |
| U11 | optics/cauchy.rs:17-22 | Cauchy has no PropertyId (A, B, C, range), so it is only built by hand; coefficient units (B um^2, C um^4) are convention only. Valid for the transparent visible range (Born & Wolf). |
| U12 | optics/dispersion.rs:60 | `EpsilonInfinity` silently defaults to 1.0; real metals use 1 to 10. |
| U13 | derive.rs:122-170 | Isotropic identities are applied to any record, including composites and woods that also have orthotropy fields (E1 != E2); the derived G, K are meaningless there. Gate on the absence of an Orthotropy field. |

## 3. OK

| file:line | Formula as implemented | Reference | Verdict |
|---|---|---|---|
| mix.rs:155-157 | Density, Porosity: volume average | Rule of mixtures; exact for mass per volume without void change (Callister, Materials Science and Engineering, ch. 16) | OK |
| mix.rs:164-170 | Specific heat: `sum(f rho c)/sum(f rho)` | Exact: heat per volume over mass per volume | OK |
| mix.rs:172-178 | nu = (3K-2G)/(2(3K+G)) from Hill K, G | Isotropic identity, Landau & Lifshitz, Theory of Elasticity, sec. 5 | OK |
| mix.rs:144-153 | Voigt (arithmetic) and Reuss (harmonic); a zero part gives Reuss 0 | Voigt 1889, Reuss 1929, Hill 1952 mean | OK |
| derive.rs:91-115 | sigma = 1/rho | Definition | OK |
| derive.rs:133-170 | G=E/2(1+nu); K=E/3(1-2nu); E=2G(1+nu); E=3K(1-2nu); nu=E/2G-1; nu=(1-E/3K)/2; E=9KG/(3K+G); nu limits -0.999..0.4999 | Landau & Lifshitz sec. 5. Al 6061: E 68.9 GPa, nu 0.33 gives G 25.9, K 67.5 (ASM datasheet 26, 67.6) | OK |
| derive.rs:199-213 | alpha = k/(rho cp) | Incropera, Fundamentals of Heat and Mass Transfer, eq. 2.14 | OK |
| derive.rs:219-225 | A = 1-R-T, range-checked | Energy conservation | OK |
| derive.rs:250-262 (scale only) | HV[kgf/mm^2] = UTS[Pa]/1e6/3.3 | Tabor; scale right, applicability is BUG 4 | OK (scale) |
| optics/sellmeier.rs:24-34, 43 | n^2 = 1 + sum B L^2/(L^2-C), L in um, C in um^2 | Schott TIE-29. Script: n(587.6 nm)=1.51680, n(1064 nm)=1.50663 vs Schott 1.5168, 1.5066 | OK in range |
| canon.rs:257-259 + optics/dispersion.rs:51 | Sellmeier C um^2 -> m^2 (1e-12) -> back to um^2 (1e12) | Round trip is exact; the m^2 vs um^2 trap is handled | OK |
| optics/drude_lorentz.rs:36-44 | eps = einf - wp^2/(E^2+g^2) + i wp^2 g/(E(E^2+g^2)) | Drude model, Ashcroft & Mermin ch. 1 (e^-iwt convention) | OK |
| optics/drude_lorentz.rs:46-58 | Lorentz `f w0^2/(w0^2-E^2-i g E)` split into Re and Im | Fox, Optical Properties of Solids, ch. 2 | OK |
| optics/drude_lorentz.rs:72-80 | n, k from eps: `sqrt((\|e\| +/- Re e)/2)` | Fox ch. 2 | OK |
| optics/drude_lorentz.rs:11 | hc = 1239.84 eV nm | CODATA 2022: 1239.841984 (1.6e-6 relative) | OK (duplicates constants.rs) |
| optics/dispersion.rs:32-37 | Fresnel R with k | Born & Wolf sec. 13.4 | OK |
| optics/dispersion.rs:39-41 | hbar w / e | CODATA 2022, exact values | OK |
| optics/planck.rs:15-23 | B = 2hc^2 / (l^5 (exp(hc/lkT)-1)) | Planck 1900; 2019 SI exact constants | OK |
| optics/planck.rs tests | Wien peak within 0.5%, Stefan-Boltzmann within 1% | Wien b = 2.897772e-3 m K; sigma 5.670374e-8 | OK |
| optics/lines.rs:20-35 | Gaussian line profile | Doppler/instrument profile; centre and sigma come from data | OK |
| optics/xyz.rs:21-25 | XYZ to linear sRGB matrix (D65) | IEC 61966-2-1 | OK |
| orthotropy.rs:65-88 | 1/E(d) = l1^4/E1 + l2^4/E2 + l3^4/E3 + l1^2 l2^2 (1/G12 - 2nu12/E1) + l1^2 l3^2 (1/G13 - 2nu13/E1) + l2^2 l3^2 (1/G23 - 2nu23/E2). Matched against full compliance rotation at 0, 15, 45, 60, 90 deg (E1 140, E2 10 GPa, G12 5, nu12 0.3): identical | Jones, Mechanics of Composite Materials, ch. 2 off-axis modulus; Lekhnitskii | OK |
| orthotropy.rs:38-62 | G23 = E2/(2(1+nu23)) when missing | Transversely isotropic identity (Jones) | OK |
| orthotropy.rs:90-107 | Tsai-Hill: 1/s^2 = c^4/X^2 + (1/S^2-1/X^2)c^2 s^2 + s^4/Y^2 | Tsai & Hahn, Introduction to Composite Materials (1980); Jones ch. 2 | OK (tension only; see missing) |
| concrete.rs:44-51 | fctm = 0.30 fck^(2/3) (<= C50); 2.12 ln(1+fcm/10) above; continuous at C50 (4.07 vs 4.06) | fib MC2010 eq. 5.1-3a/b (EN 1992-1-1 Table 3.1 agrees) | OK |
| concrete.rs:53-55 | Gf = 73 fcm^0.18 N/m | MC2010 sec. 5.1.5.2 | OK |
| concrete.rs:58-60 | Eci = 21.5 GPa (fcm/10)^(1/3) | MC2010 eq. 5.1-21 (but see U9) | OK as MC2010 |
| concrete.rs:24-41 | eps_c1 and eps_c,lim by grade, linear interpolation | MC2010 Table 5.1-8 (values match as recalled; not re-fetched) | OK |
| constants.rs:40-101 | c, h, e, k, N_A exact; m_e 9.1093837139e-31, m_p, m_n, u 1.66053906892e-27, a0, E_h, alpha, R_inf, MeV/u, m_p/m_e | CODATA 2022 (Mohr et al., Rev. Mod. Phys. 97, 025002). Script: a0, alpha, R_inf, E_h rebuilt from h, e, m_e, c, eps0 agree to 1e-11; E_h in eV to 6e-15 | OK |
| canon.rs:104-125 | GPa 1e9; MPa 1e6; kJ/m^2 to J/m^2 1e3; g/cm^3 to kg/m^3 1e3; mm 1e-3; um 1e-6; pm 1e-12; nOhm m 1e-9 | SI prefixes | OK |
| canon.rs:111-112, 273-274 | Brewster = 1e-12 m^2/N | Definition (1 B = 1e-12 Pa^-1) | OK |
| canon.rs:124, 244 | %IACS: 5.8e5 S/m per percent | IEC 60028 (1913 standard, 58 MS/m) | OK |
| canon.rs:233, 246 | mm^2/s to m^2/s 1e-6; kV/mm to V/m 1e6 | SI | OK |
| canon.rs:122-123, 285-292 | kJ/mol to J per particle 1e3/N_A; eV to J exact; u to kg | CODATA 2022 | OK |
| canon.rs:251, 277 | percent to fraction 0.01 | SI | OK |
| property.rs:92-209 | Declared units: Pa for moduli and strengths, Pa*m^0.5 for K_IC, J/m^2 for energies, W/(m K), J/(kg K), 1/K, m^2/s, Ohm m, S/m, V/m, m^2 for Sellmeier C, rad/s, T, A/m, m^2/N for stress-optic | Dimensional check against each quantity | OK except ParisC (U7); HV/HB/HRC/HK/%/cycles are deliberate non-SI scales |

## 4. Formulas PlayTow will need that are missing

1. Refractive index mixing: `RefractiveIndexN` is not mixable (mix.rs:91-92 gives NotMixable), yet
   MATERIALS.md asks for Lorentz-Lorenz: `(n^2-1)/(n^2+2) = sum f_i (n_i^2-1)/(n_i^2+2)`. Check: n 1.5
   and 1.0 at 50/50 gives 1.232 (linear average 1.25). Complex n+ik needs Maxwell-Garnett or Bruggeman.
2. Mixing for RelativePermittivity (Lichtenecker or Maxwell-Garnett), Emissivity, Reflectance,
   Transmittance (Beer-Lambert per thickness), absorption coefficient, MaxTemperature (minimum).
3. Temperature dependence: k(T), cp(T) (Debye/Einstein or polynomial), E(T) and yield(T)
   (Johnson-Cook thermal term), `rho(T) = rho0 (1 + a (T - T0))` (`ResistivityTempCoeff` is stored with no
   reference temperature and no formula), thermo-optic dn/dT, viscosity (Arrhenius, Vogel-Fulcher),
   vapour pressure. Per playtow-sheets.md (b), none may hide a reference temperature.
4. Strain-rate and hardening evaluators: Johnson-Cook rate term, Hollomon `sigma = K eps^n` (K and n are
   stored, no formula), Ramberg-Osgood.
5. Failure: von Mises and Tresca evaluators (properties exist, no formulas), Tsai-Hill compression
   branch (compressive strengths stored, unused), Tsai-Wu, Paris-law crack growth (C, m stored),
   Basquin S-N curve.
6. Optics: Cauchy as PropertyIds, Sellmeier validity range, Drude-Lorentz interband terms for metals,
   band-integrated total emissivity, Abbe number computed from a model (stored only).
7. Fluids and acoustics: speed of sound `sqrt(K/rho)` (K and rho present, no formula), viscosity,
   surface tension.
8. Magnetics: B-H curve from Remanence and IntrinsicCoercivity (loop shape only).
