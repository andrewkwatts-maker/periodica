# Reference data provenance

Every curated reference dataset in periodica is a JSON file with a top-level
`_provenance` object:

| Field | Meaning |
|---|---|
| `source` | Full citation of the primary source (DOI or URL) |
| `retrieved` | Date the data was downloaded or transcribed |
| `license_or_terms` | Licence / terms of use found for the source |
| `transform` | Generator script under `tools/refdata/` or "hand transcription" |
| `units` | Units of every numeric field |
| `uncertainty` | Per-value uncertainty where the source states one |

Numbers are scientific facts and are reproduced with attribution; where an
aggregator was used to locate data, the provenance names the primary paper.
All generator scripts live in `tools/refdata/` and use only the Python
standard library, so every file can be regenerated with
`python tools/refdata/<script>.py` (downloads are cached under
`$REFDATA_CACHE`, default `%TEMP%/periodica-refdata`).

---

## Q2 — Gaussian basis sets (Basis Set Exchange)

| File | Basis | Elements | Shells |
|---|---|---|---|
| `rust/periodica-qm/data/basis/sto-3g.json` | STO-3G | H–Kr (36) | 124 |
| `rust/periodica-qm/data/basis/6-31g.json` | 6-31G | H–Kr (36) | 186 |
| `rust/periodica-qm/data/basis/6-31g_st_.json` | 6-31G* | H–Kr (36) | 220 |

* **Source:** Basis Set Exchange REST API, `https://www.basissetexchange.org/api/basis/<name>/format/json/?version=1&elements=1-36`,
  BSE data version 1 ("Data from Gaussian09" / "Gaussian 09/GAMESS", revision 2018-06-19).
  Cite B. P. Pritchard *et al.*, *J. Chem. Inf. Model.* **59**, 4814 (2019), doi:10.1021/acs.jcim.9b00725.
* **Terms:** BSD-3-Clause (MolSSI BSE repository licence). BSE asks users to cite Pritchard 2019 **and** the
  original basis-set papers, which are embedded in each file under `_provenance.basis_set_references`
  (full bibliographic entries with DOIs) and referenced per element by key:
  * STO-3G — Hehre, Stewart & Pople 1969 (doi:10.1063/1.1672392), Hehre *et al.* 1970 (doi:10.1063/1.1673374),
    Pietro *et al.* 1980 (doi:10.1021/ic50210a005), Pietro & Hehre 1983 (doi:10.1002/jcc.540040215).
  * 6-31G / 6-31G* — Ditchfield 1971 (H), Dill & Pople 1975 (Li, B), Binkley & Pople 1977 (Be), Hehre 1972 (C–F),
    Gordon 1982 + Francl 1982 (Na–Ar), Rassolov 1998 (Sc–Zn), Rassolov 2001 (K, Ca, Ga–Kr),
    Hariharan & Pople 1973 (d polarisation, C–F). **He and Ne 6-31G(*) have no literature source in BSE —
    they are cited to Gaussian 09 Rev. E.01.**
* **Transform:** `tools/refdata/fetch_bse_basis.py` — BSE "complete" JSON (`molssi_bse_schema` 0.1) is stored
  verbatim; only `_provenance` is added (plus repair of Latin-1 mojibake in one BSE author name). The download
  SHA-256 is recorded. BSE's own DOI for Rassolov 2001 reads `10.1002/jcc.1058.abs` (kept verbatim).
* **Units / conventions** (also in `_provenance.conventions`): exponents in bohr⁻² as strings (no float round
  trip); contraction coefficients multiply **normalised** primitives; Pople `sp` shells carry
  `angular_momentum: [0, 1]` with two coefficient rows; `function_type` `gto_cartesian` marks Cartesian (6D)
  d shells — the 6-31G* convention — and `gto_spherical` marks pure f shells (Sc–Zn in 6-31G*). Using spherical
  d for 6-31G* will not reproduce published energies.
* **Uncertainty:** not applicable (defined parameters).

---

## Q1-alt — Hartree–Fock regression fixtures

PySCF does not install on the development machine, so the RHF fixtures come from published, independently
computed test cases.

| Directory (`rust/periodica-qm/tests/fixtures/`) | Contents | Reference energy |
|---|---|---|
| `h2o_sto3g/` | `geometry.json`, `basis.json`, `one_electron.json` (S, T, V, dipole; 7×7), `eri.json` (228 unique), `reference.json` | E_SCF = −74.942079928192 Eh; E_MP2(corr) = −0.049149636120 |
| `ch4_sto3g/` | same layout (9×9, 912 unique ERIs) | E_SCF = −39.726850324347 Eh; E_MP2(corr) = −0.056046676165 |
| `h2_sto3g/` | `reference.json` (published values + labelled cross-check) | E_tot(R = 1.4 bohr) = −1.117 (S&O, printed); −1.1167143252 recomputed with BSE STO-3G |

**H2O / CH4 (Crawford group).**
* **Source:** T. D. Crawford group, *Programming Projects* #3 (SCF) and #4 (MP2),
  <https://github.com/CrawfordGroup/ProgrammingProjects>, commit `297fda1`; integrals computed with Psi3.
  The project credits Y. Yamaguchi (University of Georgia). Download SHA-256s are recorded per file.
* **Terms:** the repository declares **no licence** (no LICENSE file; GitHub API `license: null`, checked
  2026-10-02), so default copyright covers its prose and code. Only machine-computed numbers (integrals,
  energies — facts) are reproduced, with attribution, for regression testing. If a formally licensed fixture is
  ever needed, regenerate with PySCF (Apache-2.0) elsewhere and diff.
* **Geometry:** H2O is the Crawford **test geometry** R(OH) = 1.1 Å, ∠HOH = 104.0° — *not* the experimental
  equilibrium; CH4 is r(CH) = 1.085 Å, T_d. Coordinates in bohr, as used for the integrals.
* **Basis (exactness matters):** `basis.json` holds the exact STO-3G parameters Psi3 used — H2O from Psi3
  `lib/pbasis.dat` (psi4/psi3 commit `b74be46`, GPL-2.0; numbers only), CH4 from the explicit basis block in the
  Crawford input (differs from pbasis.dat in the 8th digit). With these parameters an independent closed-form
  evaluation (`tools/refdata/_gto_s.py`) reproduces every s-type S/T/V/ERI to ≤ 1.3e-12 Eh. With **BSE STO-3G**
  the same integrals differ by up to 4.3e-6 Eh (recorded in `one_electron.json → crosscheck`), so 1e-10 integral
  gates must load `basis.json`; a BSE-basis run needs a ~1e-5 tolerance.
* **Conventions (documented in every file's `_provenance.conventions`):** 0-based AO indices (source is 1-based);
  AO order `[X 1s, X 2s, X 2px, X 2py, X 2pz, H1 1s, …]`, verified from overlap signs against the geometry;
  contractions normalised (S_ii = 1); one-electron matrices stored full and symmetric; ERIs in chemists'
  notation (ij|kl), unique set with i ≥ j, k ≥ l, ij ≥ kl (ij = i(i+1)/2 + j), missing entries are zero;
  dipole integrals already carry the electron charge (μ = 2 Σ D_ij μ_ij + Σ Z_A R_A, checked against the
  reported dipole 0.603521296525 au); MP2 correlates all occupied MOs (no frozen core).
* **Transform:** `tools/refdata/convert_crawford_fixtures.py`.

**H2 (Szabo & Ostlund).**
* **Source:** A. Szabo & N. S. Ostlund, *Modern Quantum Chemistry* (Dover 1996 republication of the 1989 revised
  edition), ISBN 0-486-69186-1, §3.5 (minimal-basis STO-3G H2, ζ = 1.24, R = 1.4 bohr): S12 = 0.6593,
  T11 = 0.7600, T12 = 0.2365, V¹11 = −1.2266, V¹12 = −0.5974, V¹22 = −0.6538, (11|11) = 0.7746,
  (11|22) = 0.5697, (21|11) = 0.4441, (21|21) = 0.2970, ε1 = −0.578, ε2 = 0.670, E0 = −1.831, E_tot = −1.117.
  The page number (p. 167) comes from a secondary citation (Psi4 forum) and should be checked in the book; the
  integral values are corroborated by the McCullagh-lab notebook.
* **Cross-check (derived, labelled as such):** `tools/refdata/crosscheck_h2_sto3g.py` evaluates the same
  quantities in closed form (minimal-basis H2 is symmetry-determined): E_tot = −1.1167143214 (S&O 6-digit basis)
  and **−1.1167143252 Eh (BSE STO-3G)**, consistent with Psi4's non-DF −1.116714. Every published value is
  reproduced within its printed precision. **Note:** the master plan's "≈ −1.11676 Eh" is the value near
  R = 0.74 Å (1.398 bohr), not at R = 1.4 bohr.

---

## Q5 — Element rendering data

All three files live in `src/periodica/data/reference/elements/`, list every element Z = 1–118 under
`elements["<Z>"]` (with `symbol`) and provide `by_symbol` → Z. Elements without data carry `null` values and a
`note`. Generator: `tools/refdata/build_element_rendering.py` (needs PyMuPDF for the Alvarez PDF).

| File | Primary source | Coverage | Uncertainty |
|---|---|---|---|
| `vdw_radii_alvarez2013.json` | S. Alvarez, *Dalton Trans.* **42**, 8617 (2013), doi:10.1039/C3DT50599E, Table 1 | 93 elements (Z 1–60, 62–83, 89–99) | No σ in source; "differences of 0.1 Å or less should not be considered significant"; flags `larger_uncertainty` (italic: He, Ne, Si, K, Ag, Sb, Pb, Bi, Ac, Pa, Cm–Es) and `rough_estimate` (bracketed: He, Ne, Ac, Pa, Cm, Bk, Cf, Es) |
| `covalent_radii_cordero2008.json` | B. Cordero *et al.*, *Dalton Trans.* 2008, 2832, doi:10.1039/B801115J, Table 2 | 96 elements (Z 1–96) | `esd_A` = printed standard deviation (null for He, Ne, Pm, At, Rn, Fr, Ac, Pa) |
| `jmol_colors.json` | Jmol `org.jmol.c.PAL.argbsCpk` (Jmol-SwingJS commit `31d2245`) | 109 elements (Z 1–109) | n/a — a display convention |

* **Alvarez 2013** — the article is open access (CC BY-NC 3.0); the table is parsed directly from the PDF
  (University of Barcelona repository copy, hdl:2445/48823, SHA-256 recorded) by column position, reading the
  italic/bracket markers from the font data. Also stored: number of contact distances, % vdW peak, and the Bondi
  (1964, doi:10.1021/j100785a001) and Batsanov (2001, doi:10.1023/A:1011625728803) columns of the same table. The
  free-text "Observations" column is not reproduced. Cross-checks: identical to ASE's `vdw_alvarez.py`; mendeleev
  differs only for Ar, Kr, Xe, Rn because it substitutes the later noble-gas values of Vogt & Alvarez,
  *Inorg. Chem.* **53**, 9260 (2014) (correction doi:10.1021/ic502140y) — **recommended follow-up:** curate those
  four values from the primary paper if noble-gas rendering matters.
* **Cordero 2008** — the article is closed access; the numbers (facts) are transcribed from Wikipedia's
  reproduction of Table 2 ("Covalent radius", revision 1341480590, pinned) and cross-checked against two
  independent transcriptions: identical to ASE's `covalent_radii` (Z 1–96); mendeleev differs only for C, Mn, Fe, Co,
  where it stores the mean of the variants. `radius_A` is the default variant — **sp³ for C** (sp² 0.73, sp 0.69
  under `variants`) and **low spin for Mn, Fe, Co** (high spin under `variants`). The number of distances N from
  Table 2 is not stored (not in the transcription).
* **Jmol colours** — Jmol is LGPL-2.1+; colours are a **convention**, not physical data. Z = 110–118 have no Jmol
  colour (`hex: null`); `_provenance.default_hex` = `#FF1493` is Jmol's colour for unknown elements. Cross-checked
  exactly against mendeleev's `jmol_color` and within 1/255 against ASE's `jmol_colors`.
