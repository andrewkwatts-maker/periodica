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
  *Inorg. Chem.* **53**, 9260 (2014), doi:10.1021/ic501364h (correction doi:10.1021/ic502140y) — **recommended follow-up:** curate those
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

---

## Q6 — Atomic Roothaan–Hartree–Fock STO tables

| File | Source | Atoms | Validation (`tools/refdata/check_sto_tables.py`) |
|---|---|---|---|
| `rust/periodica-qm/data/sto/bunge1993.json` | Bunge, Barrientos & Bunge, *At. Data Nucl. Data Tables* **53**, 113 (1993), doi:10.1006/adnd.1993.1003 | He–Xe (Z 2–54, 53 atoms) | \|⟨R\|R⟩−1\| ≤ 3.0e-6; same-l overlap ≤ 1.8e-6; tabulated ⟨r⟩, ⟨r²⟩, ⟨1/r⟩ reproduced to 3e-6 rel. beyond print precision, ⟨1/r²⟩, ⟨1/r³⟩ to 5.4e-5; printed RHOat0 and Kato cusp reproduced to ≤ 2.2e-6 |
| `rust/periodica-qm/data/sto/koga2000.json` | Koga, Kanayama, Watanabe, Imai & Thakkar, *Theor. Chem. Acc.* **104**, 411 (2000), doi:10.1007/s002140000150 | Cs–Lr (Z 55–103, 49 atoms) | \|⟨R\|R⟩−1\| ≤ 4.5e-7; same-l overlap ≤ 2.7e-7; Σ occupations = Z; virial ratio as printed |

**Layout.** `atoms["<Z>"]` → `symbol`, `name`, `configuration`, `term`, `total_energy`, `kinetic_energy`,
`potential_energy`, `virial_ratio`, and `shells.{s,p,d,f}` = `basis` [(n, ζ)] + `orbitals` [`label`, `l`,
`occupation`, `energy`, `coefficients` (one per basis function), and for Bunge the printed `expectation`
values `r`, `r2`, `r_inv`, `r_inv2`, `r_inv3`]. `by_symbol` maps symbols to Z. Units: atomic units.
**STO convention:** χ = N r^(n−1) e^(−ζr) Y_lm with N = (2ζ)^(n+½)/√((2n)!); coefficients multiply
normalised STOs — confirmed numerically for every orbital of every atom (He, Be and Ne printed in detail by
`python tools/refdata/check_sto_tables.py He Be Ne`).

**Bunge 1993.**
* Machine-readable file `RHF.TABLES` released by C. F. Bunge for free anonymous-FTP distribution (announcement
  to the CCL list, 13 Sep 1993), mirrored at <https://server.ccl.net/cca/data/atomic-RHF-wavefunctions/tables>
  (SHA-256 recorded). **No licence terms stated**; reproduced as published scientific data with citation.
* Source quirks handled and recorded: the Xe 1s orbital energy overflows its fixed-width field and loses its
  minus sign (restored, and confirmed against Koga 1999); Mn has no printed term symbol (`term: null` +
  `term_note`); `source_RHOat0` = Σ over s orbitals of R_ns(0)² with every orbital counted once (equal to
  4πρ(0)/2 only for closed shells — verified for all 53 atoms); `source_kato_cusp` = −ρ′(0)/(Zρ(0)), exact = 2.
* **Independent cross-check:** against the Koga *et al.* 1999 cusp-constrained He–Xe wave functions
  (*Int. J. Quantum Chem.* **71**, 491; independent basis optimisation) — total energies agree to 1.1e-8 rel.,
  every orbital ⟨r⟩ to ≤ 2.2e-4 rel. (worst In 5p). This doubles as the ⟨r^k⟩ validation the plan wanted
  from Saito 2009 (see below).

**Koga 2000.**
* The paper states the functions are "available upon request from the authors or from the Web page
  <http://www.unb.ca/chem/ajit/download.htm>" (file `stf/k99heavy.zip`; the page is archived at the Wayback
  Machine (2003) but the zip is not, and the UNB page is gone). The identical-format per-atom files were
  obtained from the **AtomDB** redistribution, `theochem/AtomDB` commit `9562659`,
  `atomdb/data/slater_atom.tar.xz` (`neutral/*.slater`, Z ≥ 55; SHA-256 recorded).
* **Terms — owner decision recommended:** the authors set no licence; AtomDB's repository is **GPL-3.0**. Only
  the numerical wave-function data (facts from a published paper, authored by Koga *et al.*, not AtomDB) is used
  and no AtomDB code. If a cleaner chain is wanted, request the files from A. J. Thakkar / T. Koga directly and
  diff against `koga2000.json`.
* Configurations/terms are Koga's choices (e.g. Ce [Xe]4f¹5d¹6s² ¹G, U [Rn]5f³6d¹7s² ⁵L; Yb lists 5d(0), so no 5d
  orbital is present).
* No ⟨r^k⟩ are printed in these files, so validation is normalisation, orthogonality and electron count.

**Saito 2009 ⟨r^k⟩ — not obtained.** S. L. Saito, *At. Data Nucl. Data Tables* **95**, 836 (2009),
doi:10.1016/j.adt.2009.06.001 is paywalled (Elsevier) and no machine-readable or redistributable copy was found
(no supplementary data, not in any open repository located). **Fallback:** (1) for Z ≤ 54 use the Bunge-vs-Koga
1999 agreement above (⟨r⟩ ≤ 2.2e-4) as the independent model check, and the Bunge-printed ⟨r^k⟩ as the
transcription/implementation gate; (2) for Z ≥ 55, or to gate against the true HF limit, compute HF-limit
⟨r^k⟩ once with a fully numerical atomic HF code (e.g. HelFEM, S. Lehtola, *Int. J. Quantum Chem.* **119**,
e25945 (2019)) and commit the outputs as a fixture; or (3) hand-transcribe a subset of Saito's tables from an
institutional copy, citing the table and page. Numbers were **not** invented for any of these.

---

## Q3 (start) — Curated experimental molecular geometries

Two files per molecule, keyed by the same stem as `src/periodica/data/active/molecules/<name>.json`:

* `src/periodica/data/reference/molecules/geometry/<name>.json` — the **independent transcription**: internal
  coordinates exactly as the source states them (value, printed form where not decimal, structure type, primary
  citation, `gate` flag), 0-based atom indices in the molecule's atom order.
* `src/periodica/data/active/geometry/<name>.json` — `{name, formula, kind, atoms[{element,x,y,z}] (Å, centroid
  at origin), bonds[{i,j,order}], internal, uncertainty_A, source, citation, notes, verification}`. Cartesians are
  **constructed from the source internal coordinates** (Z-matrix / symmetry construction in
  `tools/refdata/build_geometries.py`, helpers in `_geometry.py`), never copied.

Verification: every gated source coordinate is re-measured from the Cartesians — all 16 molecules pass with
worst deviation ≤ 0.48 × tolerance (≤ 1e-4 Å, ≤ 0.01°); `python tools/refdata/check_geometries.py` repeats the
gate from the committed JSON alone (what the test suite should do). CCCBDB's own Cartesians are used only as an
independent cross-check (RMSD after superposition, stored in `verification`).

| Molecule | kind | Source values | uncertainty_A | RMSD vs CCCBDB (Å) | Primary source |
|---|---|---|---|---|---|
| H2 | r_e | r 0.74144 | 1e-5 | 0.0000 | Huber & Herzberg 1979 |
| N2 | r_e | r 1.09768(5) | 1e-5 | 0.0000 | Huber & Herzberg 1979 |
| O2 | r_e | r 1.20752 | 1e-5 | 0.0000 | Huber & Herzberg 1979 |
| CO | r_e | r 1.128323 | 1e-6 | 0.0001 | Huber & Herzberg 1979 |
| HCl | r_e | r 1.27455(2) | 1e-5 | 0.0000 | Huber & Herzberg 1979 |
| NaCl | r_e | r 2.36079(5) (gas-phase monomer) | 1e-5 | 0.0000 | Huber & Herzberg 1979 (Rice & Klemperer 1957) |
| H2O | r_e | r 0.958, ∠ 104.4776° | 5e-4 | 0.0002 | Hoy & Bunker, *J. Mol. Spectrosc.* 74, 1 (1979) |
| CH4 | r_e | r 1.087, T_d | 5e-4 | 0.0000 | Hirota, *J. Mol. Spectrosc.* 77, 213 (1979) |
| NH3 | unspecified | r 1.012, ∠HNH 106.67° | 5e-4 | 0.0003 | Herzberg 1966 |
| O3 | unspecified | r 1.278, ∠ 116.8° | 5e-4 | 0.0000 | Herzberg 1966 |
| CO2 | unspecified | r 1.162, linear | 5e-4 | 0.0001 | Herzberg 1966 |
| C6H6 | unspecified | r(CC) 1.397, r(CH) 1.084, D6h | 5e-4 | 0.0000 | Herzberg 1966 |
| C2H4 | unspecified | r(CC) 1.339, r(CH) 1.086, ∠HCH 117.6°, ∠HCC 121.2° | 5e-4 | 0.0000 | Herzberg 1966 |
| C2H6 | unspecified | r(CC) 1.536, r(CH) 1.091, ∠HCH 108.0°, staggered D3d | 5e-4 | 0.0000 | Herzberg 1966 |
| HNO3 | r_s | O–H 0.964, (H)O–N 1.406, N–O cis 1.211 / trans 1.199, ∠HON 102°9′, ∠ONO cis 115°53′ / trans 113°51′ | 5e-4 | 0.0059 | Cox & Riveros, *J. Chem. Phys.* 42, 3106 (1965) |
| H2SO4 | r_0 (inferred) | O–H 0.97(1), S–O 1.574(10), S=O 1.422(10), ∠SOH 108.5(15)°, ∠HO–S–OH 101.3(10)°, ∠O=S=O 123.3(10)°, τ −90.9(10)°, SO2 planes 88.4° | 0.01 | 0.60 (see notes) | Kuczkowski, Suenram & Lovas, *J. Am. Chem. Soc.* 103, 2561 (1981) |

**Sources and terms.**
* Diatomics: NIST Chemistry WebBook (SRD 69) "Constants of diatomic molecules" (Huber & Herzberg compilation),
  read for the X state with full precision (CCCBDB quotes only three decimals); H&H subscript (uncertain) digits are
  kept in the value and set `uncertainty_A` (one unit in the last certain digit — precision, not σ).
* Polyatomics: NIST CCCBDB (SRD 101, Release 22, doi:10.18434/T47C7Z) experimental-geometry pages, recording the
  primary reference CCCBDB cites per value. Both databases are © U.S. Secretary of Commerce under the Standard
  Reference Data Act; only individual factual values are reproduced, each attributed to its primary publication,
  with the database cited as the locator. `uncertainty_A` for CCCBDB values is the rounding of the quoted precision
  (CCCBDB gives no σ).
* HNO3 and H2SO4 were taken from the **published abstracts of the primary papers** (via Crossref/OpenAlex metadata),
  which include the parameters (and, for H2SO4, their uncertainties). This exposed two CCCBDB errors:
  HNO3 cis-ONO is listed as 115.0883° (a transposed-digit transcription of 115°53′ = 115.8833°; CCCBDB's own 130.267°
  confirms the primary value) and the H2SO4 torsion is assigned to an S=O oxygen, so CCCBDB's H2SO4 Cartesians place
  both H atoms differently (hence the 0.60 Å RMSD). The H2SO4 plane-angle handedness relative to the torsion is not
  fixed by the abstract; the alternative reading is 0.032 Å RMSD away (documented in the file).
* "kind": `r_e` where CCCBDB/H&H state equilibrium values; `r_s`/`r_0` from the primary abstracts; **`unspecified`**
  for the Herzberg (1966) values, which CCCBDB quotes without a structure type — check against the book (or replace
  with modern equilibrium structures, e.g. O3 Tanaka & Morino 1970, CO2 r_e ≈ 1.160 Å) before using them as
  r_e gates.

**Errors in `active/molecules` this fixes:** NH3 ∠HNH 106.67° (was 88° in Atoms3D), O3 coordinates now consistent
with 116.8°, HNO3 bond pattern/geometry from the microwave structure, NaCl gas-phase r_e = 2.3608 Å (was the 2.82 Å
crystal distance).

**Not curated (with recommendation):**

| Molecule | Finding | Recommendation |
|---|---|---|
| Acetic acid | CCCBDB (Landolt–Börnstein II/7, 1976) gives only the heavy-atom skeleton + one C–H; no H positions, no Cartesians. Partial transcription kept in `reference/molecules/geometry/AceticAcid.json` (`has_active_geometry: false`). | Transcribe a complete gas-phase structure from the primary ED/MW literature, or PubChem3D CID 176 (public domain) refined by own HF/6-31G (B5). |
| Ethanol | CCCBDB entry (Coussan *et al.* 1998, a matrix-isolation IR paper) lacks the H–C–H/H–C–C angles, and its Cartesians contradict its own internal list (methyl C–H 1.088/1.098 on swapped atoms). Partial transcription kept. | Primary trans-ethanol microwave structure, or PubChem3D CID 702 + HF/6-31G. |
| Glucose | No experimental gas-phase geometry (CCCBDB has only inositol for C6H12O6; glucose has several tautomers). | PubChem3D α-D-glucopyranose CID 79025 (public domain), labelled "computed conformer (MMFF94s)". |
| Aspirin | Not in CCCBDB experimental geometries. | PubChem3D CID 2244 (MMFF94s), labelled; or CSD crystal structure, labelled. |
| Caffeine | Not in CCCBDB experimental geometries. | PubChem3D CID 2519 (MMFF94s), labelled. |
