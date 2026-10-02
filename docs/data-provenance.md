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
