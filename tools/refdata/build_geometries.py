"""Curated experimental molecular geometries (todo Q3, start).

For every molecule of ``src/periodica/data/active/molecules`` this script either
writes a curated geometry or records why it cannot (see ``NOT_CURATED``):

* ``src/periodica/data/reference/molecules/geometry/<name>.json`` - the internal
  coordinates exactly as the source states them (CCCBDB page, NIST WebBook
  diatomic table, or the primary paper's abstract), with the primary citation.
* ``src/periodica/data/active/geometry/<name>.json`` - Cartesian coordinates
  (angstrom) built *from those internal coordinates* by the construction
  functions below (Z-matrix / symmetry), plus bonds, kind (r_e / r_0 / r_s),
  uncertainty and citation.

Every source coordinate is measured back from the Cartesians and must agree to
<= 1e-4 angstrom / 0.01 degree; the result is also compared with CCCBDB's own
Cartesians (RMSD after superposition) as an independent check.

Usage::

    python tools/refdata/build_geometries.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import _cccbdb
import _webbook
from _common import REPO_ROOT, provenance, rel, write_json
from _geometry import Vec, angle, centre, dihedral, distance, kabsch_rmsd, zmatrix_to_cartesian

ACTIVE_DIR = REPO_ROOT / "src" / "periodica" / "data" / "active" / "geometry"
REFERENCE_DIR = REPO_ROOT / "src" / "periodica" / "data" / "reference" / "molecules" / "geometry"
DIST_TOL, ANGLE_TOL = 1e-4, 0.01


# --- source coordinates ----------------------------------------------------------------------
@dataclass
class SourceCoord:
    kind: str  # "distance" | "angle" | "dihedral" | "plane_angle"
    atoms: list[int]  # 0-based, in the molecule's atom order
    value: float
    label: str
    reference: str  # short key into Spec.citations
    structure_type: str | None = None  # r_e / r_0 / r_s as stated by the source, if stated
    printed: str | None = None  # value as printed when it is not a plain decimal (e.g. 102 deg 9')
    note: str | None = None
    used_in_build: bool = True
    gate: bool = True  # re-measured from the Cartesians and required to agree (<= 1e-4 A / 0.01 deg)


@dataclass
class Spec:
    name: str
    formula: str
    elements: list[str]
    kind: str  # r_e / r_0 / r_s / unspecified
    kind_note: str
    citations: dict[str, str]  # key -> full citation with DOI
    locator: str  # where the values were read
    coords: list[SourceCoord]
    build: Callable[[Callable[[str], float]], list[Vec]]
    bonds: list[tuple[int, int, float]]
    uncertainty_A: float
    uncertainty_note: str
    cccbdb: tuple[str, str | None] | None = None  # (formula, species) for the RMSD cross-check
    notes: list[str] = field(default_factory=list)
    license: str = _cccbdb.TERMS


def _deg(d: int, m: int) -> float:
    return d + m / 60.0


# --- construction helpers -------------------------------------------------------------------
def _diatomic(r: float) -> list[Vec]:
    return [(0.0, 0.0, 0.0), (0.0, 0.0, r)]


def _c3v_top(centre_at: Vec, r: float, interbond: float, azimuths: list[float], down: float) -> list[Vec]:
    """Three equivalent bonds of length r from ``centre_at`` with mutual angle ``interbond``.

    The C3 axis is z; ``down`` = +1 puts the substituents on the +z side, -1 on the -z side.
    """
    sin2b = (1.0 - math.cos(math.radians(interbond))) / 1.5
    sb, cb = math.sqrt(sin2b), math.sqrt(1.0 - sin2b)
    return [
        (centre_at[0] + r * sb * math.cos(math.radians(a)), centre_at[1] + r * sb * math.sin(math.radians(a)), centre_at[2] + down * r * cb)
        for a in azimuths
    ]


def _build_water(v: Callable[[str], float]) -> list[Vec]:
    return zmatrix_to_cartesian([(None, None, None, None, None, None), (0, v("rOH"), None, None, None, None), (0, v("rOH"), 1, v("aHOH"), None, None)])


def _build_ozone(v: Callable[[str], float]) -> list[Vec]:
    return zmatrix_to_cartesian([(None, None, None, None, None, None), (0, v("rOO"), None, None, None, None), (0, v("rOO"), 1, v("aOOO"), None, None)])


def _build_co2(v: Callable[[str], float]) -> list[Vec]:
    r = v("rCO")
    return [(0.0, 0.0, 0.0), (0.0, 0.0, r), (0.0, 0.0, -r)]


def _build_ammonia(v: Callable[[str], float]) -> list[Vec]:
    return [(0.0, 0.0, 0.0), *_c3v_top((0.0, 0.0, 0.0), v("rNH"), v("aHNH"), [270.0, 30.0, 150.0], -1.0)]


def _build_methane(v: Callable[[str], float]) -> list[Vec]:
    s = v("rCH") / math.sqrt(3.0)  # T_d: the source angle 109.471 is the tetrahedral angle
    return [(0.0, 0.0, 0.0), (s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]


def _build_benzene(v: Callable[[str], float]) -> list[Vec]:
    rcc, rch = v("rCC"), v("rCH")  # D6h: ring radius = C-C bond length
    ring = [(rcc * math.cos(math.radians(90 - 60 * k)), rcc * math.sin(math.radians(90 - 60 * k)), 0.0) for k in range(6)]
    hyd = [((rcc + rch) * math.cos(math.radians(90 - 60 * k)), (rcc + rch) * math.sin(math.radians(90 - 60 * k)), 0.0) for k in range(6)]
    return ring + hyd


def _build_ethylene(v: Callable[[str], float]) -> list[Vec]:
    rcc, rch, hcc = v("rCC"), v("rCH"), math.radians(v("aHCC"))
    c1, c2 = (0.0, 0.0, rcc / 2), (0.0, 0.0, -rcc / 2)
    dy, dz = rch * math.sin(math.pi - hcc), rch * math.cos(math.pi - hcc)
    return [c1, c2, (0.0, dy, c1[2] + dz), (0.0, -dy, c1[2] + dz), (0.0, dy, c2[2] - dz), (0.0, -dy, c2[2] - dz)]


def _build_ethane(v: Callable[[str], float]) -> list[Vec]:
    rcc, rch, hch = v("rCC"), v("rCH"), v("aHCH")  # staggered D3d
    c1, c2 = (0.0, 0.0, rcc / 2), (0.0, 0.0, -rcc / 2)
    return [c1, c2, *_c3v_top(c1, rch, hch, [180.0, 60.0, 300.0], 1.0), *_c3v_top(c2, rch, hch, [0.0, 240.0, 120.0], -1.0)]


def _build_nitric_acid(v: Callable[[str], float]) -> list[Vec]:
    # atoms: N1, O2 (hydroxyl O), O3 (cis to H), O4 (trans), H5 ; planar C_s
    return zmatrix_to_cartesian(
        [
            (None, None, None, None, None, None),
            (0, v("rNO(H)"), None, None, None, None),
            (0, v("rNO_cis"), 1, v("aONO_cis"), None, None),
            (0, v("rNO_trans"), 1, v("aONO_trans"), 2, 180.0),
            (1, v("rOH"), 0, v("aHON"), 2, 0.0),
        ]
    )


def _build_sulfuric_acid(v: Callable[[str], float]) -> list[Vec]:
    # atoms: S1, O2, O3 (S=O), O4, O5 (S-OH), H6 (on O5), H7 (on O4); C2 axis = z
    r_so2, r_so1, r_oh = v("rS=O"), v("rS-O"), v("rOH")
    half2, half1 = math.radians(v("aO=S=O") / 2), math.radians(v("aHO-S-OH") / 2)
    phi = math.radians(-v("plane_angle"))  # O4 azimuth relative to O2 (see notes)
    s = (0.0, 0.0, 0.0)
    o2 = (r_so2 * math.sin(half2), 0.0, r_so2 * math.cos(half2))
    o3 = (-o2[0], -o2[1], o2[2])
    o4 = (r_so1 * math.sin(half1) * math.cos(phi), r_so1 * math.sin(half1) * math.sin(phi), -r_so1 * math.cos(half1))
    o5 = (-o4[0], -o4[1], o4[2])
    from _geometry import place

    h7 = place(o5, s, o4, r_oh, v("aSOH"), v("tO'SOH"))
    h6 = (-h7[0], -h7[1], h7[2])
    return [s, o2, o3, o4, o5, h6, h7]


# --- molecule specifications ------------------------------------------------------------------
HERZBERG_1966 = (
    "G. Herzberg, Molecular Spectra and Molecular Structure III. Electronic Spectra and Electronic Structure of "
    "Polyatomic Molecules (Van Nostrand, New York, 1966)"
)
HERZBERG_KIND = (
    "CCCBDB quotes these values from Herzberg (1966) without stating the structure type (r_e or r_0); the "
    "compilation mixes both. Treat as 'unspecified' until checked against the book."
)
CCCBDB_UNC = (
    "Experimental uncertainty not given by the source as quoted; uncertainty_A is the rounding of the "
    "quoted precision (half a unit in the last decimal of the bond lengths), not a standard deviation."
)


def _cccbdb_coords(formula: str, species: str | None, labels: dict[tuple[str, tuple[int, ...]], str]) -> tuple[list[SourceCoord], dict[str, str], str | None]:
    """Source coordinates from a CCCBDB page, keyed by our labels (CCCBDB 1-based atoms -> 0-based)."""
    page = _cccbdb.parse_page(_cccbdb.fetch_page(formula, species))
    out = []
    for c in page.internal:
        key = (c.description, tuple(c.atoms))
        if key not in labels:
            continue
        kind = {"r": "distance", "a": "angle", "d": "dihedral"}[c.description[0]]
        stype = {"re": "r_e", "equilibrium": "r_e"}.get(c.comment.split(" ")[0]) if c.comment else None
        out.append(SourceCoord(kind, [a - 1 for a in c.atoms], c.value, labels[key], c.reference, stype, note=c.comment or None))
    missing = set(labels.values()) - {c.label for c in out}
    if missing:
        raise SystemExit(f"CCCBDB {formula}: coordinates {sorted(missing)} not found on the page")
    refs = {k: (r["reference"] + (f", doi:{r['doi'].strip()}" if r["doi"].strip() else "")).replace("Â", "") for k, r in page.references.items()}
    return out, refs, page.point_group


def specs() -> list[Spec]:
    out: list[Spec] = []

    # Diatomics: NIST WebBook (Huber & Herzberg 1979), r_e with 5-6 decimals.
    diatomics = [
        ("Hydrogen", "H2", ["H", "H"], "1333740", 1.0),
        ("Nitrogen", "N2", ["N", "N"], "7727379", 3.0),
        ("Oxygen", "O2", ["O", "O"], "7782447", 2.0),
        ("CarbonMonoxide", "CO", ["C", "O"], "630080", 3.0),
        ("HydrogenChloride", "HCl", ["Cl", "H"], "7647010", 1.0),
        ("SodiumChloride", "NaCl", ["Na", "Cl"], "7647145", 1.0),
    ]
    for name, formula, elements, cas, order in diatomics:
        d = _webbook.ground_state_re(cas)
        label = f"r{elements[0]}{elements[1]}"
        out.append(
            Spec(
                name=name,
                formula=formula,
                elements=elements,
                kind="r_e",
                kind_note="Equilibrium internuclear distance of the X ground state (Huber & Herzberg).",
                citations={"HH1979": _webbook.HUBER_HERZBERG + " (measurements: " + "; ".join(d.sources) + ")"},
                locator=f"{_webbook.CITATION}; {d.url}",
                coords=[
                    SourceCoord(
                        "distance", [0, 1], d.r_e, label, "HH1979", "r_e", printed=d.text,
                        note="last digit printed as a subscript (uncertain)" if d.uncertain_last_digit else None,
                    )
                ],
                build=lambda v, lab=label: _diatomic(v(lab)),
                bonds=[(0, 1, order)],
                uncertainty_A=10.0 ** (-(d.decimals - 1 if d.uncertain_last_digit else d.decimals)),
                uncertainty_note="Huber & Herzberg print uncertain last digits as subscripts (kept in the value); "
                "uncertainty_A = one unit in the last digit they regard as certain - an indication of the quoted "
                "precision, not a standard deviation.",
                cccbdb=(formula, None),
                notes=[f"Ground state {d.state}."]
                + (["Gas-phase NaCl monomer r_e; the crystal Na-Cl distance (2.82 A) is a different quantity."] if name == "SodiumChloride" else []),
                license=_webbook.TERMS,
            )
        )

    # Polyatomics from CCCBDB.
    def poly(name: str, formula: str, species: str | None, elements: list[str], labels: dict[tuple[str, tuple[int, ...]], str], kind: str, kind_note: str, build: Callable[[Callable[[str], float]], list[Vec]], bonds: list[tuple[int, int, float]], unc: float, notes: list[str]) -> Spec:
        coords, refs, pg = _cccbdb_coords(formula, species, labels)
        used = {c.reference for c in coords}
        return Spec(
            name=name,
            formula=formula,
            elements=elements,
            kind=kind,
            kind_note=kind_note,
            citations={k: v for k, v in refs.items() if k in used},
            locator=f"{_cccbdb.CITATION} (experimental geometry page for {formula}{', ' + species if species else ''}; point group {pg})",
            coords=coords,
            build=build,
            bonds=bonds,
            uncertainty_A=unc,
            uncertainty_note=CCCBDB_UNC,
            cccbdb=(formula, species),
            notes=notes,
        )

    out.append(
        poly("Water", "H2O", None, ["O", "H", "H"], {("rOH", (1, 2)): "rOH", ("aHOH", (2, 1, 3)): "aHOH"}, "r_e",
             "CCCBDB marks both values as equilibrium (r_e) from Hoy & Bunker (1979).", _build_water, [(0, 1, 1), (0, 2, 1)], 5e-4,
             ["CCCBDB quotes r_e(OH) to three decimals (0.958); its own Cartesians use 0.9578."])
    )
    out.append(
        poly("Ozone", "O3", None, ["O", "O", "O"], {("rOO", (1, 2)): "rOO", ("aOOO", (2, 1, 3)): "aOOO"}, "unspecified", HERZBERG_KIND,
             _build_ozone, [(0, 1, 1.5), (0, 2, 1.5)], 5e-4,
             ["Herzberg (1966) values. A later equilibrium structure exists (Tanaka & Morino 1970: r_e = 1.2717 A, "
              "117.47 deg) but was not transcribed here because the primary text was not accessed."])
    )
    out.append(
        poly("CarbonDioxide", "CO2", None, ["C", "O", "O"], {("rCO", (1, 2)): "rCO", ("aOCO", (2, 1, 3)): "aOCO"}, "unspecified", HERZBERG_KIND,
             _build_co2, [(0, 1, 2), (0, 2, 2)], 5e-4, ["Herzberg (1966) value; the equilibrium r_e(CO) is about 1.160 A (not transcribed)."])
    )
    out.append(
        poly("Ammonia", "NH3", None, ["N", "H", "H", "H"], {("rNH", (1, 2)): "rNH", ("aHNH", (2, 1, 3)): "aHNH"}, "unspecified", HERZBERG_KIND,
             _build_ammonia, [(0, 1, 1), (0, 2, 1), (0, 3, 1)], 5e-4,
             ["C3v pyramid built from r(NH) and the H-N-H angle; CCCBDB's 'aXNH 112.15' is derived from symmetry and not a source value."])
    )
    out.append(
        poly("Methane", "CH4", None, ["C", "H", "H", "H", "H"], {("rCH", (1, 2)): "rCH", ("aHCH", (2, 1, 3)): "aHCH"}, "r_e",
             "CCCBDB marks r(CH) as r_e (Hirota 1979); the angle is the tetrahedral angle (T_d).", _build_methane,
             [(0, k, 1) for k in range(1, 5)], 5e-4, ["Exact T_d symmetry imposed (109.4712206 deg); the source quotes 109.471."])
    )
    out.append(
        poly("Benzene", "C6H6", "Benzene", ["C"] * 6 + ["H"] * 6,
             {("rCC", (1, 2)): "rCC", ("rCH", (1, 7)): "rCH", ("aCCC", (1, 2, 3)): "aCCC", ("aHCC", (1, 2, 8)): "aHCC"}, "unspecified",
             HERZBERG_KIND, _build_benzene, [(k, (k + 1) % 6, 1.5) for k in range(6)] + [(k, k + 6, 1) for k in range(6)], 5e-4, ["D6h."])
    )
    out.append(
        poly("Ethylene", "C2H4", None, ["C", "C", "H", "H", "H", "H"],
             {("rCC", (1, 2)): "rCC", ("rCH", (1, 3)): "rCH", ("aHCH", (3, 1, 4)): "aHCH", ("aHCC", (1, 2, 5)): "aHCC"}, "unspecified",
             HERZBERG_KIND, _build_ethylene, [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)], 5e-4,
             ["D2h planar; H-C-H = 360 - 2 x H-C-C holds exactly for the quoted values."])
    )
    out.append(
        poly("Ethane", "C2H6", None, ["C", "C", "H", "H", "H", "H", "H", "H"],
             {("rCC", (1, 2)): "rCC", ("rCH", (1, 3)): "rCH", ("aHCH", (3, 1, 4)): "aHCH", ("aHCC", (1, 2, 6)): "aHCC"}, "unspecified",
             HERZBERG_KIND, _build_ethane, [(0, 1, 1)] + [(0, k, 1) for k in (2, 3, 4)] + [(1, k, 1) for k in (5, 6, 7)], 5e-4,
             ["Staggered D3d built from r(CC), r(CH) and H-C-H; H-C-C (110.91, 'by symmetry' in CCCBDB) is reproduced.",
              "CCCBDB labels the point group D3h although its Cartesians are staggered (D3d)."])
    )
    for s in out:
        if s.name == "Ethane":
            for c in s.coords:
                if c.label == "aHCC":
                    c.used_in_build = False
                    c.note = "derived by symmetry in the source; checked, not used to build"
        if s.name == "Methane":
            for c in s.coords:
                if c.label == "aHCH":
                    c.used_in_build = False
                    c.note = "tetrahedral angle; T_d imposed exactly"

    # Nitric acid: primary values from the Cox & Riveros (1965) abstract (CCCBDB has a transposed digit).
    cox = "A. P. Cox and J. M. Riveros, 'Microwave Spectrum and Structure of Nitric Acid', J. Chem. Phys. 42, 3106 (1965), doi:10.1063/1.1696387"
    hno3 = [
        SourceCoord("distance", [1, 4], 0.964, "rOH", "Cox1965", "r_s"),
        SourceCoord("distance", [0, 1], 1.406, "rNO(H)", "Cox1965", "r_s"),
        SourceCoord("distance", [0, 2], 1.211, "rNO_cis", "Cox1965", "r_s"),
        SourceCoord("distance", [0, 3], 1.199, "rNO_trans", "Cox1965", "r_s"),
        SourceCoord("angle", [0, 1, 4], _deg(102, 9), "aHON", "Cox1965", "r_s", printed="102°9′"),
        SourceCoord("angle", [1, 0, 2], _deg(115, 53), "aONO_cis", "Cox1965", "r_s", printed="115°53′"),
        SourceCoord("angle", [1, 0, 3], _deg(113, 51), "aONO_trans", "Cox1965", "r_s", printed="113°51′"),
        SourceCoord("distance", [2, 3], 2.184, "rO..O(NO2)", "Cox1965", "r_s", used_in_build=False, gate=False,
                    note="Stated in the abstract; the other stated parameters imply 2.1865 A (0.0025 A apart), so it is recorded but not gated."),
    ]
    out.append(
        Spec(
            name="NitricAcid", formula="HNO3", elements=["N", "O", "O", "O", "H"], kind="r_s",
            kind_note="Substitution structure (Kraitchman / double-substitution from ground-state moments of six isotopologues), per the abstract.",
            citations={"Cox1965": cox}, locator="Values transcribed from the published abstract of Cox & Riveros (1965) (via Crossref/OpenAlex metadata); "
            "cross-checked against " + _cccbdb.CITATION,
            coords=hno3, build=_build_nitric_acid, bonds=[(0, 1, 1), (0, 2, 1.5), (0, 3, 1.5), (1, 4, 1)], uncertainty_A=5e-4,
            uncertainty_note="No standard deviations in the abstract; uncertainty_A is the rounding of the quoted precision (0.001 A); angles are quoted to 1 arc-minute.",
            cccbdb=("HNO3", None),
            notes=["Planar (C_s): O3 is cis to the hydroxyl H, O4 trans.",
                   "CCCBDB lists the cis O-N-O angle as 115.0883 deg: a transposed-digit transcription of 115 deg 53' = 115.8833 deg "
                   "(its 130.267 deg for O3-N-O4 equals 360 - 115.8833 - 113.85, confirming the primary value)."],
            license="Numerical values from the abstract of a published paper (facts), cited; CCCBDB used only as a cross-check.",
        )
    )

    kuc = ("R. L. Kuczkowski, R. D. Suenram and F. J. Lovas, 'Microwave spectrum, structure, and dipole moment of sulfuric acid', "
           "J. Am. Chem. Soc. 103, 2561-2566 (1981), doi:10.1021/ja00400a013")
    h2so4 = [
        SourceCoord("distance", [4, 5], 0.97, "rOH", "Kuc1981", note="0.97(1)"),
        SourceCoord("distance", [0, 3], 1.574, "rS-O", "Kuc1981", note="1.574(10); S-O(H)"),
        SourceCoord("distance", [0, 1], 1.422, "rS=O", "Kuc1981", note="1.422(10)"),
        SourceCoord("angle", [0, 4, 5], 108.5, "aSOH", "Kuc1981", note="108.5(15)"),
        SourceCoord("angle", [3, 0, 4], 101.3, "aHO-S-OH", "Kuc1981", note="101.3(10)"),
        SourceCoord("angle", [1, 0, 2], 123.3, "aO=S=O", "Kuc1981", note="123.3(10)"),
        SourceCoord("dihedral", [4, 0, 3, 6], -90.9, "tO'SOH", "Kuc1981", note="-90.9(10); O1'-S-O1-H with O1, O1' the hydroxyl oxygens"),
        SourceCoord("plane_angle", [1, 0, 2, 3, 0, 4], 88.4, "plane_angle", "Kuc1981", note="88.4(1) as printed; angle between the two SO2 planes"),
    ]
    out.append(
        Spec(
            name="SulfuricAcid", formula="H2SO4", elements=["S", "O", "O", "O", "O", "H", "H"], kind="r_0",
            kind_note="Effective ground-state structure fitted to rotational constants of four isotopologues (normal, 34S, D1, D2); "
            "the abstract does not name the structure type, r_0 is inferred.",
            citations={"Kuc1981": kuc},
            locator="Values (with their stated uncertainties) transcribed from the published abstract of Kuczkowski et al. (1981); "
            "cross-checked against " + _cccbdb.CITATION,
            coords=h2so4, build=_build_sulfuric_acid,
            bonds=[(0, 1, 2), (0, 2, 2), (0, 3, 1), (0, 4, 1), (4, 5, 1), (3, 6, 1)], uncertainty_A=0.01,
            uncertainty_note="Stated by the source: 0.01 A for r(OH), r(S-O), r(S=O); angles 1.0-1.5 deg.",
            cccbdb=("H2SO4", None),
            notes=["C2 symmetry (axis z). Atoms: S1, O2/O3 (S=O), O4/O5 (S-OH), H6 on O5, H7 on O4.",
                   "Handedness: the abstract fixes |plane angle| = 88.4 deg and the torsion O1'-S-O1-H = -90.9 deg but not on which side "
                   "of the S=O plane the hydroxyl plane is rotated; the hydroxyl O4 is placed 88.4 deg from O2 (as in CCCBDB's frame). "
                   "The alternative reading (91.6 deg) gives a structure 0.032 A RMSD away (H atoms only), well inside the stated "
                   "angle uncertainties.",
                   "CCCBDB's internal list assigns the -90.9 deg torsion to O3-S1-O4-H7 (an S=O oxygen) and its Cartesians follow that, "
                   "so CCCBDB's H positions differ from this primary-based structure (larger RMSD expected)."],
            license="Numerical values from the abstract of a published paper (facts), cited; CCCBDB used only as a cross-check.",
        )
    )
    return out


# Molecules of active/molecules that are not curated here, with the reason and recommendation.
NOT_CURATED = {
    "AceticAcid": "CCCBDB (Landolt-Bornstein II/7, 1976) gives only the heavy-atom skeleton and one C-H distance, no hydrogen "
    "positions and no Cartesians. Recommend: transcribe a complete gas-phase structure from the primary electron-diffraction / "
    "microwave literature, or use the PubChem3D conformer (CID 176, public domain) refined by own HF/6-31G (B5).",
    "Ethanol": "CCCBDB's entry (citing Coussan et al. 1998, a matrix-isolation IR paper) is incomplete (no H-C-H / H-C-C angles) and "
    "its Cartesians contradict its own internal list (methyl C-H 1.088/1.098 assigned to swapped atoms). Recommend: the trans-ethanol "
    "microwave structure from the primary literature, or PubChem3D (CID 702) + HF/6-31G refinement.",
    "Glucose": "No experimental gas-phase geometry (CCCBDB has none for C6H12O6 except inositol; glucose exists as several ring/"
    "open-chain tautomers). Recommend PubChem3D conformer of alpha-D-glucopyranose (CID 79025, public domain), labelled as a "
    "computed conformer.",
    "aspirin": "Not in CCCBDB experimental geometries. Recommend PubChem3D conformer (CID 2244, public domain; MMFF94s) or the CSD "
    "crystal structure, labelled as such.",
    "caffeine": "Not in CCCBDB experimental geometries. Recommend PubChem3D conformer (CID 2519, public domain; MMFF94s), labelled.",
}


# --- build & verify ---------------------------------------------------------------------------
def _measure(xyz: list[Vec], c: SourceCoord) -> float:
    p = [xyz[a] for a in c.atoms]
    if c.kind == "distance":
        return distance(*p)
    if c.kind == "angle":
        return angle(*p)
    if c.kind == "dihedral":
        return dihedral(*p)
    # plane angle between planes (a,b,c) and (d,e,f), in [0, 90]
    from _geometry import _cross, _dot, _sub, _unit

    n1 = _unit(_cross(_sub(p[0], p[1]), _sub(p[2], p[1])))
    n2 = _unit(_cross(_sub(p[3], p[4]), _sub(p[5], p[4])))
    a = math.degrees(math.acos(min(1.0, abs(_dot(n1, n2)))))
    return a


def build_one(spec: Spec) -> tuple[dict[str, Any], dict[str, Any]]:
    values = {c.label: c.value for c in spec.coords}
    xyz = centre(spec.build(lambda key: values[key]))
    xyz = [(round(x, 6) + 0.0, round(y, 6) + 0.0, round(z, 6) + 0.0) for x, y, z in xyz]  # + 0.0 drops -0.0
    checks, worst_d, worst_a = [], 0.0, 0.0
    for c in spec.coords:
        measured = _measure(xyz, c)
        dev = abs(measured - c.value)
        if c.kind == "dihedral":
            dev = min(dev, 360.0 - dev)
        gated = c.gate
        tol = DIST_TOL if c.kind == "distance" else ANGLE_TOL
        if gated and dev > tol:
            raise SystemExit(f"{spec.name}: {c.label} measured {measured:.6f} vs source {c.value} (dev {dev:.2e})")
        if gated:
            if c.kind == "distance":
                worst_d = max(worst_d, dev)
            else:
                worst_a = max(worst_a, dev)
        checks.append({"label": c.label, "source": c.value, "measured": round(measured, 6), "gated": gated})
    rmsd = None
    if spec.cccbdb:
        page = _cccbdb.parse_page(_cccbdb.fetch_page(*spec.cccbdb))
        if len(page.atoms) == len(xyz) and [a["element"] for a in page.atoms] == spec.elements:
            rmsd = round(kabsch_rmsd(xyz, [(a["x"], a["y"], a["z"]) for a in page.atoms]), 4)
    citation = "; ".join(spec.citations.values())
    prov = provenance(
        source=[citation, spec.locator],
        license_or_terms=spec.license,
        transform="tools/refdata/build_geometries.py",
        units={"coordinates": "angstrom", "angles": "degree"},
        uncertainty=spec.uncertainty_note,
    )
    reference = {
        "_provenance": {
            **prov,
            "transform": "tools/refdata/build_geometries.py (values as stated by the source; no derivation)",
            "units": {"distance": "angstrom", "angle": "degree", "dihedral": "degree", "plane_angle": "degree"},
        },
        "name": spec.name,
        "formula": spec.formula,
        "atom_order": [f"{e}{i + 1}" for i, e in enumerate(spec.elements)],
        "atom_index_base": 0,
        "coordinates": [
            {
                "type": c.kind,
                "atoms": c.atoms,
                "value": c.value,
                "unit": "angstrom" if c.kind == "distance" else "degree",
                "label": c.label,
                **({"printed": c.printed} if c.printed else {}),
                **({"structure_type": c.structure_type} if c.structure_type else {}),
                "reference": spec.citations.get(c.reference, c.reference),
                **({"note": c.note} if c.note else {}),
                "gate": c.gate,
            }
            for c in spec.coords
        ],
        "has_active_geometry": True,
    }
    bonds_lengths = [{"atoms": [i, j], "value_A": round(distance(xyz[i], xyz[j]), 6)} for i, j, _ in spec.bonds]
    active = {
        "_provenance": {
            **prov,
            "transform": "tools/refdata/build_geometries.py: Cartesians constructed from the source internal coordinates "
            "(Z-matrix / symmetry construction), centroid at origin; every source value re-measured (<= 1e-4 A, <= 0.01 deg).",
        },
        "name": spec.name,
        "formula": spec.formula,
        "kind": spec.kind,
        "kind_note": spec.kind_note,
        "atoms": [{"element": e, "x": p[0], "y": p[1], "z": p[2]} for e, p in zip(spec.elements, xyz, strict=True)],
        "bonds": [{"i": i, "j": j, "order": o} for i, j, o in spec.bonds],
        "internal": {
            "source_values": [{"label": c.label, "type": c.kind, "atoms": c.atoms, "value": c.value} for c in spec.coords if c.used_in_build],
            "bond_lengths_A": bonds_lengths,
        },
        "uncertainty_A": spec.uncertainty_A,
        "uncertainty_note": spec.uncertainty_note,
        "source": spec.locator,
        "citation": citation,
        "notes": spec.notes,
        "verification": {
            "max_abs_dev_distance_A": float(f"{worst_d:.2e}"),
            "max_abs_dev_angle_deg": float(f"{worst_a:.2e}"),
            "checks": checks,
            "rmsd_vs_cccbdb_cartesians_A": rmsd,
        },
    }
    return reference, active


# Partial CCCBDB entries kept as reference transcriptions only (no active geometry).
PARTIAL_CCCBDB = {"AceticAcid": ("C2H4O2", "Acetic acid", "CH3COOH"), "Ethanol": ("C2H6O", "Ethanol", "C2H6O")}


def partial_reference(name: str) -> dict[str, Any]:
    formula, species, display = PARTIAL_CCCBDB[name]
    page = _cccbdb.parse_page(_cccbdb.fetch_page(formula, species))
    refs = {k: (r["reference"] + (f", doi:{r['doi'].strip()}" if r["doi"].strip() else "")).replace("Â", "") for k, r in page.references.items()}
    return {
        "_provenance": provenance(
            source=[f"{_cccbdb.CITATION} (experimental geometry page for {formula}, {species})"] + sorted({refs.get(c.reference, c.reference) for c in page.internal}),
            license_or_terms=_cccbdb.TERMS,
            transform="tools/refdata/build_geometries.py (CCCBDB internal-coordinate table parsed verbatim)",
            units={"distance": "angstrom", "angle": "degree"},
            uncertainty="Not given by the source.",
        ),
        "name": name,
        "formula": display,
        "atom_index_base": 0,
        "atom_numbering": "CCCBDB numbering minus one",
        "coordinates": [
            {
                "type": {"r": "distance", "a": "angle", "d": "dihedral"}[c.description[0]],
                "atoms": [a - 1 for a in c.atoms],
                "value": c.value,
                "unit": "angstrom" if c.description[0] == "r" else "degree",
                "label": c.description,
                "reference": refs.get(c.reference, c.reference),
                **({"note": c.comment} if c.comment else {}),
                "gate": False,
            }
            for c in page.internal
        ],
        "has_active_geometry": False,
        "why_not_curated": NOT_CURATED[name],
    }


def main() -> int:
    for name in PARTIAL_CCCBDB:
        write_json(REFERENCE_DIR / f"{name}.json", partial_reference(name))
    for spec in specs():
        reference, active = build_one(spec)
        write_json(REFERENCE_DIR / f"{spec.name}.json", reference)
        write_json(ACTIVE_DIR / f"{spec.name}.json", active)
        v = active["verification"]
        print(f"{spec.name:18s} {spec.kind:12s} max|dr|={v['max_abs_dev_distance_A']:.1e} A  max|da|={v['max_abs_dev_angle_deg']:.1e} deg  "
              f"RMSD vs CCCBDB = {v['rmsd_vs_cccbdb_cartesians_A']}")
    print(f"not curated: {', '.join(NOT_CURATED)}")
    print(f"wrote {rel(ACTIVE_DIR)} and {rel(REFERENCE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
