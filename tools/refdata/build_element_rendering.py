"""Element rendering data: vdW radii, covalent radii and Jmol colours (todo Q5).

Writes, under ``src/periodica/data/reference/elements/``:

* ``vdw_radii_alvarez2013.json`` - parsed directly from Table 1 of the
  open-access article (S. Alvarez, Dalton Trans. 2013), including the
  italic ("larger uncertainty") and bracket ("rough estimate") markers, which
  are read from the PDF font information.  Requires PyMuPDF (``fitz``).
* ``covalent_radii_cordero2008.json`` - Cordero et al. 2008 Table 2 values
  with standard deviations, transcribed from the Wikipedia reproduction of
  that table (pinned revision) because the article is not open access.
* ``jmol_colors.json`` - the Jmol CPK palette (``PAL.argbsCpk``) from the Jmol
  source code (pinned commit).

Every table is cross-checked against two independent transcriptions (ASE and
mendeleev); the only tolerated differences are the documented ones.

Usage::

    python tools/refdata/build_element_rendering.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from _common import REPO_ROOT, SYMBOLS, cache_dir, element_table, fetch, provenance, rel, sha256, write_json

OUT_DIR = REPO_ROOT / "src" / "periodica" / "data" / "reference" / "elements"

# --- sources (pinned) -------------------------------------------------------------------------
ALVAREZ_PDF = "https://diposit.ub.edu/server/api/core/bitstreams/35dddff6-d683-4e8a-a7f3-4504a732d023/content"
ALVAREZ_HANDLE = "https://hdl.handle.net/2445/48823"
WIKI_REVISION = 1341480590
WIKI_RAW = f"https://en.wikipedia.org/w/index.php?title=Covalent_radius&oldid={WIKI_REVISION}&action=raw"
JMOL_COMMIT = "31d22450c621fbfe2bc3c7e447b8c8a0663d3485"
JMOL_PAL = f"https://raw.githubusercontent.com/BobHanson/Jmol-SwingJS/{JMOL_COMMIT}/src/org/jmol/c/PAL.java"
ASE_COMMIT = "0a7257774301fed11a5b4e5d9394c28c1401f275"
ASE_VDW = f"https://gitlab.com/ase/ase/-/raw/{ASE_COMMIT}/ase/data/vdw_alvarez.py"
ASE_DATA = f"https://gitlab.com/ase/ase/-/raw/{ASE_COMMIT}/ase/data/__init__.py"
ASE_COLORS = f"https://gitlab.com/ase/ase/-/raw/{ASE_COMMIT}/ase/data/colors.py"
MENDELEEV_COMMIT = "4e02078d08988fad881bb7d99f2d50bf94e22823"
MENDELEEV_DB = f"https://github.com/lmmentel/mendeleev/raw/{MENDELEEV_COMMIT}/mendeleev/elements.db"

ALVAREZ_CITATION = (
    "S. Alvarez, 'A cartography of the van der Waals territories', Dalton Trans. 42, 8617-8636 (2013), "
    "doi:10.1039/C3DT50599E, Table 1 (open-access copy: Diposit Digital UB, " + ALVAREZ_HANDLE + ")."
)
CORDERO_CITATION = (
    "B. Cordero, V. Gomez, A. E. Platero-Prats, M. Reves, J. Echeverria, E. Cremades, F. Barragan and "
    "S. Alvarez, 'Covalent radii revisited', Dalton Trans. 2008, 2832-2838, doi:10.1039/B801115J, Table 2."
)
JMOL_CITATION = (
    "Jmol: an open-source Java viewer for chemical structures in 3D, http://www.jmol.org/ ; colour table "
    "org.jmol.c.PAL.argbsCpk (Jmol-SwingJS, " + JMOL_PAL + "); also published at https://jmol.sourceforge.net/jscolors/"
)


# --- helpers ---------------------------------------------------------------------------------
def _sqlite_rows(sql: str) -> list[tuple[Any, ...]]:
    data = fetch(MENDELEEV_DB)
    path = Path(tempfile.gettempdir()) / f"mendeleev-{sha256(data)[:12]}.db"
    if not path.exists():
        path.write_bytes(data)
    with sqlite3.connect(path) as con:
        return list(con.execute(sql))


def _ase_array(url: str, name: str) -> dict[str, float | None]:
    """Parse an ASE ``name = np.array([... value,  # Sym ...])`` table into {symbol: value}."""
    text = fetch(url).decode("utf-8")
    block = text[text.index(f"{name} = np.array([") :]
    block = block[: block.index("])")]
    out: dict[str, float | None] = {}
    for value, sym in re.findall(r"^\s*([\w.]+),\s*#\s*(\w+)", block, re.M):
        if value == "_R_DUMMY":
            continue
        out[sym] = None if value in ("np.nan", "missing") else float(value)
    return out


def _parse_alvarez() -> dict[int, dict[str, Any]]:
    import fitz  # PyMuPDF; only this table needs it

    pdf = fetch(ALVAREZ_PDF)
    doc = fitz.open(stream=pdf, filetype="pdf")
    # Column x-origins (pt) of Table 1 in the published layout.
    columns = [("Z", 40), ("E", 75), ("bondi", 115), ("batsanov", 160), ("rvdw", 222), ("pct", 270), ("data", 330), ("obs", 385)]

    def column(x: float) -> str:
        name = columns[0][0]
        for col, x0 in columns:
            if x >= x0:
                name = col
        return name

    rows: dict[int, dict[str, Any]] = {}
    for page_no in range(doc.page_count):
        page = doc[page_no]
        if "Table 1" not in page.get_text():
            continue
        lines: dict[float, list[tuple[float, str, bool]]] = {}
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        y = round(span["bbox"][1])
                        italic = span["font"].endswith(".I") or bool(span["flags"] & 2)
                        lines.setdefault(y, []).append((span["bbox"][0], span["text"], italic))
        for spans in lines.values():
            spans.sort()
            cells: dict[str, list[tuple[str, bool]]] = {}
            for x, text, italic in spans:
                cells.setdefault(column(x), []).append((text, italic))
            z_text = "".join(t for t, _ in cells.get("Z", [])).strip()
            sym = "".join(t for t, _ in cells.get("E", [])).strip()
            if not z_text.isdigit() or not (1 <= int(z_text) <= 118) or SYMBOLS[int(z_text)] != sym:
                continue
            z = int(z_text)
            rv_text = "".join(t for t, _ in cells.get("rvdw", [])).strip()
            rv_italic = any(i for t, i in cells.get("rvdw", []) if re.search(r"\d", t))
            if not rv_text:
                continue
            bracket = rv_text.startswith("[")

            def num(col: str) -> float | None:
                text = "".join(t for t, _ in cells.get(col, [])).strip()
                return float(text) if text else None

            data_text = "".join(t for t, _ in cells.get("data", [])).replace(" ", "").replace(" ", "").strip()
            pct_text = "".join(t for t, _ in cells.get("pct", [])).strip()
            rows[z] = {
                "radius_A": float(rv_text.strip("[]")),
                "larger_uncertainty": bool(rv_italic),
                "rough_estimate": bracket,
                "vdw_peak_percent": int(pct_text) if pct_text else None,
                "n_distances": int(data_text) if data_text else None,
                "bondi_A": num("bondi"),
                "batsanov_A": num("batsanov"),
            }
    return rows


def _parse_cordero() -> dict[int, dict[str, Any]]:
    text = fetch(WIKI_RAW).decode("utf-8")
    table = text[text.index("Covalent radii in pm from analysis") :]
    table = table[: table.index("\n|}")]
    rows: list[list[str]] = []
    for line in table.split("\n"):
        if line.startswith("|-"):
            rows.append([])
        elif line.startswith("|") and not line.startswith("|+") and rows:
            for cell in line[1:].split("||"):
                # Drop a leading 'attr="..." |' prefix, keep the content.
                if re.match(r'\s*(colspan|style|valign)\S*=', cell) and "|" in cell:
                    cell = cell.rsplit("|", 1)[1]
                rows[-1].append(cell.strip())

    def is_spacer(c: str) -> bool:
        return c in ("", "&nbsp;", "*", "**") or "Radius" in c

    out: dict[int, dict[str, Any]] = {}
    i = 0
    while i + 2 < len(rows):
        syms = [c for c in rows[i] if re.fullmatch(r"[A-Z][a-z]?", c)]
        nums = [c for c in rows[i + 1] if c.isdigit()]
        vals = [c for c in rows[i + 2] if not is_spacer(c)]
        if syms and len(syms) == len(nums) == len(vals):
            for sym, z_text, val in zip(syms, nums, vals, strict=True):
                z = int(z_text)
                if SYMBOLS[z] != sym:
                    raise SystemExit(f"Cordero table: Z={z} labelled {sym}")
                out[z] = _cordero_value(val)
            i += 3
        else:
            i += 1
    return out


def _radius(token: str) -> tuple[float, float | None]:
    m = re.fullmatch(r"(\d+)(?:\((\d+)\))?", token.strip())
    if not m:
        raise SystemExit(f"cannot parse covalent radius {token!r}")
    esd = None if m.group(2) is None else int(m.group(2)) / 100
    return int(m.group(1)) / 100, esd


def _cordero_value(cell: str) -> dict[str, Any]:
    cell = re.sub(r"<sup>(\d)</sup>", r"\1", cell).replace("&nbsp;", " ")
    parts = [p.strip() for p in cell.split("<br>")]
    if len(parts) == 1:
        r, esd = _radius(parts[0])
        return {"radius_A": r, "esd_A": esd}
    variants: dict[str, dict[str, float | None]] = {}
    for part in parts:
        label, token = part.rsplit(" ", 1)
        key = {"sp3": "sp3", "sp2": "sp2", "sp": "sp", "l.s.": "low_spin", "h.s.": "high_spin"}[label.strip()]
        r, esd = _radius(token)
        variants[key] = {"radius_A": r, "esd_A": esd}
    default = "sp3" if "sp3" in variants else "low_spin"
    return {**variants[default], "default_variant": default, "variants": variants}


def _parse_jmol() -> dict[int, str]:
    text = fetch(JMOL_PAL).decode("utf-8")
    block = text[text.index("int[] argbsCpk = {") :]
    block = block[: block.index("};")]
    out: dict[int, str] = {}
    for argb, sym, z in re.findall(r"0x([0-9A-Fa-f]{8}),?\s*//\s*([A-Z][a-z]?|Xx)\s+(\d+)", block):
        z_int = int(z)
        if z_int == 0:
            continue
        if SYMBOLS[z_int] != sym:
            raise SystemExit(f"Jmol table: Z={z_int} labelled {sym}")
        out[z_int] = "#" + argb[2:].upper()
    return out


# --- cross-checks ---------------------------------------------------------------------------
def _crosscheck(name: str, ours: dict[int, float | None], other: dict[int, float | None], allowed: dict[int, str]) -> dict[str, Any]:
    diffs = {}
    for z in range(1, 119):
        a, b = ours.get(z), other.get(z)
        if (a is None) != (b is None) or (a is not None and b is not None and abs(a - b) > 1e-9):
            if z not in allowed:
                raise SystemExit(f"{name}: unexplained difference at Z={z} ({SYMBOLS[z]}): ours {a} vs {b}")
            diffs[SYMBOLS[z]] = {"ours": a, "other": b, "reason": allowed[z]}
    return {"agrees_except": diffs} if diffs else {"agrees_except": {}}


def build_vdw() -> dict[str, Any]:
    rows = _parse_alvarez()
    if len(rows) < 90:
        raise SystemExit(f"Alvarez Table 1: parsed only {len(rows)} rows")
    ase = _ase_array(ASE_VDW, "vdw_radii")
    ase_by_z = {z: ase.get(SYMBOLS[z]) for z in range(1, 119)}
    mend = {z: (None if v is None else v / 100) for z, v in _sqlite_rows("select atomic_number, vdw_radius_alvarez from elements")}
    ours = {z: r["radius_A"] for z, r in rows.items()}
    noble = "mendeleev replaces the noble gases with Vogt & Alvarez 2014 values"
    checks = {
        "ase": {"source": ASE_VDW, **_crosscheck("vdW vs ASE", ours, ase_by_z, {})},
        "mendeleev": {
            "source": MENDELEEV_DB,
            **_crosscheck("vdW vs mendeleev", ours, mend, {18: noble, 36: noble, 54: noble, 86: noble}),
        },
    }
    missing = {z: {"radius_A": None, "note": "Not given in Alvarez 2013 Table 1."} for z in range(1, 119) if z not in rows}
    payload = {
        "_provenance": provenance(
            source=ALVAREZ_CITATION,
            license_or_terms=(
                "Article is open access under CC BY-NC 3.0 (RSC; repository copy cc by-nc 3.0 es). Only factual "
                "numerical values are extracted, with attribution; no prose, figures or layout are reproduced. "
                "Bondi and Batsanov columns are as reprinted in the same table (A. Bondi, J. Phys. Chem. 68, 441 "
                "(1964), doi:10.1021/j100785a001; S. S. Batsanov, Inorg. Mater. 37, 871 (2001), "
                "doi:10.1023/A:1011625728803)."
            ),
            transform="tools/refdata/build_element_rendering.py (PDF Table 1 parsed by column position with PyMuPDF)",
            units={"radius_A": "angstrom", "bondi_A": "angstrom", "batsanov_A": "angstrom", "vdw_peak_percent": "%"},
            uncertainty=(
                "The paper gives no per-value standard deviation. Its stated guidance: 'differences of 0.1 A or "
                "less should not be considered to be significant'. 'larger_uncertainty' = value printed in italics; "
                "'rough_estimate' = value printed in square brackets (very small structural datasets)."
            ),
            download_sha256=sha256(fetch(ALVAREZ_PDF)),
            crosschecks=checks,
            related=(
                "Noble-gas radii were revised by A. Vogt and S. Alvarez, Inorg. Chem. 53, 9260 (2014), doi:10.1021/ic501364h "
                "(correction doi:10.1021/ic502140y); not stored here (primary text not accessed)."
            ),
            n_elements_with_radius=len(rows),
        ),
        **element_table({**rows, **missing}),
    }
    return payload


def build_covalent() -> dict[str, Any]:
    rows = _parse_cordero()
    if len(rows) != 96:
        raise SystemExit(f"Cordero table: expected Z = 1-96, parsed {len(rows)}")
    ase = _ase_array(ASE_DATA, "covalent_radii")
    ase_by_z = {z: ase.get(SYMBOLS[z]) for z in range(1, 119)}
    mend = {z: (None if v is None else round(v / 100, 4)) for z, v in _sqlite_rows("select atomic_number, covalent_radius_cordero from elements")}
    ours = {z: r["radius_A"] for z, r in rows.items()}
    avg = "mendeleev stores the average of the hybridisation / spin variants"
    checks = {
        "ase": {"source": ASE_DATA, **_crosscheck("covalent vs ASE", ours, ase_by_z, {})},
        "mendeleev": {"source": MENDELEEV_DB, **_crosscheck("covalent vs mendeleev", ours, mend, {6: avg, 25: avg, 26: avg, 27: avg})},
    }
    missing = {z: {"radius_A": None, "esd_A": None, "note": "Not covered by Cordero 2008 (Z > 96)."} for z in range(97, 119)}
    for row in rows.values():
        if row["esd_A"] is None and "variants" not in row:
            row["note"] = "No standard deviation given in the source (estimate / very few data)."
    return {
        "_provenance": provenance(
            source=CORDERO_CITATION,
            license_or_terms=(
                "Article is not open access; the numerical radii are facts reproduced with attribution. "
                f"Transcribed from the reproduction of Table 2 in Wikipedia 'Covalent radius' (revision {WIKI_REVISION}, "
                "text CC BY-SA 4.0; numbers only used), and cross-checked against two independent transcriptions "
                "(ASE, LGPL-2.1+; mendeleev, MIT)."
            ),
            transform="tools/refdata/build_element_rendering.py (wikitext table parsed; pm -> angstrom)",
            units={"radius_A": "angstrom", "esd_A": "angstrom (standard deviation of the CSD distance sample)"},
            uncertainty="esd_A = the standard deviation printed in parentheses in the source; null where none is given.",
            conventions=(
                "radius_A is the default variant: sp3 for carbon (sp2 0.73, sp 0.69 in 'variants') and low spin "
                "for Mn, Fe, Co (high spin in 'variants'), matching the common ASE/Open Babel convention."
            ),
            download_sha256=sha256(fetch(WIKI_RAW)),
            crosschecks=checks,
            n_elements_with_radius=len(rows),
        ),
        **element_table({**rows, **missing}),
    }


def build_colors() -> dict[str, Any]:
    colors = _parse_jmol()
    if sorted(colors) != list(range(1, 110)):
        raise SystemExit(f"Jmol palette: expected Z = 1-109, got {len(colors)}")
    mend = dict(_sqlite_rows("select atomic_number, jmol_color from elements"))
    for z, hexcode in colors.items():
        other = mend.get(z)
        if other is None or other.upper() != hexcode:
            raise SystemExit(f"Jmol colour Z={z}: {hexcode} vs mendeleev {other}")
    ase_text = fetch(ASE_COLORS).decode("utf-8")
    block = ase_text[ase_text.index("jmol_colors = np.array([") :]
    triples = re.findall(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", block[: block.index("])")])
    ase_mismatch = []
    for z, hexcode in colors.items():
        r, g, b = (float(v) for v in triples[z])
        rgb = tuple(int(hexcode[k : k + 2], 16) / 255 for k in (1, 3, 5))
        if max(abs(x - y) for x, y in zip(rgb, (r, g, b), strict=True)) > 2.5e-3:
            ase_mismatch.append(SYMBOLS[z])
    if ase_mismatch:
        raise SystemExit(f"Jmol colours disagree with ASE for {ase_mismatch}")
    values: dict[int, dict[str, Any]] = {z: {"hex": h, "rgb": [int(h[k : k + 2], 16) for k in (1, 3, 5)]} for z, h in colors.items()}
    for z in range(110, 119):
        values[z] = {"hex": None, "rgb": None, "note": "Not defined by Jmol; Jmol draws undefined elements in its 'Xx' colour."}
    return {
        "_provenance": provenance(
            source=JMOL_CITATION,
            license_or_terms=(
                "Jmol is LGPL-2.1-or-later; the colour values are a display convention (not physical data) "
                "reproduced with attribution. No Jmol code is copied."
            ),
            transform="tools/refdata/build_element_rendering.py (PAL.argbsCpk parsed; ARGB -> #RRGGBB)",
            units={"hex": "sRGB #RRGGBB", "rgb": "sRGB 0-255"},
            uncertainty="Not applicable: colours are a convention. Jmol notes several entries were 'changed from ghemical'.",
            download_sha256=sha256(fetch(JMOL_PAL)),
            default_hex="#FF1493",
            default_note="Jmol's colour for index 0 ('Xx', unknown element); use for Z = 110-118.",
            crosschecks={
                "mendeleev": {"source": MENDELEEV_DB, "agrees_except": {}},
                "ase_jmol_colors": {"source": ASE_COLORS, "agrees_within": "1/255 per channel"},
            },
        ),
        **element_table(values),
    }


def main() -> int:
    cache_dir()
    outputs = {
        "vdw_radii_alvarez2013.json": build_vdw,
        "covalent_radii_cordero2008.json": build_covalent,
        "jmol_colors.json": build_colors,
    }
    for name, builder in outputs.items():
        payload = builder()
        path = OUT_DIR / name
        write_json(path, payload)
        n = sum(1 for e in payload["elements"].values() if e.get("radius_A") is not None or e.get("hex") is not None)
        print(f"{rel(path)}: {n} elements with data; crosschecks: {payload['_provenance'].get('crosschecks')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
