"""Fetch and parse NIST CCCBDB experimental-geometry pages (helper for todo Q3).

CCCBDB (NIST Standard Reference Database 101) serves the experimental
geometry of a species from a session-based form: POST the formula to
``getformx.asp``; if several species share the formula, a selection page
(``getonex.asp``) is returned and the wanted species is chosen by POSTing
its radio value to ``gotonex.asp``.  The resulting ``expgeom2x.asp`` page
lists internal coordinates (with the primary reference "squib" and a type
comment such as "re"), Cartesian coordinates and the full references (with
DOIs).  Raw pages are cached like every other download.
"""

from __future__ import annotations

import hashlib
import html
import http.cookiejar
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from _common import USER_AGENT, cache_dir

BASE = "https://cccbdb.nist.gov"
CITATION = (
    "NIST Computational Chemistry Comparison and Benchmark Database, NIST Standard Reference Database "
    "Number 101, Release 22, May 2022, Editor: Russell D. Johnson III, http://cccbdb.nist.gov/, "
    "doi:10.18434/T47C7Z"
)
TERMS = (
    "CCCBDB is NIST SRD 101, (c) U.S. Secretary of Commerce, all rights reserved (Standard Reference Data Act). "
    "Only individual factual values (bond lengths, angles) are reproduced, each attributed to the primary "
    "publication CCCBDB cites; the database itself is cited as the locator."
)


@dataclass
class Coordinate:
    description: str  # e.g. "rOH", "aHOH", "dHCCH"
    value: float
    atoms: list[int]  # 1-based CCCBDB atom numbers
    reference: str  # squib, e.g. "1979Hoy/Bun:1"
    comment: str


@dataclass
class Page:
    title: str
    point_group: str | None
    internal: list[Coordinate] = field(default_factory=list)
    atoms: list[dict[str, Any]] = field(default_factory=list)  # CCCBDB Cartesians (label, element, x, y, z)
    references: dict[str, dict[str, str]] = field(default_factory=dict)
    sha256: str = ""


REQUEST_GAP_S = 8.0  # CCCBDB rate-limits (HTTP 429); be polite
_last_request = [0.0]


def _open(opener: urllib.request.OpenerDirector, request: urllib.request.Request) -> tuple[str, str]:
    for attempt in range(6):
        wait = _last_request[0] + REQUEST_GAP_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        try:
            with opener.open(request, timeout=120) as response:  # noqa: S310 - fixed https source
                return response.geturl(), response.read().decode("latin-1")
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == 5:
                raise
            time.sleep(60.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _post(opener: urllib.request.OpenerDirector, path: str, data: dict[str, str], referer: str) -> tuple[str, str]:
    body = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(f"{BASE}/{path}", data=body, headers={"User-Agent": USER_AGENT, "Referer": referer})
    return _open(opener, request)


def fetch_page(formula: str, species_name: str | None = None, *, refresh: bool = False) -> str:
    """Return the raw expgeom2x.asp HTML for ``formula`` (choosing ``species_name`` if ambiguous)."""
    key = hashlib.sha256(f"cccbdb-expgeom|{formula}|{species_name}".encode()).hexdigest()[:24]
    path = cache_dir() / key
    if path.exists() and not refresh:
        return path.read_text(encoding="latin-1")
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    _open(opener, urllib.request.Request(f"{BASE}/expgeom1x.asp", headers={"User-Agent": USER_AGENT}))
    url, text = _post(opener, "getformx.asp", {"formula": formula, "submit1": "Submit"}, f"{BASE}/expgeom1x.asp")
    if "gotonex.asp" in text:
        value = _choose_species(text, species_name)
        url, text = _post(opener, "gotonex.asp", {"which": value}, url)
    if "expgeom2x" not in url:
        raise RuntimeError(f"CCCBDB: no experimental geometry page for {formula} ({species_name}); landed on {url}")
    path.write_text(text, encoding="latin-1")
    return text


def _choose_species(text: str, species_name: str | None) -> str:
    """Pick the ground-state row of the named species (or of the first species when no name is given)."""
    rows = re.findall(r'(?is)<tr>\s*<td><INPUT TYPE="radio" NAME="which" VALUE=(\d+)[^>]*>(.*?)</tr>', text)
    for value, row in rows:
        cells = [_clean(c) for c in re.findall(r"(?is)<td[^>]*>(.*?)(?=<td|</tr|$)", row)]
        # full rows: [species, name, state, conformer, description, CAS, sketch]
        if len(cells) >= 4 and cells[2].lower() == "ground":
            if species_name is None or cells[1].lower() == species_name.lower():
                return value
    raise RuntimeError(f"CCCBDB: species {species_name!r} not offered; rows: {[r[0] for r in rows]}")


def _clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _tables(text: str) -> list[list[list[str]]]:
    tables = []
    for table in re.findall(r"(?is)<table[^>]*>(.*?)</table>", text):
        rows = []
        for row in re.findall(r"(?is)<tr[^>]*>(.*?)(?=<tr|$)", table):
            cells = [_clean(c) for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)(?=<t[dh]|</tr|$)", row)]
            if any(cells):
                rows.append(cells)
        tables.append(rows)
    return tables


def parse_page(text: str) -> Page:
    title_match = re.search(r"(?is)<H1>\s*Listing of experimental geometry data for\s*(.*?)</H1>", text)
    title = _clean(title_match.group(1)) if title_match else ""
    pg = re.search(r"(?is)<H2>\s*Point Group\s*(.*?)</H2>", text)
    page = Page(title=title, point_group=_clean(pg.group(1)).replace(" ", "") if pg else None)
    for table in _tables(text):
        header = " ".join(" ".join(r) for r in table[:2])
        if "Description" in header and "Connectivity" in header:
            for cells in table:
                if len(cells) >= 2 and re.fullmatch(r"[rad][A-Z][A-Za-z]*", cells[0]) and _is_number(cells[1]):
                    atoms = [int(c) for c in cells[2:6] if c.isdigit()]
                    rest = [c for c in cells[2 + len(atoms) :] if c]
                    reference = next((c for c in rest if re.match(r"\d{4}", c)), "")
                    comment = " ".join(c for c in rest if c != reference)
                    page.internal.append(Coordinate(cells[0], float(cells[1]), atoms, reference, comment))
        elif table and table[0][:4] == ["Atom", "x (Å)", "y (Å)", "z (Å)"]:
            for cells in table[1:]:
                label = cells[0]
                element = re.match(r"[A-Z][a-z]?", label)
                if element and len(cells) >= 4:
                    page.atoms.append(
                        {"label": label, "element": element.group(0), "x": float(cells[1]), "y": float(cells[2]), "z": float(cells[3])}
                    )
        elif table and table[0][:2] == ["squib", "reference"]:
            for cells in table[1:]:
                if cells and cells[0]:
                    page.references[cells[0]] = {"reference": cells[1] if len(cells) > 1 else "", "doi": cells[2] if len(cells) > 2 else ""}
    page.sha256 = hashlib.sha256(text.encode("latin-1")).hexdigest()
    return page


def _is_number(text: str) -> bool:
    return re.fullmatch(r"-?\d+(\.\d+)?", text) is not None
