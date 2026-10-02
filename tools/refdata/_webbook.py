"""Ground-state r_e of diatomic molecules from the NIST Chemistry WebBook (helper for todo Q3).

The WebBook "Constants of diatomic molecules" tables (NIST SRD 69) are the
Huber & Herzberg (1979) compilation with later updates.  CCCBDB quotes the
same r_e values but only to three decimals, so diatomics are taken from here.
A subscript digit in the WebBook means "last digit uncertain"; it is kept as
part of the value.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from _common import fetch, sha256

URL = "https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas}&Mask=1000"
CITATION = (
    "P.J. Linstrom and W.G. Mallard (eds.), NIST Chemistry WebBook, NIST Standard Reference Database Number 69, "
    "National Institute of Standards and Technology, Gaithersburg MD, doi:10.18434/T4D303; 'Constants of diatomic "
    "molecules' compiled by K. P. Huber and G. H. Herzberg"
)
HUBER_HERZBERG = (
    "K. P. Huber and G. Herzberg, Molecular Spectra and Molecular Structure IV. Constants of Diatomic Molecules "
    "(Van Nostrand Reinhold, New York, 1979), doi:10.1007/978-1-4757-0961-2"
)
TERMS = (
    "NIST Chemistry WebBook is NIST SRD 69, copyright U.S. Secretary of Commerce (Standard Reference Data Act). "
    "Only the single factual r_e value per molecule is reproduced, attributed to Huber & Herzberg 1979 and the "
    "original measurements the WebBook lists."
)


@dataclass
class DiatomicRe:
    cas: str
    state: str
    r_e: float
    text: str  # value as printed (subscript digit appended)
    decimals: int
    uncertain_last_digit: bool  # Huber & Herzberg printed the last digit as a subscript
    sources: list[str]  # short citations of the measurements listed for the ground state
    url: str
    sha256: str


def _cell_text(cell: str) -> str:
    cell = re.sub(r"(?is)<a [^>]*>.*?</a>", "", cell)  # footnote markers
    cell = re.sub(r"(?is)<sub>(\d+)</sub>", r"\1", cell)  # uncertain last digit
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", cell))).strip()


def ground_state_re(cas: str) -> DiatomicRe:
    url = URL.format(cas=cas)
    raw = fetch(url)
    text = raw.decode("utf-8", errors="replace")
    for table in re.findall(r"(?is)<table[^>]*>(.*?)</table>", text):
        rows = re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", table)
        if not rows:
            continue
        header = [_cell_text(c) for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", rows[0])]
        if not header or header[0] != "State" or "re" not in header:
            continue
        col = header.index("re")
        for i, row in enumerate(rows):
            cells = re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)
            if cells and _cell_text(cells[0]).startswith("X"):
                value_text = _cell_text(cells[col])
                uncertain = re.search(r"(?is)<sub>\d+</sub>", re.sub(r"(?is)<a [^>]*>.*?</a>", "", cells[col])) is not None
                sources: list[str] = []
                for follow in rows[i + 1 : i + 3]:
                    if "&#8627;" in follow or "↳" in follow:
                        sources = [_cell_text(a) for a in re.findall(r"(?is)<a [^>]*>(.*?)</a>", follow)]
                        break
                return DiatomicRe(
                    cas=cas,
                    state=_cell_text(cells[0]),
                    r_e=float(value_text),
                    text=value_text,
                    decimals=len(value_text.split(".")[1]),
                    uncertain_last_digit=uncertain,
                    sources=sources,
                    url=url,
                    sha256=sha256(raw),
                )
    raise RuntimeError(f"WebBook: no ground-state r_e found for CAS {cas}")
