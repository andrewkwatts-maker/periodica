"""Integrity checks over the 118 curated element data sheets.

Validates the bundled `data/active/elements/NNN_Sy.json` files themselves:
one sheet per element, the required fields present, and the expected
s/p/d/f-block and period populations. (The accuracy comparisons that used
to live here exercised the deleted `simulation_schema` propagation chain.)
"""

import json
from collections import Counter
from pathlib import Path

import pytest

ELEMENTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "periodica" / "data" / "active" / "elements"
)

REQUIRED_FIELDS = (
    "symbol", "name", "atomic_number", "atomic_mass", "block", "period", "isotopes",
)


@pytest.fixture(scope="module")
def elements():
    """Map Z -> parsed element sheet for every NNN_*.json file."""
    sheets = {}
    for z in range(1, 119):
        matches = sorted(ELEMENTS_DIR.glob(f"{z:03d}_*.json"))
        if matches:
            sheets[z] = json.loads(matches[0].read_text(encoding="utf-8"))
    return sheets


def test_element_count(elements):
    assert len(elements) == 118


def test_all_elements_load(elements):
    missing = [z for z in range(1, 119) if z not in elements]
    assert not missing, f"no data sheet for Z={missing}"


def test_all_elements_have_required_fields(elements):
    gaps = [
        (z, sheet.get("symbol", "?"), field)
        for z, sheet in elements.items()
        for field in REQUIRED_FIELDS
        if field not in sheet
    ]
    assert not gaps, f"missing fields: {gaps[:10]}"


def test_blocks_distribution(elements):
    blocks = Counter(sheet.get("block") for sheet in elements.values())
    assert blocks["s"] > 10
    assert blocks["p"] > 30
    assert blocks["d"] > 35
    assert blocks["f"] > 25


def test_periods_distribution(elements):
    periods = Counter(sheet.get("period") for sheet in elements.values())
    assert [periods[p] for p in range(1, 8)] == [2, 8, 8, 18, 18, 32, 32]
