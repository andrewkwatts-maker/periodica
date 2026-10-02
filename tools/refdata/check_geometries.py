"""Check curated geometries against their independent internal-coordinate transcriptions (todo Q3).

Reads only the committed JSON files: for every
``src/periodica/data/reference/molecules/geometry/<name>.json`` with
``has_active_geometry`` it loads ``src/periodica/data/active/geometry/<name>.json``
and re-measures each gated source coordinate from the Cartesians
(<= 1e-4 angstrom, <= 0.01 degree).  This is the same gate the periodica test
suite should apply.

Usage::

    python tools/refdata/check_geometries.py
"""

from __future__ import annotations

import json
import math
import sys

from _common import REPO_ROOT
from _geometry import _cross, _dot, _sub, _unit, angle, dihedral, distance

REFERENCE_DIR = REPO_ROOT / "src" / "periodica" / "data" / "reference" / "molecules" / "geometry"
ACTIVE_DIR = REPO_ROOT / "src" / "periodica" / "data" / "active" / "geometry"
TOL = {"distance": 1e-4, "angle": 0.01, "dihedral": 0.01, "plane_angle": 0.01}


def measure(kind: str, pts: list[tuple[float, float, float]]) -> float:
    if kind == "distance":
        return distance(*pts)
    if kind == "angle":
        return angle(*pts)
    if kind == "dihedral":
        return dihedral(*pts)
    n1 = _unit(_cross(_sub(pts[0], pts[1]), _sub(pts[2], pts[1])))
    n2 = _unit(_cross(_sub(pts[3], pts[4]), _sub(pts[5], pts[4])))
    return math.degrees(math.acos(min(1.0, abs(_dot(n1, n2)))))


def main() -> int:
    failures = 0
    for ref_path in sorted(REFERENCE_DIR.glob("*.json")):
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        if not ref.get("has_active_geometry"):
            print(f"{ref['name']:18s} reference only: {ref.get('why_not_curated', '')[:70]}...")
            continue
        active = json.loads((ACTIVE_DIR / ref_path.name).read_text(encoding="utf-8"))
        xyz = [(a["x"], a["y"], a["z"]) for a in active["atoms"]]
        worst = 0.0
        for c in ref["coordinates"]:
            if not c["gate"]:
                continue
            got = measure(c["type"], [xyz[i] for i in c["atoms"]])
            dev = abs(got - c["value"])
            if c["type"] == "dihedral":
                dev = min(dev, 360.0 - dev)
            worst = max(worst, dev / TOL[c["type"]])
            if dev > TOL[c["type"]]:
                failures += 1
                print(f"FAIL {ref['name']} {c['label']}: {got:.6f} vs {c['value']}")
        print(f"{ref['name']:18s} {active['kind']:12s} {len(xyz):2d} atoms  worst deviation = {worst:.3f} x tolerance")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
