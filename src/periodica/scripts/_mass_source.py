"""Build-time curated masses for generated tiers (D3).

Interim Python twin of ``rust/periodica_core/src/mass_source.rs``, used by the
build runner until the scripts call the native composer (pyfacade wiring,
then M7). The Rust test ``derived_tiers_are_reproduced_by_the_native_composer``
regenerates every derived entry with the Rust implementation and compares it
with the committed files, so the two cannot drift apart silently.

The rules live in ``data/config/composition_rules.json`` under
``mass_sources``: which tier takes its mass from which reference dataset, and
the CODATA factors used to derive ``Mass_MeVc2`` and ``Mass_kg``.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_RULES = _DATA_DIR / "config" / "composition_rules.json"


class MassSourceError(ValueError):
    """An entry cannot be given a curated mass."""


def _load_rules() -> Any:
    # The rules file may carry `//` comments (JSONC); it holds no URLs.
    text = re.sub(r"//.*?(?=\n|$)", "", _RULES.read_text(encoding="utf-8"))
    return json.loads(text)


def _load(path: Path) -> Any:
    # Reference datasets are strict JSON; their provenance holds URLs, so
    # they must not go through `//` comment stripping.
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _sources() -> Optional[Dict[str, Any]]:
    cfg = _load_rules().get("mass_sources")
    if not cfg:
        return None
    elements = _load(_DATA_DIR / cfg["element_weights"]["path"])
    nuclides = _load(_DATA_DIR / cfg["nuclide_masses"]["path"])
    return {
        "cfg": cfg,
        "elements": {int(e["Z"]): e for e in elements["elements"]},
        "nuclides": {(int(n["Z"]), int(n["A"])): n for n in nuclides["nuclides"]},
    }


def _count(entry: Dict[str, Any], key: str) -> int:
    v = (entry.get("Composition") or {}).get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != int(v):
        raise MassSourceError(f"MassSource: entry has no integer Composition.{key} count")
    return int(v)


def _neutral_mass(src: Dict[str, Any], policy: str, z: int, a: int) -> Tuple[float, str]:
    cfg = src["cfg"]
    el_label = cfg["element_weights"]["label"]
    if policy == "element":
        el = src["elements"].get(z)
        if el is not None and el.get("Mass_amu") is not None:
            what = (
                "abridged atomic weight (the standard weight is an interval)"
                if el.get("value_kind") == "abridged"
                else "standard atomic weight"
            )
            return float(el["Mass_amu"]), f"{el_label} {what} of {el['symbol']}"
    nuc = src["nuclides"].get((z, a))
    if nuc is None:
        raise MassSourceError(f"MassSource: no curated mass for element Z={z} or nuclide Z={z} A={a}")
    label = f"{cfg['nuclide_masses']['label']} atomic mass of {nuc['nuclide']}"
    if nuc.get("estimated"):
        label += " (estimated)"
    if policy == "element":
        label += f"; {el_label} gives no standard atomic weight for this element"
    return float(nuc["atomic_mass_u"]), label


def apply_mass_source(tier: str, entry: Dict[str, Any]) -> bool:
    """Overwrite ``entry``'s masses with the curated value for ``tier``.

    Returns False (entry untouched) for a tier with no declared source.
    """
    src = _sources()
    if src is None:
        return False
    cfg = src["cfg"]
    policy = cfg.get("tiers", {}).get(tier)
    if policy is None:
        return False
    keys = cfg.get("composition_keys", {})
    z = _count(entry, keys.get("protons", "P"))
    n = _count(entry, keys.get("neutrons", "N"))
    e = _count(entry, keys.get("electrons", "E"))
    neutral, label = _neutral_mass(src, policy, z, z + n)
    excess = e - z
    mass_amu = neutral + float(excess) * cfg["electron_mass_u"]
    if excess:
        k = abs(excess)
        noun = "electron mass" if k == 1 else "electron masses"
        label += f", {'plus' if excess > 0 else 'minus'} {k} {noun}"
    entry["Mass_amu"] = mass_amu
    entry["Mass_MeVc2"] = mass_amu * cfg["MeV_per_u"]
    entry["Mass_kg"] = mass_amu * cfg["kg_per_u"]
    entry["MassSource"] = label
    return True
