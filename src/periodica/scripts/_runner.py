"""Generic JSON-input driver: load rows, Get(spec, from_=...), Save(name, ..., tier=...).

Input file shape:

    {
        "from": "subatomic" | ["subatomic","fundamentals"],
        "tier": "atoms",
        "requires": ["fundamentals", "subatomic"],   // optional
        "rows": [
            { "name": "H", "spec": { "P": 1, "N": 0, "E": 1 } },
            ...
        ]
    }

A bare JSON array is also accepted (legacy); in that case `from_` is
unrestricted and outputs go to `data/derived/general/`.

Discovery & ordering
--------------------
`default_inputs()` lists every JSON under `scripts/inputs/`.
`build_all()` topologically sorts them by their `requires` field and runs
each. New inputs can be added by dropping a JSON into the inputs folder
or by passing `extra_dirs=[...]` paths to `build_all`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from periodica.get import Get, Save, list_tiers, reload_registry


_INPUTS_DIR = Path(__file__).parent / "inputs"


def _resolve_input_path(input_name_or_path: str) -> Path:
    p = Path(input_name_or_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    candidate = _INPUTS_DIR / p
    if candidate.exists():
        return candidate
    if not str(p).endswith(".json"):
        candidate2 = _INPUTS_DIR / f"{p}.json"
        if candidate2.exists():
            return candidate2
    raise FileNotFoundError(
        f"Input not found: tried {p!r} and {candidate!r}"
    )


def _normalize_input(payload: Any) -> dict:
    if isinstance(payload, list):
        return {
            "from": None,
            "tier": "general",
            "requires": [],
            "rows": payload,
        }
    if isinstance(payload, dict) and "rows" in payload:
        return {
            "from": payload.get("from"),
            "tier": payload.get("tier") or "general",
            "requires": list(payload.get("requires") or []),
            "rows": payload["rows"],
        }
    raise ValueError(
        "Input must be either a JSON array or an object with a 'rows' key."
    )


def build_from_input(
    input_name_or_path: str,
    *,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Read an input file and Save() each row.

    Returns ``{"ok": int, "total": int, "errors": [(name, msg), ...], "tier": str, "from": str|list|None}``.
    """
    path = _resolve_input_path(input_name_or_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = _normalize_input(payload)
    rows = cfg["rows"]
    from_ = cfg["from"]
    tier = cfg["tier"]

    if not isinstance(rows, list):
        raise ValueError(f"Input {path}: 'rows' must be a JSON array")
    total = len(rows)
    ok = 0
    errors = []

    for i, row in enumerate(rows, start=1):
        name = row.get("name")
        spec = row.get("spec")
        if not name or spec is None:
            errors.append((name or "<missing>", "row missing name or spec"))
            continue
        try:
            composed = Get(spec, from_=from_)
            # Optional row enrichments: bulk properties + field model + aliases.
            # An explicit "symbol" intentionally shadows curated entries with
            # the same Symbol (e.g. Hydrogen 'H' shadows Higgs Boson 'H'); omit
            # to keep the entry at filename-stem priority only.
            sym = row.get("symbol")
            if sym:
                composed["Symbol"] = str(sym)
            props = row.get("properties")
            if isinstance(props, dict) and props:
                composed["Properties"] = dict(props)
            field = row.get("field")
            if isinstance(field, dict) and field:
                composed["Field"] = dict(field)
            aliases = row.get("aliases")
            if isinstance(aliases, list) and aliases:
                composed["Aliases"] = list(aliases)
            Save(name, composed, tier=tier, dir=output_dir)
            ok += 1
            if verbose:
                print(f"  {i}/{total}: {name}")
        except Exception as e:
            errors.append((name, f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"  {i}/{total}: {name}  [error: {type(e).__name__}: {e}]")

    print(f"Done: {ok}/{total} entries saved to tier {tier!r}.  errors: {len(errors)}")
    for n, msg in errors:
        print(f"  - {n}: {msg}", file=sys.stderr)
    return {
        "ok": ok,
        "total": total,
        "errors": errors,
        "tier": tier,
        "from": from_,
        "path": str(path),
    }


def default_inputs() -> List[Path]:
    """Return all default input JSON paths shipped with the package."""
    return sorted(_INPUTS_DIR.glob("*.json"))


def _read_meta(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_input(payload)


def _topo_sort(inputs: List[Path]) -> List[Path]:
    """Order inputs so each runs after the tiers it `requires` are ready.

    Active-tier dependencies (fundamentals, subatomic) are assumed always
    available. Other dependencies must be satisfied by an earlier input.
    """
    metas = [(p, _read_meta(p)) for p in inputs]
    produces = {meta["tier"]: p for p, meta in metas}
    static_tiers = {"fundamentals", "subatomic"}

    ordered: List[Path] = []
    placed: set = set()

    def visit(p: Path, meta: dict, stack: set):
        if p in placed:
            return
        if p in stack:
            raise RuntimeError(
                f"Cyclic dependency involving {p.name}; chain: {[s.name for s in stack]}"
            )
        stack.add(p)
        for req in meta.get("requires", []):
            if req in static_tiers:
                continue
            producer = produces.get(req)
            if producer is None:
                # Unknown requirement; skip and let the runner surface errors at row time.
                continue
            visit(producer, dict(_read_meta(producer)), stack)
        stack.discard(p)
        placed.add(p)
        ordered.append(p)

    for p, meta in metas:
        visit(p, meta, set())
    return ordered


def build_all(
    *,
    extra_dirs: Optional[Iterable[Path]] = None,
    only: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> List[dict]:
    """Run every default input plus any in `extra_dirs`, in dependency order.

    `only` (optional iterable of input filenames or stems) restricts the run.
    Returns the list of per-script result dicts.
    """
    inputs: List[Path] = list(default_inputs())
    for d in extra_dirs or ():
        d = Path(d)
        if d.is_dir():
            inputs.extend(sorted(d.glob("*.json")))
        elif d.is_file():
            inputs.append(d)
    if only is not None:
        wanted = set()
        for o in only:
            stem = Path(o).stem
            wanted.add(stem)
        inputs = [p for p in inputs if p.stem in wanted]
    ordered = _topo_sort(inputs)

    if verbose:
        print(f"Running {len(ordered)} input(s) in order:")
        for p in ordered:
            print(f"  - {p.name}")
        print()

    results: List[dict] = []
    for p in ordered:
        if verbose:
            print(f"=== {p.name} ===")
        results.append(build_from_input(str(p), verbose=verbose))
        reload_registry()
        if verbose:
            print()
    return results
