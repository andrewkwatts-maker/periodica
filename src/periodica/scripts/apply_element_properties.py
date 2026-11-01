"""Merge the measured element bulk-property table into data/active/elements.

The 118 element sheets shipped with 16 fields each and no thermal, electrical
or elastic data at all. That left every element -- and every alloy, ceramic or
composite whose properties are averaged over its elements -- resolving the
mechanical and thermal engine channels to bare constants, so a stiffness query
about iron answered 100 GPa (the documented default) rather than 211.

This script copies ``data/reference/elements/bulk_properties.json`` into the
element sheets. It is idempotent and non-destructive: a field already present
on a sheet is never overwritten, so hand edits and later curation win. Each
touched sheet records what was added under ``_bulk_properties`` for audit.

    python -m periodica.scripts.apply_element_properties [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_DATA = Path(__file__).parent.parent / "data"
_ELEMENTS_DIR = _DATA / "active" / "elements"
_TABLE_PATH = _DATA / "reference" / "elements" / "bulk_properties.json"


def load_table() -> dict:
    """The curated reference table."""
    return json.loads(_TABLE_PATH.read_text(encoding="utf-8"))


def apply(*, dry_run: bool = False, verbose: bool = True) -> dict:
    """Merge the table into every matching element sheet.

    Returns ``{"updated": int, "fields_added": int, "skipped": [...],
    "unmatched": [...]}``.
    """
    table = load_table()
    per_element: Dict[str, dict] = table["elements"]

    updated = 0
    fields_added = 0
    skipped: List[str] = []
    seen: set = set()

    for path in sorted(_ELEMENTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        symbol = data.get("symbol") or data.get("Symbol")
        if not symbol:
            skipped.append(f"{path.name}: no symbol field")
            continue
        values = per_element.get(str(symbol))
        if values is None:
            skipped.append(f"{path.name}: {symbol} not in the reference table")
            continue
        seen.add(str(symbol))

        added: List[str] = []
        for key, value in values.items():
            if key in data:
                continue  # never overwrite what the sheet already asserts
            data[key] = value
            added.append(key)
        if not added:
            continue

        data["_bulk_properties"] = {
            "source": table.get("source"),
            "conditions": table.get("conditions"),
            "table_version": table.get("version"),
            "fields": sorted(added),
        }
        fields_added += len(added)
        updated += 1
        if verbose:
            print(f"  {path.name}: +{len(added)} ({', '.join(sorted(added))})")
        if not dry_run:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    unmatched = sorted(set(per_element) - seen)
    result = {
        "updated": updated,
        "fields_added": fields_added,
        "skipped": skipped,
        "unmatched": unmatched,
        "dry_run": dry_run,
    }
    print(
        f"{'Would update' if dry_run else 'Updated'} {updated} element sheet(s), "
        f"{fields_added} field(s) added."
    )
    if unmatched:
        print(f"  table entries with no element sheet: {unmatched}", file=sys.stderr)
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only print the summary line.")
    args = parser.parse_args(argv)
    result = apply(dry_run=args.dry_run, verbose=not args.quiet)
    return 0 if not result["skipped"] or result["updated"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
