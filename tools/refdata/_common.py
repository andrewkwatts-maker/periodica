"""Shared helpers for the reference-data generator scripts in ``tools/refdata``.

Every script in this folder downloads (or transcribes) a published dataset and
writes a JSON file carrying a top-level ``_provenance`` object.  The helpers
here keep download caching, deterministic JSON output and the provenance
block identical across scripts.

Downloads are cached under ``$REFDATA_CACHE`` (default: the system temp
folder) so re-running a script is cheap and does not hammer the source.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "periodica-refdata/1.0 (+https://github.com/andrewkwatts-maker/periodica)"


def cache_dir() -> Path:
    """Return the download cache folder, creating it if needed."""
    root = Path(os.environ.get("REFDATA_CACHE", Path(tempfile.gettempdir()) / "periodica-refdata"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch(url: str, *, refresh: bool = False) -> bytes:
    """Download ``url`` (cached by URL hash) and return the raw bytes."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    path = cache_dir() / key
    if path.exists() and not refresh:
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https sources
        data: bytes = response.read()
    path.write_bytes(data)
    return data


def fetch_json(url: str, *, refresh: bool = False) -> Any:
    """Download ``url`` and decode it as JSON."""
    return json.loads(fetch(url, refresh=refresh).decode("utf-8"))


def sha256(data: bytes) -> str:
    """Hex SHA-256 of ``data`` (recorded in provenance to pin the exact download)."""
    return hashlib.sha256(data).hexdigest()


def today() -> str:
    """ISO date used for the ``retrieved`` provenance field (override with REFDATA_DATE)."""
    return os.environ.get("REFDATA_DATE", _dt.date.today().isoformat())


def provenance(
    *,
    source: str | list[str],
    license_or_terms: str,
    transform: str,
    units: str | dict[str, str],
    retrieved: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the standard ``_provenance`` block (key order is fixed for stable diffs)."""
    block: dict[str, Any] = {
        "source": source,
        "retrieved": retrieved or today(),
        "license_or_terms": license_or_terms,
        "transform": transform,
        "units": units,
    }
    block.update(extra)
    return block


def write_json(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    """Write ``payload`` as UTF-8 JSON with a trailing newline (LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")


def rel(path: Path) -> str:
    """Repository-relative POSIX path, for messages and provenance fields."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()
