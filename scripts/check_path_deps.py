"""Fail if any Cargo.toml path dependency escapes the repo, except the allowed list.

CI checks out this repo alone, so a `path = "../<sibling>"` dependency breaks it
unless the workflow fetches that sibling. Siblings the workflow fetches are listed
in ALLOWED_SIBLINGS.
"""
import re
import sys
from pathlib import Path

ALLOWED_SIBLINGS = {"Arithma"}
ROOT = Path(__file__).resolve().parent.parent
PATH_RE = re.compile(r'path\s*=\s*"([^"]+)"')


def escaping_paths(manifest: Path, root: Path = ROOT):
    """Yield (path, sibling) for path deps resolving outside root and not allowed."""
    for m in PATH_RE.finditer(manifest.read_text(encoding="utf-8")):
        target = (manifest.parent / m.group(1)).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            sibling = target.relative_to(root.parent).parts[0]
            if sibling not in ALLOWED_SIBLINGS:
                yield m.group(1), sibling


def main() -> int:
    bad = [(mf, p) for mf in ROOT.rglob("Cargo.toml")
           if "target" not in mf.parts for p, _ in escaping_paths(mf)]
    for mf, p in bad:
        print(f"{mf.relative_to(ROOT)}: path dependency leaves the repo: {p}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
