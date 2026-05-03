"""Generate data/derived/composites/ from inputs/composites.json.

Composites typically declare an anisotropic field model so that
sample(name, prop, at=...) returns direction-dependent values.
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("composites.json")


if __name__ == "__main__":
    main()
