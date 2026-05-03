"""Generate data/derived/isotopes/<name>.json from inputs/isotopes.json.

Composes specific isotopes (H-1, H-2, C-12, C-14, U-235, ...) from
{P, N, E} counts. Independent of the periodic-table generator.
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("isotopes.json")


if __name__ == "__main__":
    main()
