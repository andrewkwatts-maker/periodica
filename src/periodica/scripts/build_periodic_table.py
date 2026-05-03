"""Generate data/derived/<element>.json from inputs/periodic_table.json.

Composes atomic entries (H, He, Li, ...) from {P, N, E} counts. Run after
build_hadrons (or directly: Proton/Neutron are already in data/active).
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("periodic_table.json")


if __name__ == "__main__":
    main()
