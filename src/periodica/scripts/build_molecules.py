"""Generate data/derived/<molecule>.json from inputs/molecules.json.

Composes molecules (H2O, CH4, ...) from atomic entries already present in
data/derived/. Run AFTER build_periodic_table.
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("molecules.json")


if __name__ == "__main__":
    main()
