"""Generate data/derived/<hadron>.json from inputs/hadrons.json.

Composes named hadrons (Proton, Neutron, ...) from quark fundamentals.
After this runs, names from the input table become valid for Get().
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("hadrons.json")


if __name__ == "__main__":
    main()
