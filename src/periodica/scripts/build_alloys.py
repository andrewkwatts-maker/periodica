"""Generate data/derived/alloys/ from inputs/alloys.json.

Atomic composition resolves through the `atoms` tier; bulk mechanical /
thermal properties are attached from the input table directly.
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("alloys.json")


if __name__ == "__main__":
    main()
