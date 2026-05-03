"""Generate data/derived/ceramics/ from inputs/ceramics.json."""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("ceramics.json")


if __name__ == "__main__":
    main()
