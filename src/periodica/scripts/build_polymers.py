"""Generate data/derived/polymers/ from inputs/polymers.json."""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("polymers.json")


if __name__ == "__main__":
    main()
