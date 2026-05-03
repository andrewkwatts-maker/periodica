"""Generate data/derived/ions/<name>.json from inputs/ions.json.

Composes common ions (H+, OH-, Na+, Ca2+, Fe3+, ...) from {P, N, E}
counts where E != P. No hardcoded charge logic - the charge falls out
naturally from the Charge_e sums in the constituent JSON.
"""
from periodica.scripts._runner import build_from_input


def main() -> dict:
    return build_from_input("ions.json")


if __name__ == "__main__":
    main()
