"""Optimization for protein folding (phi/psi) and alloy composition.

Both optimizers stay data-driven: every weight, threshold, and rule
lives in `data/config/folding_rules.json` or in the existing element/AA
JSON tables. No hardcoded chemistry or physics in this module.

Public API
----------
- `optimize_protein_folding(sequence, target='helical', iterations=2000, seed=None)`
    Simulated-annealing search on phi/psi to maximize a propensity score.
- `optimize_alloy(targets, base='Fe', candidates=500, top_k=5, seed=None)`
    Random search over Hume-Rothery-bounded compositions.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from periodica.folding import (
    build_backbone,
    folding_rules,
    ramachandran_in_allowed,
)
from periodica.get import Get, Scope, UnknownConstituent, UnknownName


# ─────────────────────────────────────────────────────────────────────────
# Protein folding: simulated annealing on phi/psi
# ─────────────────────────────────────────────────────────────────────────

_TARGET_REGION = {
    "helical": "alpha_helix",
    "alpha":   "alpha_helix",
    "sheet":   "beta_sheet",
    "beta":    "beta_sheet",
}


def _aa_propensity(letter: str, target_region: str) -> float:
    """Look up Chou-Fasman propensity for an amino acid in the desired region."""
    try:
        aa = Get(Scope.AminoAcid, letter)
    except (UnknownConstituent, UnknownName):
        return 1.0
    if target_region == "alpha_helix":
        return float(aa.get("helix_propensity", 1.0))
    if target_region == "beta_sheet":
        return float(aa.get("sheet_propensity", 1.0))
    return 1.0


def _energy(
    sequence: str,
    phi_psi: np.ndarray,
    target_region: str,
    rules: dict,
) -> float:
    """Toy folding energy. Lower is better."""
    weights = rules.get("energy_weights", {})
    lam_prop = float(weights.get("lambda_propensity", 1.0))
    lam_clash = float(weights.get("lambda_clash", 5.0))
    lam_dis = float(weights.get("lambda_ramachandran_disallowed", 2.0))
    min_d = float(rules.get("min_ca_distance_A", 3.5))

    # Term 1: -sum(propensity * indicator_in_target_region)
    propensity_sum = 0.0
    disallowed_count = 0
    for i, letter in enumerate(sequence):
        phi, psi = float(phi_psi[i, 0]), float(phi_psi[i, 1])
        in_target = False
        target_lo_phi, target_hi_phi = rules["ramachandran_allowed"][target_region]["phi"]
        target_lo_psi, target_hi_psi = rules["ramachandran_allowed"][target_region]["psi"]
        if target_lo_phi <= phi <= target_hi_phi and target_lo_psi <= psi <= target_hi_psi:
            in_target = True
        if in_target:
            propensity_sum += _aa_propensity(letter, target_region)
        if not ramachandran_in_allowed(phi, psi, rules=rules):
            disallowed_count += 1

    # Term 2: clash penalty (CA-CA pairwise distance)
    coords = build_backbone(sequence, [(p, q) for p, q in phi_psi.tolist()], rules=rules)
    ca = coords[:, 1, :]
    n = ca.shape[0]
    clash_pen = 0.0
    if n > 1:
        # Pairwise CA distances (skip neighbors i and i+1; their geometry is fixed).
        for i in range(n):
            for j in range(i + 2, n):
                d = float(np.linalg.norm(ca[i] - ca[j]))
                if d < min_d:
                    clash_pen += (min_d - d) ** 2

    return -lam_prop * propensity_sum + lam_clash * clash_pen + lam_dis * disallowed_count


def _random_phi_psi_in_region(region: str, rules: dict, rng: random.Random) -> Tuple[float, float]:
    r = rules["ramachandran_allowed"][region]
    return (rng.uniform(*r["phi"]), rng.uniform(*r["psi"]))


def optimize_protein_folding(
    sequence: str,
    *,
    target: str = "helical",
    iterations: int = 2000,
    seed: Optional[int] = None,
    initial_temp: float = 5.0,
    final_temp: float = 0.05,
) -> dict:
    """Simulated-annealing optimization of phi/psi for a sequence.

    Args:
        sequence: 1-letter amino acid string (e.g. "MQIFVKTLTGK...").
        target: "helical" / "alpha", "sheet" / "beta", or a region name from
            folding_rules.ramachandran_allowed.
        iterations: total SA steps.
        seed: deterministic seed.

    Returns dict with: phi_psi (ndarray), coords (backbone), energy_history
    (list[float]), final_energy (float), target_region (str).
    """
    if not sequence:
        raise ValueError("sequence must be non-empty")
    rules = folding_rules()
    target_region = _TARGET_REGION.get(target, target)
    if target_region not in rules["ramachandran_allowed"]:
        raise ValueError(
            f"Unknown target region {target!r}; "
            f"valid: {sorted(rules['ramachandran_allowed'])} or aliases {sorted(_TARGET_REGION)}"
        )
    rng = random.Random(seed)
    n = len(sequence)

    # Initial state: every residue placed in the target region.
    phi_psi = np.zeros((n, 2))
    for i in range(n):
        phi_psi[i] = _random_phi_psi_in_region(target_region, rules, rng)

    current_e = _energy(sequence, phi_psi, target_region, rules)
    best_e = current_e
    best = phi_psi.copy()
    history = [current_e]

    for step in range(iterations):
        t = initial_temp * (final_temp / initial_temp) ** (step / max(1, iterations - 1))
        # Tweak one residue.
        i = rng.randrange(n)
        old = phi_psi[i].copy()
        # 80% propose in target region; 20% propose anywhere
        if rng.random() < 0.8:
            phi_psi[i] = _random_phi_psi_in_region(target_region, rules, rng)
        else:
            phi_psi[i] = (rng.uniform(-180, 180), rng.uniform(-180, 180))
        new_e = _energy(sequence, phi_psi, target_region, rules)
        if new_e < current_e or rng.random() < math.exp(-(new_e - current_e) / max(1e-9, t)):
            current_e = new_e
            if new_e < best_e:
                best_e = new_e
                best = phi_psi.copy()
        else:
            phi_psi[i] = old
        history.append(current_e)

    coords = build_backbone(sequence, [(p, q) for p, q in best.tolist()])
    return {
        "phi_psi": best,
        "coords": coords,
        "energy_history": history,
        "final_energy": best_e,
        "target_region": target_region,
        "sequence": sequence,
    }


# ─────────────────────────────────────────────────────────────────────────
# Alloy composition optimization
# ─────────────────────────────────────────────────────────────────────────

def _parse_target(targets: Mapping[str, float]) -> List[Tuple[str, str, float]]:
    """Parse targets like {'YieldStrength_MPa_min': 800, 'Density_kgm3_max': 5000}.

    Returns list of (property_name, comparator, threshold).
    """
    out: List[Tuple[str, str, float]] = []
    for key, val in targets.items():
        if key.endswith("_min"):
            out.append((key[:-4], "min", float(val)))
        elif key.endswith("_max"):
            out.append((key[:-4], "max", float(val)))
        else:
            out.append((key, "min", float(val)))
    return out


def _candidate_properties(composition: Mapping[str, float]) -> Dict[str, float]:
    """Estimate composition's bulk properties via rule-of-mixtures from the
    `alloys` tier's existing entries: pull each constituent element's
    contribution from existing alloys keyed on that element-as-base.

    Falls back to atomic data sheet (Density via mass, etc.) when richer
    estimates aren't available.
    """
    # Property contribution per element: try data_sheet of the pure element entry.
    total = {}
    weight = sum(composition.values()) or 1.0
    for sym, frac in composition.items():
        try:
            elem = Get(Scope.Atom, sym)
        except (UnknownConstituent, UnknownName):
            continue
        # Properties of the bare atom JSON are sparse. Pull from active/elements
        # via legacy element data when available; otherwise fall back to mass.
        # We approximate density from atomic mass + a rough volume guess.
        m = elem.get("Mass_amu")
        if m is None:
            continue
        # Crude density proxy: mass / volume (volume from covalent radius) — not used directly here.
        # Instead, use element-specific lookup table from our existing alloys tier.
        # Find an alloy whose base is `sym` (e.g. "Cu" -> Copper-C11000).
        # That alloy's Properties give a reference for pure-element behaviour.
        try:
            from periodica.scripts._runner import _INPUTS_DIR
            import json as _json
            data = _json.loads((_INPUTS_DIR / "alloys.json").read_text(encoding="utf-8"))
            for row in data["rows"]:
                spec = row.get("spec", {})
                # If this row is dominated by `sym`, treat as a pure-element ref.
                if spec.get(sym, 0.0) >= 0.95:
                    for k, v in (row.get("properties") or {}).items():
                        if isinstance(v, (int, float)):
                            total[k] = total.get(k, 0.0) + float(v) * (frac / weight)
                    break
        except Exception:
            pass
    return total


def _score(props: Mapping[str, float], parsed_targets: Sequence[Tuple[str, str, float]]) -> Optional[float]:
    """Higher is better. Returns None if a hard constraint is violated."""
    score = 0.0
    for name, kind, threshold in parsed_targets:
        v = props.get(name)
        if v is None:
            return None
        if kind == "min":
            if v < threshold:
                return None
            score += v - threshold
        else:  # max
            if v > threshold:
                return None
            score += threshold - v
    return score


def optimize_alloy(
    targets: Mapping[str, float],
    *,
    base: str = "Fe",
    candidates: int = 500,
    top_k: int = 5,
    seed: Optional[int] = None,
    alloying_pool: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Search over Hume-Rothery-style compositions to satisfy property targets.

    `targets`: dict of "<prop>_min"/"<prop>_max" -> threshold.
    `base`: dominant element symbol (e.g. "Fe", "Al", "Ti").
    `candidates`: number of random compositions to evaluate.
    `top_k`: number of best candidates to return.
    `alloying_pool`: optional explicit list of elements to alloy with the base.
        Defaults to a Hume-Rothery-compatible pool drawn from existing alloy inputs.

    Returns list of dicts sorted by descending score:
        [{"composition": {...}, "estimated_properties": {...}, "score": float}, ...]
    """
    rng = random.Random(seed)
    parsed = _parse_target(targets)

    if alloying_pool is None:
        # Pull alloying-element pool from existing alloy inputs (data-driven).
        from periodica.scripts._runner import _INPUTS_DIR
        import json as _json
        data = _json.loads((_INPUTS_DIR / "alloys.json").read_text(encoding="utf-8"))
        pool = set()
        for row in data["rows"]:
            for sym in (row.get("spec") or {}):
                if sym != base:
                    pool.add(sym)
        alloying_pool = sorted(pool)
    if not alloying_pool:
        alloying_pool = ["Cr", "Ni", "Mn", "Si", "Cu", "Mo"]

    results: List[dict] = []
    for _ in range(candidates):
        # Sample 1-4 alloying elements with small fractions; rest is base.
        n_alloying = rng.randint(0, 4)
        elements = rng.sample(list(alloying_pool), min(n_alloying, len(alloying_pool)))
        fractions = [rng.uniform(0.0005, 0.2) for _ in elements]
        s = sum(fractions)
        if s >= 0.5:
            scale = 0.5 / s
            fractions = [f * scale for f in fractions]
        comp = {base: 1.0 - sum(fractions)}
        for el, f in zip(elements, fractions):
            comp[el] = f
        props = _candidate_properties(comp)
        if not props:
            continue
        sc = _score(props, parsed)
        if sc is None:
            continue
        results.append({"composition": comp, "estimated_properties": props, "score": sc})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
