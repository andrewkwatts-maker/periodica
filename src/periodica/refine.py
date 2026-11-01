"""Post-composition refinements for `Get()`.

`_compose()` in `periodica.get` is deliberately content-agnostic: it sums the
additive properties of whatever constituents a spec names and knows nothing
about protons, elements or nuclei. That keeps the composer honest, but pure
addition is not physics -- it produces masses that are systematically too
large, because binding energy is missing::

    Get({"P": 26, "N": 30, "E": 26})["Mass_amu"]
        56.4634   summed free-constituent masses
        55.9349   measured mass of Fe-56
        +0.95 %   error, and the same sign for every nucleus

Refinements close that gap without teaching the composer chemistry. Each one
is a small named function registered here and switched on declaratively in
``data/config/composition_rules.json``::

    "refinements": [
      { "refiner": "nuclear_binding", "params": { ... } },
      { "refiner": "mass_units",      "params": { ... } }
    ]

A refiner receives the freshly composed dict plus its own params and mutates
it in place. It must be a no-op for a spec it does not apply to -- the
nuclear refiner sees no proton or neutron constituents in ``{H: 2, O: 1}``
and leaves that molecule alone -- so refiners can be listed unconditionally.

Adding a refinement is a config edit plus one function; nothing in `get.py`
changes. Removing every entry from ``refinements`` restores raw additive
composition exactly.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

#: A refiner mutates the composed entry in place. Return value is ignored.
Refiner = Callable[[dict, Mapping[str, Any]], None]

_REFINERS: Dict[str, Refiner] = {}
_lock = threading.Lock()


def register_refiner(name: str, fn: Refiner) -> None:
    """Register a refiner under `name` so config can switch it on."""
    if not isinstance(name, str) or not name:
        raise ValueError("Refiner name must be a non-empty string.")
    if not callable(fn):
        raise TypeError(f"Refiner {name!r} must be callable.")
    with _lock:
        _REFINERS[name] = fn


def list_refiners() -> List[str]:
    """Every registered refiner name, sorted."""
    return sorted(_REFINERS)


def apply_refinements(composed: dict, rules: Optional[Mapping[str, Any]] = None) -> dict:
    """Run the refinements declared in the composition rules over `composed`.

    Unknown refiner names raise, so a typo in config surfaces immediately
    rather than silently skipping physics. A refiner that raises is
    reported with its name attached.
    """
    if rules is None:
        from periodica.get import _config  # local import: avoids a cycle
        rules = _config()
    for spec in rules.get("refinements") or ():
        if not isinstance(spec, Mapping):
            raise ValueError(f"Each refinement must be an object, got {spec!r}")
        name = spec.get("refiner")
        fn = _REFINERS.get(name)
        if fn is None:
            raise KeyError(
                f"Unknown refiner {name!r}. Registered: {list_refiners()}"
            )
        try:
            fn(composed, spec.get("params") or {})
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Refiner {name!r} failed: {exc}") from exc
    return composed


# ── Helpers ─────────────────────────────────────────────────────────────

def _count_of(composition: Mapping[str, Any], symbols) -> float:
    """Total count in `composition` across any of `symbols` (case-insensitive)."""
    if not symbols:
        return 0.0
    wanted = {str(s).casefold() for s in symbols}
    total = 0.0
    for sym, count in composition.items():
        if str(sym).casefold() in wanted:
            try:
                total += float(count)
            except (TypeError, ValueError):
                continue
    return total


def _as_int(value: Any) -> Optional[int]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    i = int(round(f))
    return i if abs(f - i) < 1e-9 else None


# ── nuclear_binding ─────────────────────────────────────────────────────

def _nuclear_binding(composed: dict, params: Mapping[str, Any]) -> None:
    """Subtract the nuclear mass defect from an additively composed nucleus.

    Applies only when the spec names both proton- and neutron-like
    constituents (symbols come from params, so nothing is hardcoded) and the
    mass number is at least 2 -- a lone proton or a bare electron has no
    binding energy to remove.

    Binding energy comes from the semi-empirical mass formula already in the
    package (`utils.predictors.nuclear.semf_predictor`), so this refiner adds
    no new physics of its own; it only connects the SEMF to composition.

    Writes, alongside the corrected masses:
        AtomicNumber_Z, NeutronNumber_N, MassNumber_A
        BindingEnergy_MeV, BindingEnergyPerNucleon_MeV, MassDefect_amu
    """
    composition = composed.get("Composition")
    if not isinstance(composition, Mapping):
        return

    proton_syms = params.get("proton_symbols") or ()
    neutron_syms = params.get("neutron_symbols") or ()
    z_f = _count_of(composition, proton_syms)
    n_f = _count_of(composition, neutron_syms)
    if z_f <= 0:
        return
    Z, N = _as_int(z_f), _as_int(n_f)
    if Z is None or N is None:
        return  # fractional nucleon counts are not a nucleus
    A = Z + N
    if A < 2:
        return

    # Identity first, and unconditionally: these are facts about the spec, not
    # results of the binding calculation, and later refinements join on them.
    composed["AtomicNumber_Z"] = Z
    composed["NeutronNumber_N"] = N
    composed["MassNumber_A"] = A

    from periodica.utils.predictors.nuclear.semf_predictor import SEMFNuclearPredictor

    binding_mev = float(SEMFNuclearPredictor().calculate_binding_energy(Z, N))
    if binding_mev <= 0:
        # SEMF is not calibrated for the lightest nuclei and can return a
        # non-positive figure there. Reporting no binding energy is honest;
        # subtracting a negative one would inflate the mass.
        return

    mev_per_amu = float(params.get("mev_per_amu", 931.49410242))
    defect_amu = binding_mev / mev_per_amu

    composed["BindingEnergy_MeV"] = binding_mev
    composed["BindingEnergyPerNucleon_MeV"] = binding_mev / A
    composed["MassDefect_amu"] = defect_amu

    if isinstance(composed.get("Mass_amu"), (int, float)):
        composed["Mass_amu"] = float(composed["Mass_amu"]) - defect_amu
    if isinstance(composed.get("Mass_MeVc2"), (int, float)):
        composed["Mass_MeVc2"] = float(composed["Mass_MeVc2"]) - binding_mev


# ── mass_units ──────────────────────────────────────────────────────────

def _mass_units(composed: dict, params: Mapping[str, Any]) -> None:
    """Fill in mass expressed in other units from whichever one is known.

    Constituent JSON carries mass in amu and MeV/c2 but not kilograms, so
    additive composition produced ``Mass_kg: 0`` on every composed entry --
    a zero that reads as a real measurement. This derives the missing units
    instead of leaving them at zero, and keeps all three consistent after
    `nuclear_binding` has adjusted two of them.
    """
    kg_per_amu = float(params.get("kg_per_amu", 1.66053906660e-27))
    mev_per_amu = float(params.get("mev_per_amu", 931.49410242))

    amu = composed.get("Mass_amu")
    mev = composed.get("Mass_MeVc2")
    amu = float(amu) if isinstance(amu, (int, float)) else None
    mev = float(mev) if isinstance(mev, (int, float)) else None

    if amu is None and mev is not None and mev > 0:
        amu = mev / mev_per_amu
        composed["Mass_amu"] = amu
    if mev is None and amu is not None and amu > 0:
        composed["Mass_MeVc2"] = amu * mev_per_amu

    if amu is not None and amu > 0:
        # Only overwrite a missing or zero Mass_kg; a curated value stands.
        existing = composed.get("Mass_kg")
        if not isinstance(existing, (int, float)) or existing == 0:
            composed["Mass_kg"] = amu * kg_per_amu


# ── cross_reference ─────────────────────────────────────────────────────

def _cross_reference(composed: dict, params: Mapping[str, Any]) -> None:
    """Attach measured properties from a curated tier onto a composed entry.

    A composed atom knows its proton count but nothing measurable: no
    density, no melting point, no electronegativity. Those numbers exist in
    the curated ``elements`` tier, keyed by atomic number. This refiner joins
    the two, so ``Get({"P": 26, "N": 30, "E": 26})`` carries Iron's measured
    bulk properties instead of only its composition arithmetic.

    Params:
        tier          registry tier to search
        match_field   field on the composed entry holding the join value
        match_key     field on the candidate entry to match it against
        merge_into    key under which to place the copied properties
        copy_fields   explicit field list, or omit to copy every scalar
        skip_fields   fields never copied
        also_copy     {target_key_on_composed: source_field} for a few
                      identity fields worth promoting to the root

    Sources are recorded under ``_provenance`` so a consumer can tell a
    measured value from a composed one.
    """
    tier = params.get("tier")
    match_field = params.get("match_field")
    match_key = params.get("match_key")
    if not (tier and match_field and match_key):
        return
    value = composed.get(match_field)
    if value is None:
        return

    from periodica.get import _registry

    index = _registry().by_tier.get(str(tier))
    if index is None:
        return

    target = _as_int(value)
    match: Optional[dict] = None
    for entry in index.exact.values():
        candidate = _as_int(entry.get(match_key))
        if candidate is not None and candidate == target:
            match = entry
            break
    if match is None:
        return

    skip = {str(s) for s in (params.get("skip_fields") or ())}
    copy_fields = params.get("copy_fields")
    merge_into = str(params.get("merge_into") or "Properties")

    props: Dict[str, Any] = {}
    if copy_fields:
        for field in copy_fields:
            if field in match and field not in skip:
                props[str(field)] = match[field]
    else:
        for key, val in match.items():
            if key in skip or str(key).startswith("_"):
                continue
            if isinstance(val, (int, float, str)) and not isinstance(val, bool):
                props[str(key)] = val

    if props:
        bucket = composed.setdefault(merge_into, {})
        if isinstance(bucket, dict):
            # Never overwrite a value the composer or an earlier refiner set.
            for key, val in props.items():
                bucket.setdefault(key, val)

    for target_key, source_field in (params.get("also_copy") or {}).items():
        if source_field in match:
            composed.setdefault(str(target_key), match[source_field])

    provenance = composed.setdefault("_provenance", {})
    if isinstance(provenance, dict):
        provenance[merge_into] = {
            "tier": str(tier),
            "matched_on": f"{match_key}={target}",
            "entry": match.get("Name") or match.get("name"),
            "fields": sorted(props),
        }


register_refiner("nuclear_binding", _nuclear_binding)
register_refiner("mass_units", _mass_units)
register_refiner("cross_reference", _cross_reference)


__all__ = [
    "Refiner",
    "apply_refinements",
    "list_refiners",
    "register_refiner",
]
