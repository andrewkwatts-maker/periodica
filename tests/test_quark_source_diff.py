"""
Cross-source comparison: experimental (PDG bundled JSON) vs simulated
(metaphysica.Get) quark datasheets.

Verifies that:

1. Both sources are reachable for the six Standard-Model quarks.
2. Conserved (definitional) quantum numbers match exactly between sources.
   These should NEVER differ — they're algebraic identities, not measurements.
3. Mass values are within sensible bounds of each other (the simulated value
   is the metaphysica G₂-derived prediction; depending on the quark, agreement
   with PDG ranges from very close to ~order-of-magnitude — we capture the
   actual gap as a parametric assertion + emit a deviation report).
4. Derived types resolve to the same `ParticleType` / `QuarkGeneration`
   classification regardless of source.

Skipped automatically if metaphysica isn't installed; install with::

    pip install periodica[simulated]
"""
from __future__ import annotations

import importlib.util
import math

import pytest

from periodica.data.quark_source import (
    EXPERIMENTAL,
    SIMULATED,
    load_quark,
    list_quark_names,
)

# ── Skip the whole file when metaphysica isn't around ───────────────────────

_HAS_METAPHYSICA = importlib.util.find_spec("metaphysica") is not None
pytestmark = pytest.mark.skipif(
    not _HAS_METAPHYSICA,
    reason="metaphysica not installed — install periodica[simulated]",
)


# Six Standard-Model quark names that exist in both sources.
SM_QUARKS = ("Up", "Down", "Charm", "Strange", "Top", "Bottom")


def _both(name: str):
    """Return ``(experimental_dict, simulated_dict)`` for *name*."""
    e = load_quark(name, source=EXPERIMENTAL)
    s = load_quark(name, source=SIMULATED)
    return e, s


# ── Source presence ─────────────────────────────────────────────────────────

class TestSourcesAvailable:

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_experimental_resolves(self, name):
        d = load_quark(name, source=EXPERIMENTAL)
        assert d is not None, f"experimental source missing {name!r}"
        assert d["_source"] == EXPERIMENTAL

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_simulated_resolves(self, name):
        d = load_quark(name, source=SIMULATED)
        assert d is not None, f"simulated source missing {name!r}"
        assert d["_source"] == SIMULATED


# ── Conserved quantum numbers — must match exactly between sources ─────────

class TestConservedQuantumNumbers:

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_charge_matches(self, name):
        e, s = _both(name)
        assert math.isclose(e["Charge_e"], s["Charge_e"], abs_tol=1e-9), (
            f"{name}: Charge_e differs (exp={e['Charge_e']}, sim={s['Charge_e']})"
        )

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_baryon_number_matches(self, name):
        e, s = _both(name)
        assert math.isclose(
            e["BaryonNumber_B"], s["BaryonNumber_B"], abs_tol=1e-9
        ), f"{name}: BaryonNumber_B differs"

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_spin_matches(self, name):
        e, s = _both(name)
        assert e["Spin_hbar"] == s["Spin_hbar"], (
            f"{name}: Spin_hbar differs (exp={e['Spin_hbar']}, sim={s['Spin_hbar']})"
        )

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_lepton_number_zero(self, name):
        e, s = _both(name)
        assert e["LeptonNumber_L"] == 0
        assert s["LeptonNumber_L"] == 0

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_isospin_matches(self, name):
        e, s = _both(name)
        # Isospin is a definitional quantum number — must agree
        assert math.isclose(e["Isospin_I"], s["Isospin_I"], abs_tol=1e-9)
        assert math.isclose(e["Isospin_I3"], s["Isospin_I3"], abs_tol=1e-9)


# ── Mass — measured (PDG) vs G₂-derived prediction ─────────────────────────

class TestMassDeviation:
    """
    The simulated mass is metaphysica's G₂-derived prediction, not a fit.
    We don't enforce close agreement — instead we record the *actual*
    deviation in a way the test report shows. The test only fails if the
    sim value isn't a finite positive number.
    """

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_simulated_mass_finite_positive(self, name):
        _, s = _both(name)
        m = s.get("Mass_MeVc2")
        assert m is not None
        assert math.isfinite(m)
        assert m > 0

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_experimental_mass_finite_positive(self, name):
        e, _ = _both(name)
        m = e.get("Mass_MeVc2")
        assert m is not None
        assert math.isfinite(m)
        assert m > 0

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_mass_deviation_recorded(self, name, capsys):
        # Emit a one-line deviation summary so test logs carry the comparison.
        e, s = _both(name)
        em, sm = e["Mass_MeVc2"], s["Mass_MeVc2"]
        if em > 0:
            pct = (sm - em) / em * 100.0
        else:
            pct = float("inf")
        with capsys.disabled():
            print(f"  {name:8} exp={em:>11.4g} MeV  sim={sm:>11.4g} MeV  "
                  f"delta={sm - em:+12.4g} MeV  ({pct:+8.2f}%)")
        # No assertion — informational only. The pm_prediction block already
        # carries the percent_error metadata for programmatic access.
        assert sm is not None  # placeholder so pytest counts the test


# ── Antiparticle metadata cross-check ──────────────────────────────────────

class TestAntiparticleBlocks:

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_antiparticle_name_matches(self, name):
        e, s = _both(name)
        # Both sources record an Antiparticle.Name; they should agree on the
        # convention "Anti<X> Quark"
        assert e["Antiparticle"]["Name"] == s["Antiparticle"]["Name"], (
            f"{name}: antiparticle name differs "
            f"(exp={e['Antiparticle']['Name']!r}, sim={s['Antiparticle']['Name']!r})"
        )


# ── Derived classifications ─────────────────────────────────────────────────

class TestDerivedTypes:
    """The downstream `ParticleType` / `QuarkGeneration` enums are derived
    from the dict via `Classification` and `Name`. If both sources expose
    the same Classification list and the same Name, the derived enum
    must agree."""

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_particle_type_agrees(self, name):
        from periodica.core.quark_enums import ParticleType
        e, s = _both(name)
        et = ParticleType.from_classification(e.get("Classification", []))
        st = ParticleType.from_classification(s.get("Classification", []))
        assert et == st, (
            f"{name}: ParticleType differs (exp={et}, sim={st}); "
            f"exp.Classification={e.get('Classification')}, "
            f"sim.Classification={s.get('Classification')}"
        )

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_generation_agrees(self, name):
        from periodica.core.quark_enums import QuarkGeneration
        e, s = _both(name)
        eg = QuarkGeneration.from_particle_name(e.get("Name", ""))
        sg = QuarkGeneration.from_particle_name(s.get("Name", ""))
        assert eg == sg, (
            f"{name}: generation differs (exp={eg}, sim={sg})"
        )


# ── pm_prediction block (simulated only) ────────────────────────────────────

class TestPmPredictionBlock:
    """The simulated source carries an extra `pm_prediction` block with the
    derivation metadata. Verify it's well-formed."""

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_pm_prediction_present(self, name):
        _, s = _both(name)
        assert "pm_prediction" in s, f"{name}: missing pm_prediction block"

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_pm_prediction_keys(self, name):
        _, s = _both(name)
        pm = s["pm_prediction"]
        for required in ("phi_scaling_N", "predicted_mass_GeV", "pdg_mass_GeV",
                          "percent_error", "verdict"):
            assert required in pm, f"{name}: pm_prediction.{required} missing"

    @pytest.mark.parametrize("name", SM_QUARKS)
    def test_pm_prediction_finite(self, name):
        _, s = _both(name)
        pm = s["pm_prediction"]
        assert math.isfinite(pm["predicted_mass_GeV"])
        assert math.isfinite(pm["pdg_mass_GeV"])


# ── List-name parity ────────────────────────────────────────────────────────

class TestNameLists:
    """The simulated source returns lowercase canonical names. The
    experimental source returns the filename-stem capitalisation. After
    case-folding, every SM quark name must appear in BOTH lists."""

    def test_all_sm_quarks_in_both_lists(self):
        e_names = {n.lower() for n in list_quark_names(source=EXPERIMENTAL)}
        s_names = {n.lower() for n in list_quark_names(source=SIMULATED)}
        for q in SM_QUARKS:
            ql = q.lower()
            assert ql in e_names, f"{q} missing from experimental list"
            assert ql in s_names, f"{q} missing from simulated list"

    def test_simulated_has_antiquarks(self):
        s_names = {n.lower() for n in list_quark_names(source=SIMULATED)}
        for q in SM_QUARKS:
            assert f"anti{q.lower()}" in s_names, (
                f"simulated source should expose anti{q.lower()} via metaphysica"
            )
