"""Tests for the `python -m periodica` CLI and the new tiers it builds."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from periodica import Get, Scope, list_tiers, reload_registry
from periodica.__main__ import main as cli_main
from periodica.scripts._runner import (
    _topo_sort,
    _read_meta,
    build_all,
    default_inputs,
)


@pytest.fixture(autouse=True)
def _refresh_registry():
    reload_registry()
    yield
    reload_registry()


# ─────────────────────────────────────────────────────────────────────────
# Default-input discovery + dependency ordering
# ─────────────────────────────────────────────────────────────────────────

class TestDefaultInputsAndOrdering:
    def test_default_inputs_present(self):
        names = [p.stem for p in default_inputs()]
        for required in ("hadrons", "periodic_table", "ions", "isotopes", "molecules"):
            assert required in names, f"missing default input: {required}"

    def test_topo_sort_atoms_before_molecules(self):
        inputs = default_inputs()
        ordered = _topo_sort(inputs)
        atoms = next(p for p in ordered if p.stem == "periodic_table")
        molecules = next(p for p in ordered if p.stem == "molecules")
        assert ordered.index(atoms) < ordered.index(molecules)

    def test_molecules_input_declares_atoms_dependency(self):
        meta = _read_meta(Path(default_inputs()[0]).parent / "molecules.json")
        assert "atoms" in meta["requires"]


# ─────────────────────────────────────────────────────────────────────────
# CLI subcommands (no subprocess; call `main(argv)` directly)
# ─────────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_inputs_subcommand_runs(self, capsys):
        rc = cli_main(["inputs"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "molecules.json" in out

    def test_list_subcommand_summary(self, capsys):
        rc = cli_main(["list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "fundamentals" in out
        assert "subatomic" in out

    def test_list_specific_tier(self, capsys):
        # `atoms` exists post-build; build first if not.
        if "atoms" not in list_tiers():
            cli_main(["build", "-q"])
            capsys.readouterr()
        rc = cli_main(["list", "atoms"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "H" in out and "He" in out and "Fe" in out

    def test_show_subcommand_returns_entry_json(self, capsys):
        if "atoms" not in list_tiers():
            cli_main(["build", "-q"])
            capsys.readouterr()
        rc = cli_main(["show", "H"])
        out = capsys.readouterr().out
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["Symbol"] == "H"
        assert parsed["Composition"] == {"P": 1, "N": 0, "E": 1}

    def test_show_unknown_returns_nonzero(self, capsys):
        rc = cli_main(["show", "Imaginary_Element_QQQ"])
        assert rc != 0

    def test_build_specific_script_only(self, tmp_path, capsys, monkeypatch):
        # Build only isotopes, confirm only that tier got entries.
        cli_main(["clear", "-y", "isotopes"])
        capsys.readouterr()
        rc = cli_main(["build", "isotopes", "-q"])
        capsys.readouterr()
        assert rc == 0
        reload_registry()
        # H-1 should be retrievable.
        h1 = Get(Scope.Atom, "H-1") if "isotopes" not in list_tiers() else Get("H-1")
        assert h1["Composition"] == {"P": 1, "N": 0, "E": 1}

    def test_clear_specific_tier(self, capsys):
        # Make sure ions exist, then clear them.
        if "ions" not in list_tiers():
            cli_main(["build", "ions", "-q"])
            capsys.readouterr()
        rc = cli_main(["clear", "-y", "ions"])
        capsys.readouterr()
        assert rc == 0
        reload_registry()
        assert "ions" not in list_tiers()

    def test_run_custom_input_file(self, tmp_path, capsys):
        custom = tmp_path / "custom.json"
        custom.write_text(json.dumps({
            "from": ["subatomic", "fundamentals"],
            "tier": "test_custom",
            "rows": [
                {"name": "Test_H", "spec": {"P": 1, "N": 0, "E": 1}},
            ],
        }))
        # Need to direct Save() to a temp tier folder so we don't pollute.
        # The CLI uses default derived/ - acceptable for this test since we
        # immediately clear afterward.
        rc = cli_main(["run", str(custom), "-q"])
        capsys.readouterr()
        assert rc == 0
        reload_registry()
        assert Get("Test_H")["Composition"] == {"P": 1, "N": 0, "E": 1}
        # cleanup
        cli_main(["clear", "-y", "test_custom"])
        capsys.readouterr()


# ─────────────────────────────────────────────────────────────────────────
# New tier sanity checks against composition rules
# ─────────────────────────────────────────────────────────────────────────

class TestNewTiers:
    def test_isotopes_h2_has_one_neutron(self):
        if "isotopes" not in list_tiers():
            cli_main(["build", "isotopes", "-q"])
        h2 = Get("H-2")
        assert h2["Composition"] == {"P": 1, "N": 1, "E": 1}

    def test_ions_charge_calculation(self):
        if "ions" not in list_tiers():
            cli_main(["build", "ions", "-q"])
        # Ca2+ has 20 protons, 18 electrons => +2 net charge.
        ca2 = Get("Ca2+")
        assert ca2["Charge_e"] == 2
        # H+ has 1 proton, 0 electrons => +1 charge.
        hp = Get("H+")
        assert hp["Charge_e"] == 1
        # Cl- has 17 protons, 18 electrons => -1 charge.
        clm = Get("Cl-")
        assert clm["Charge_e"] == -1

    def test_isotope_and_atom_coexist(self):
        # H-1 and H both exist in different tiers and don't shadow each other.
        # Atomic 'H' (atoms tier) and isotopic 'H-1' (isotopes tier).
        if "isotopes" not in list_tiers() or "atoms" not in list_tiers():
            cli_main(["build", "-q"])
        h_atom = Get(Scope.Atom, "H")
        h_iso = Get("H-1")
        # Same composition, different names.
        assert h_atom["Composition"] == h_iso["Composition"]
