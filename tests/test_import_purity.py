"""Test that the periodica library has zero UI framework dependencies.

Uses subprocess to avoid pytest plugins (pytest-qt) polluting sys.modules.
"""
import subprocess
import sys
import pytest


IMPORT_CHECK_SCRIPT = """
import sys
{import_stmt}
ui_frameworks = {{"PySide6", "PyQt5", "PyQt6", "kivy", "kivymd"}}
leaked = ui_frameworks & set(sys.modules.keys())
if leaked:
    print(f"LEAKED: {{leaked}}")
    sys.exit(1)
else:
    print("CLEAN")
    sys.exit(0)
"""


def _assert_import_pure(import_stmt: str):
    """Run an import in a clean subprocess and assert no UI frameworks leak."""
    script = IMPORT_CHECK_SCRIPT.format(import_stmt=import_stmt)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (
        f"UI framework leaked after: {import_stmt}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


class TestImportPurity:
    """Verify no UI framework is imported by the periodica library."""

    def test_top_level_import(self):
        _assert_import_pure("import periodica")

    def test_core_enums(self):
        _assert_import_pure("from periodica.core import pt_enums, molecule_enums, quark_enums")

    def test_constants(self):
        _assert_import_pure("from periodica.constants import PropertyType")

    def test_utils_physics(self):
        _assert_import_pure("from periodica.utils.physics_calculator import PhysicsConstants, AtomCalculator")

    def test_utils_pure_math(self):
        _assert_import_pure("from periodica.utils.pure_math import factorial")

    def test_utils_color_math(self):
        _assert_import_pure("from periodica.utils.color_math import wavelength_to_rgb, ev_to_wavelength")

    def test_utils_sdf_core(self):
        _assert_import_pure("from periodica.utils.sdf_core import sdf_sphere, nucleus_geometry")

    def test_layout_math(self):
        _assert_import_pure("from periodica.layout_math.element_table import TablePositioner")

    def test_simulation_schema(self):
        _assert_import_pure("from periodica.utils.simulation_schema import QuarkSimulationData")
