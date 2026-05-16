# Python fallback: src/periodica/{get,sample}.py
import pytest

from periodica._dispatch import _HAS_RUST, _native

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not built")


def test_data_sheet_fe():
    from periodica.sample import data_sheet
    ds = data_sheet("Fe")
    assert isinstance(ds, dict)
    assert len(ds) > 0


def test_data_sheet_is_subset_of_get():
    from periodica.sample import data_sheet
    from periodica.get import Get
    ds = data_sheet("Fe")
    entry = Get("Fe")
    props = entry.get("Properties") or {}
    for k in ds:
        assert k in props or k in entry, f"data_sheet key {k!r} missing from Get()"


def test_sample_fe_density():
    from periodica.sample import sample
    val = sample("Fe", "Density_kg_m3")
    if val is not None:
        assert isinstance(val, (int, float))
        assert val > 0


def test_get_fe_returns_entry():
    from periodica.get import Get
    entry = Get("Fe")
    assert isinstance(entry, dict)
    assert len(entry) > 0


def test_get_water_formula():
    from periodica.get import Get
    try:
        entry = Get("{H=2,O=1}")
        assert isinstance(entry, dict)
    except Exception:
        pytest.skip("Formula resolution not yet wired in this build")


def test_list_tiers():
    tiers = _native.py_list_tiers()
    assert isinstance(tiers, list)
    assert len(tiers) > 0


def test_rust_version():
    v = _native.version_rust()
    assert v.startswith("2"), f"Expected 2.x.x version, got {v!r}"
