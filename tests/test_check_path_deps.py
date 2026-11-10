import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "scripts" / "check_path_deps.py"
_spec = importlib.util.spec_from_file_location("check_path_deps", _p)
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)


def _manifest(tmp_path, dep):
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    mf = root / "Cargo.toml"
    mf.write_text(f'[dependencies]\nx = {{ path = "{dep}" }}\n', encoding="utf-8")
    return root, mf


def test_flags_unlisted_sibling(tmp_path):
    root, mf = _manifest(tmp_path, "../Themelios/x")
    assert [s for _, s in cpd.escaping_paths(mf, root)] == ["Themelios"]


def test_allows_arithma_and_inner_paths(tmp_path):
    root, mf = _manifest(tmp_path, "../Arithma/rust/arithma_core")
    assert list(cpd.escaping_paths(mf, root)) == []
    root2, mf2 = _manifest(tmp_path / "b", "crates/inner")
    assert list(cpd.escaping_paths(mf2, root2)) == []


def test_repo_is_clean():
    assert cpd.main() == 0
