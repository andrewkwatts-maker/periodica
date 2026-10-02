#!/usr/bin/env bash
#====== periodica/build.sh ======#
# copyright (c) 2025 Andrew Keith Watts. All rights reserved.
#
# This is the intellectual property of Andrew Keith Watts. Unauthorized
# reproduction, distribution, or modification of this code, in whole or in
# part, without the express written permission of Andrew Keith Watts is
# strictly prohibited.
#
# For inquiries, please contact AndrewKWatts@Gmail.com
#
# Full build: Rust workspace -> Python extension -> tests.
# POSIX counterpart of build.bat, required by CLAUDE.md section 2.
#
# Usage:
#   ./build.sh            Format check, clippy, cargo test, editable install, pytest
#   ./build.sh rust       Rust only (fast inner loop)
#   ./build.sh py         Python only (assumes the extension is already built)
#   ./build.sh bench      Criterion benchmarks
#   ./build.sh release    Release wheel into dist/

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-all}"
say() { printf '\n[build] === %s ===\n' "$*"; }

build_rust() {
    say "Rust: format check"
    if ! cargo fmt --all --check; then
        echo '[build] FAILED: run "cargo fmt --all" to fix formatting.' >&2
        exit 1
    fi

    say "Rust: clippy"
    # periodica_core carries pre-existing lint debt, so warnings are not yet
    # denied workspace-wide. The new crates must stay clean.
    cargo clippy -p periodica-mat --all-targets -- -D warnings
    cargo clippy -p periodica-runtime --all-targets -- -D warnings
    cargo clippy --workspace --all-targets

    # cargo test builds default features only, so feature-gated modules such
    # as pyfacade/ (behind `python`) can break unnoticed. `check` compiles
    # them all, including `extension-module`, without linking.
    say "Rust: all features compile"
    cargo check --workspace --all-features

    say "Rust: tests"
    cargo test --workspace

    say "Rust: PyO3 binding tests (python feature, links libpython)"
    # Subshell: the loader variables must not leak into the Python steps.
    ( pyo3_test_env && cargo test -p periodica_core --features python )
}

# The binding tests embed an interpreter, so the test binary must find
# libpython at run time. On Linux/macOS the shared library sits in
# `<base_prefix>/lib`. Under Git Bash on Windows it is python3X.dll in the base
# prefix; a Microsoft Store Python keeps that under C:\Program Files\WindowsApps,
# from which an unpackaged exe may not load a DLL (STATUS_ACCESS_DENIED,
# 0xc0000022), so the DLLs are copied to target/pydll and PYTHONHOME points the
# embedded interpreter back at the stdlib.
pyo3_test_env() {
    local base
    base="$(python -c 'import sys; print(sys.base_prefix)')"
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            local dlldir
            dlldir="$(cygpath -u "$base")"
            case "$base" in
                *WindowsApps*)
                    mkdir -p target/pydll
                    cp "$dlldir"/python3*.dll target/pydll/
                    dlldir="$PWD/target/pydll"
                    export PYTHONHOME="$base"
                    ;;
            esac
            export PATH="$dlldir:$PATH"
            ;;
        Darwin)
            export DYLD_LIBRARY_PATH="$base/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            ;;
        *)
            export LD_LIBRARY_PATH="$base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
    esac
}

build_py() {
    # An editable install builds the extension via maturin (features from
    # [tool.maturin]) into src/periodica. `maturin develop` is avoided because
    # it refuses to run outside a virtualenv.
    say "Python: build extension (pip install -e .)"
    if ! python -m pip install -e . --no-deps; then
        echo '[build] FAILED: editable install of the Rust extension.' >&2
        exit 1
    fi

    say "Python: assert the Rust backend is actually live"
    python - <<'PY'
import sys
import periodica
ok = getattr(periodica, "_HAS_RUST", False)
print("_HAS_RUST =", ok)
if not ok:
    print("extension built but periodica._HAS_RUST is False", file=sys.stderr)
    sys.exit(1)
PY

    say "Python: tests"
    python -m pytest tests/ -q --tb=short -m "not slow and not gemini"
}

case "$MODE" in
    rust)
        build_rust
        ;;
    py)
        build_py
        ;;
    bench)
        say "Criterion benchmarks"
        cargo bench --workspace
        ;;
    release)
        say "Release wheel"
        cargo test --workspace --release
        python -m maturin build --release --out dist  # features: [tool.maturin]
        echo "[build] wheel written to dist/"
        ;;
    all)
        build_rust
        build_py
        ;;
    *)
        echo "[build] unknown mode \"$MODE\"" >&2
        exit 2
        ;;
esac

printf '\n[build] OK\n'
