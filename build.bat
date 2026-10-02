@echo off
REM ====== periodica/build.bat ======
REM copyright (c) 2025 Andrew Keith Watts. All rights reserved.
REM
REM This is the intellectual property of Andrew Keith Watts. Unauthorized
REM reproduction, distribution, or modification of this code, in whole or in
REM part, without the express written permission of Andrew Keith Watts is
REM strictly prohibited.
REM
REM For inquiries, please contact AndrewKWatts@Gmail.com
REM
REM Full build: Rust workspace -> Python extension -> tests.
REM Required by CLAUDE.md section 2 (Build & Distribution).
REM
REM Usage:
REM   build.bat            Format check, clippy, cargo test, editable install, pytest
REM   build.bat rust       Rust only (fast inner loop)
REM   build.bat py         Python only (assumes the extension is already built)
REM   build.bat bench      Criterion benchmarks
REM   build.bat release    Release wheel into dist\

setlocal enabledelayedexpansion
cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=all

if "%MODE%"=="rust"    goto :rust
if "%MODE%"=="py"      goto :py
if "%MODE%"=="bench"   goto :bench
if "%MODE%"=="release" goto :release
if "%MODE%"=="all"     goto :rust
echo [build] unknown mode "%MODE%"
exit /b 2

:rust
echo.
echo [build] === Rust: format check ===
cargo fmt --all --check
if errorlevel 1 (
    echo [build] FAILED: run "cargo fmt --all" to fix formatting.
    exit /b 1
)

echo.
echo [build] === Rust: clippy ===
REM periodica_core carries pre-existing lint debt, so warnings are not yet
REM denied workspace-wide. The new crates must stay clean.
cargo clippy -p periodica-mat --all-targets -- -D warnings
if errorlevel 1 exit /b 1
cargo clippy -p periodica-runtime --all-targets -- -D warnings
if errorlevel 1 exit /b 1
cargo clippy --workspace --all-targets
if errorlevel 1 exit /b 1

echo.
echo [build] === Rust: all features compile ===
REM cargo test builds default features only, so feature-gated modules such as
REM pyfacade\ (behind `python`) can break unnoticed. `check` compiles them all,
REM including `extension-module`, without linking.
cargo check --workspace --all-features
if errorlevel 1 exit /b 1

echo.
echo [build] === Rust: tests ===
cargo test --workspace
if errorlevel 1 exit /b 1

echo.
echo [build] === Rust: PyO3 binding tests (python feature, links libpython) ===
REM setlocal scopes PATH / PYTHONHOME to this step so the Python steps below
REM run against the untouched environment.
setlocal
call :pyo3_test_env
cargo test -p periodica_core --features python
if errorlevel 1 (
    endlocal
    exit /b 1
)
endlocal

if "%MODE%"=="rust" goto :done

:py
echo.
REM An editable install builds the extension via maturin (features from
REM [tool.maturin]) into src\periodica. `maturin develop` refuses to run
REM outside a virtualenv, so it is not used here.
echo [build] === Python: build extension (pip install -e .) ===
python -m pip install -e . --no-deps
if errorlevel 1 (
    echo [build] FAILED: editable install of the Rust extension.
    exit /b 1
)

echo.
echo [build] === Python: assert the Rust backend is actually live ===
python -c "import periodica, sys; ok = getattr(periodica, '_HAS_RUST', False); print('_HAS_RUST =', ok); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo [build] FAILED: extension built but periodica._HAS_RUST is False.
    exit /b 1
)

echo.
echo [build] === Python: tests ===
python -m pytest tests/ -q --tb=short -m "not slow and not gemini"
if errorlevel 1 exit /b 1
goto :done

:bench
echo.
echo [build] === Criterion benchmarks ===
cargo bench --workspace
if errorlevel 1 exit /b 1
goto :done

:release
echo.
echo [build] === Release wheel ===
cargo test --workspace --release
if errorlevel 1 exit /b 1
REM Features come from [tool.maturin] in pyproject.toml (extension-module).
python -m maturin build --release --out dist
if errorlevel 1 exit /b 1
echo [build] wheel written to dist\
goto :done

:pyo3_test_env
REM The binding tests embed an interpreter, so the test exe must load
REM python3X.dll from the interpreter's base prefix at run time. A Microsoft
REM Store Python keeps it under C:\Program Files\WindowsApps, from which an
REM unpackaged exe may not load a DLL (STATUS_ACCESS_DENIED, 0xc0000022), so
REM the DLLs are copied to target\pydll and PYTHONHOME points the embedded
REM interpreter back at the stdlib.
for /f "delims=" %%P in ('python -c "import sys; print(sys.base_prefix)"') do set "PYBASE=%%P"
set "PYDLL=%PYBASE%"
echo "%PYBASE%" | findstr /i /c:"WindowsApps" >nul
if errorlevel 1 goto :pyo3_test_env_path
if not exist "target\pydll" mkdir "target\pydll"
copy /y "%PYBASE%\python3*.dll" "target\pydll\" >nul
set "PYDLL=%CD%\target\pydll"
set "PYTHONHOME=%PYBASE%"
:pyo3_test_env_path
set "PATH=%PYDLL%;%PATH%"
goto :eof

:done
echo.
echo [build] OK
endlocal
exit /b 0
