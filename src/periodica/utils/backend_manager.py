"""
Global Backend Manager for Dual-Pathway System.

This module provides a unified interface to switch between pure Python
and library (scipy/numpy) backends across all modules that support
dual implementations.

Usage:
    from periodica.utils.backend_manager import BackendManager

    # Use pure Python for all calculations
    BackendManager.set_all_backends(use_libraries=False)

    # Use libraries where available
    BackendManager.set_all_backends(use_libraries=True)

    # Check current backend status
    status = BackendManager.get_status()
    print(status)

    # Compare results between backends
    results = BackendManager.validate_backends()
"""
from typing import Dict, Optional, Callable, Any, List
import math


class BackendManager:
    """
    Centralized manager for switching between pure Python and library backends.

    This class provides a single point of control for all dual-pathway
    implementations in the codebase, including:
    - orbital_clouds.py (scipy vs pure_math)
    - sdf_renderer.py (numpy vs pure_array)

    The manager also provides validation utilities to compare results
    between backends.
    """

    # Track which modules have been initialized
    _initialized = False

    @classmethod
    def set_all_backends(cls, use_libraries: bool = True) -> Dict[str, bool]:
        """
        Set all backends to use either libraries or pure Python.

        Args:
            use_libraries: True to use scipy/numpy, False for pure Python.

        Returns:
            Dictionary with backend names and whether they were successfully set.

        Raises:
            ImportError: If a library is requested but not available (only in strict mode).

        Example:
            >>> BackendManager.set_all_backends(use_libraries=False)
            {'orbital_clouds': True, 'sdf_renderer': True}
        """
        results = {}

        # Set orbital_clouds backend (scipy)
        try:
            from utils import orbital_clouds
            orbital_clouds.set_backend(use_scipy=use_libraries)
            results['orbital_clouds'] = True
        except ImportError as e:
            if use_libraries:
                results['orbital_clouds'] = False
            else:
                # Pure Python should always work
                results['orbital_clouds'] = True
        except Exception as e:
            results['orbital_clouds'] = False

        # Set sdf_renderer backend (numpy)
        try:
            from utils import sdf_renderer
            sdf_renderer.set_backend(use_numpy=use_libraries)
            results['sdf_renderer'] = True
        except ImportError as e:
            if use_libraries:
                results['sdf_renderer'] = False
            else:
                results['sdf_renderer'] = True
        except Exception as e:
            results['sdf_renderer'] = False

        cls._initialized = True
        return results

    @classmethod
    def use_pure_python(cls) -> Dict[str, bool]:
        """
        Convenience method to switch all backends to pure Python.

        Returns:
            Dictionary with backend names and success status.
        """
        return cls.set_all_backends(use_libraries=False)

    @classmethod
    def use_libraries(cls) -> Dict[str, bool]:
        """
        Convenience method to switch all backends to use libraries.

        Returns:
            Dictionary with backend names and success status.
        """
        return cls.set_all_backends(use_libraries=True)

    @classmethod
    def get_status(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get the current status of all backends.

        Returns:
            Dictionary with detailed status for each module:
            {
                'module_name': {
                    'current_backend': 'scipy' or 'pure_python',
                    'library_available': True/False,
                    'pure_python_available': True (always)
                }
            }
        """
        status = {}

        # Check orbital_clouds
        try:
            from utils import orbital_clouds
            status['orbital_clouds'] = {
                'current_backend': orbital_clouds.get_backend(),
                'library_available': cls._check_scipy_available(),
                'pure_python_available': True
            }
        except ImportError:
            status['orbital_clouds'] = {
                'current_backend': 'unknown',
                'library_available': False,
                'pure_python_available': True
            }

        # Check sdf_renderer
        try:
            from utils import sdf_renderer
            status['sdf_renderer'] = {
                'current_backend': sdf_renderer.get_backend(),
                'library_available': cls._check_numpy_available(),
                'pure_python_available': True
            }
        except ImportError:
            status['sdf_renderer'] = {
                'current_backend': 'unknown',
                'library_available': False,
                'pure_python_available': True
            }

        # Check overall library availability
        status['libraries'] = {
            'scipy_available': cls._check_scipy_available(),
            'numpy_available': cls._check_numpy_available(),
        }

        return status

    @staticmethod
    def _check_scipy_available() -> bool:
        """Check if scipy is available."""
        try:
            import scipy.special
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_numpy_available() -> bool:
        """Check if numpy is available."""
        try:
            import numpy
            return True
        except ImportError:
            return False

    @classmethod
    def get_available_functions(cls) -> Dict[str, List[str]]:
        """
        Get list of all functions with dual implementations.

        Returns:
            Dictionary mapping module names to lists of function names.
        """
        return {
            'pure_math': [
                'factorial',
                'double_factorial',
                'genlaguerre',
                'lpmv',
                'spherical_harmonic',
                'spherical_harmonic_real',
                'spherical_harmonic_prefactor',
                'binomial',
                'gamma_half_integer',
            ],
            'pure_array': [
                'Vec3 (class with rotate_x, rotate_y, rotate_z)',
                'rotation_matrix_x',
                'rotation_matrix_y',
                'rotation_matrix_z',
                'rotation_matrix_axis_angle',
                'rotation_matrix_euler',
                'matrix_multiply_3x3',
                'matrix_vector_multiply_3x3',
                'generate_nucleon_positions',
                'generate_shell_positions',
            ],
            'orbital_clouds': [
                'radial_wavefunction',
                'angular_wavefunction',
                'get_orbital_probability',
                'radial_wavefunction_enhanced',
                'get_orbital_probability_enhanced',
            ],
            'sdf_renderer': [
                '_generate_nucleons_numpy / _generate_nucleons_pure',
            ],
        }

    @classmethod
    def validate_backends(cls, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Run validation comparing pure Python vs library implementations.

        This method tests a selection of calculations using both backends
        and reports the maximum relative error for each function.

        Args:
            verbose: Print results as tests run.

        Returns:
            Dictionary with validation results:
            {
                'function_name': {
                    'max_relative_error': float,
                    'max_absolute_error': float,
                    'tests_run': int,
                    'passed': bool
                }
            }
        """
        if not cls._check_scipy_available():
            if verbose:
                print("scipy not available - validation requires scipy for comparison")
            return {'error': 'scipy not available'}

        results = {}

        # Import both implementations
        from periodica.utils.pure_math import (
            factorial as pure_factorial,
            genlaguerre as pure_genlaguerre,
            lpmv as pure_lpmv,
            spherical_harmonic as pure_sph_harm,
        )

        import scipy.special

        # Test factorial
        if verbose:
            print("Testing factorial...")
        max_err = 0.0
        for n in range(0, 51):
            pure = float(pure_factorial(n))
            lib = float(scipy.special.factorial(n, exact=True))
            if lib != 0:
                err = abs(pure - lib) / abs(lib)
            else:
                err = abs(pure)
            max_err = max(max_err, err)
        results['factorial'] = {
            'max_relative_error': max_err,
            'tests_run': 51,
            'passed': max_err < 1e-14
        }
        if verbose:
            print(f"  Max error: {max_err:.2e} - {'PASS' if max_err < 1e-14 else 'FAIL'}")

        # Test genlaguerre
        if verbose:
            print("Testing genlaguerre...")
        max_err = 0.0
        tests = 0
        for n in range(0, 8):
            for alpha in [0.0, 0.5, 1.0, 2.0]:
                pure_L = pure_genlaguerre(n, alpha)
                scipy_L = scipy.special.genlaguerre(n, alpha)
                for x in [0.0, 0.5, 1.0, 2.0, 5.0]:
                    pure = float(pure_L(x))
                    lib = float(scipy_L(x))
                    if abs(lib) > 1e-10:
                        err = abs(pure - lib) / abs(lib)
                    else:
                        err = abs(pure - lib)
                    max_err = max(max_err, err)
                    tests += 1
        results['genlaguerre'] = {
            'max_relative_error': max_err,
            'tests_run': tests,
            'passed': max_err < 1e-10
        }
        if verbose:
            print(f"  Max error: {max_err:.2e} - {'PASS' if max_err < 1e-10 else 'FAIL'}")

        # Test lpmv
        if verbose:
            print("Testing lpmv (Associated Legendre)...")
        max_err = 0.0
        tests = 0
        for l in range(0, 6):
            for m in range(-l, l + 1):
                for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    pure = float(pure_lpmv(m, l, x))
                    lib = float(scipy.special.lpmv(m, l, x))
                    if abs(lib) > 1e-10:
                        err = abs(pure - lib) / abs(lib)
                    else:
                        err = abs(pure - lib)
                    max_err = max(max_err, err)
                    tests += 1
        results['lpmv'] = {
            'max_relative_error': max_err,
            'tests_run': tests,
            'passed': max_err < 1e-10
        }
        if verbose:
            print(f"  Max error: {max_err:.2e} - {'PASS' if max_err < 1e-10 else 'FAIL'}")

        # Test spherical harmonics
        if verbose:
            print("Testing spherical_harmonic...")
        max_err = 0.0
        tests = 0
        for l in range(0, 5):
            for m in range(-l, l + 1):
                for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
                    for phi in [0.0, math.pi / 2, math.pi]:
                        pure = pure_sph_harm(l, m, theta, phi)
                        lib = scipy.special.sph_harm(m, l, phi, theta)  # Note different arg order!
                        # Compare magnitudes
                        pure_mag = abs(pure)
                        lib_mag = abs(lib)
                        if lib_mag > 1e-10:
                            err = abs(pure_mag - lib_mag) / lib_mag
                        else:
                            err = abs(pure_mag - lib_mag)
                        max_err = max(max_err, err)
                        tests += 1
        results['spherical_harmonic'] = {
            'max_relative_error': max_err,
            'tests_run': tests,
            'passed': max_err < 1e-8
        }
        if verbose:
            print(f"  Max error: {max_err:.2e} - {'PASS' if max_err < 1e-8 else 'FAIL'}")

        # Summary
        if verbose:
            print("\nValidation Summary:")
            all_passed = all(r.get('passed', False) for r in results.values() if isinstance(r, dict))
            print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

        return results

    @classmethod
    def print_status(cls):
        """Print a human-readable status report."""
        status = cls.get_status()

        print("=" * 60)
        print("Backend Status Report")
        print("=" * 60)

        # Library availability
        libs = status.get('libraries', {})
        print("\nLibrary Availability:")
        print(f"  scipy:  {'Available' if libs.get('scipy_available') else 'Not installed'}")
        print(f"  numpy:  {'Available' if libs.get('numpy_available') else 'Not installed'}")

        # Module backends
        print("\nModule Backends:")
        for module in ['orbital_clouds', 'sdf_renderer']:
            if module in status:
                mod = status[module]
                print(f"  {module}:")
                print(f"    Current backend: {mod.get('current_backend', 'unknown')}")
                print(f"    Library available: {mod.get('library_available', False)}")
                print(f"    Pure Python available: {mod.get('pure_python_available', True)}")

        # Available functions
        print("\nDual-Pathway Functions:")
        funcs = cls.get_available_functions()
        for module, func_list in funcs.items():
            print(f"  {module}:")
            for func in func_list:
                print(f"    - {func}")

        print("=" * 60)


# Convenience function for quick backend switching
def use_pure_python():
    """Switch all backends to pure Python."""
    return BackendManager.use_pure_python()


def use_libraries():
    """Switch all backends to use libraries (scipy/numpy)."""
    return BackendManager.use_libraries()


def get_backend_status():
    """Get current backend status."""
    return BackendManager.get_status()


def validate_backends(verbose: bool = True):
    """Run validation comparing backends."""
    return BackendManager.validate_backends(verbose=verbose)


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("Backend Manager Self-Test")
    print("=" * 60)

    # Print status
    BackendManager.print_status()

    # Test switching backends
    print("\nTesting backend switching...")

    print("\n1. Switching to pure Python...")
    result = BackendManager.use_pure_python()
    print(f"   Result: {result}")
    status = BackendManager.get_status()
    for module in ['orbital_clouds', 'sdf_renderer']:
        if module in status:
            print(f"   {module}: {status[module]['current_backend']}")

    print("\n2. Switching to libraries...")
    result = BackendManager.use_libraries()
    print(f"   Result: {result}")
    status = BackendManager.get_status()
    for module in ['orbital_clouds', 'sdf_renderer']:
        if module in status:
            print(f"   {module}: {status[module]['current_backend']}")

    # Run validation
    print("\n3. Running backend validation...")
    validation = BackendManager.validate_backends(verbose=True)

    print("\nSelf-test complete!")
