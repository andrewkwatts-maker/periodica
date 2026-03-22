"""Utility functions for calculations and helpers."""

from periodica.utils.physics_calculator import (
    PhysicsConstants,
    AtomCalculator,
    SubatomicCalculator,
    MoleculeCalculator
)

# Simulation schema for data-driven physics
from periodica.utils.simulation_schema import (
    # Enums
    ParticleType, SpinType, LatticeType,
    # Constants
    SimulationConstants,
    # Base dataclasses
    Position3D, Momentum3D, QuantumState, FormFactors,
    # Particle dataclasses
    QuarkSimulationData, HadronSimulationData, AtomSimulationData,
    MoleculeSimulationData, AlloySimulationData,
    # Propagation functions
    propagate_quark_to_hadron, propagate_hadrons_to_atom,
    propagate_atoms_to_molecule, propagate_elements_to_alloy,
    # Utility converters
    dict_to_quark, dict_to_atom,
)

# Pure Python math utilities (no external dependencies)
from periodica.utils.pure_math import (
    factorial,
    double_factorial,
    genlaguerre,
    lpmv,
    GeneralizedLaguerre,
    # Spherical harmonics
    spherical_harmonic,
    spherical_harmonic_real,
    spherical_harmonic_prefactor,
    # Other utilities
    binomial,
    gamma_half_integer,
)

from periodica.utils.pure_array import (
    # Constants
    pi,
    # Math functions
    sqrt, cos, sin, acos, atan2,
    # Random utilities
    random_uniform, random_seed,
    # Vector class
    Vec3,
    # Nucleon generation
    generate_nucleon_positions,
    generate_shell_positions,
    # Utility functions
    lerp, clamp, smoothstep, distance,
    # 3D Rotation matrices
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotation_matrix_axis_angle,
    rotation_matrix_euler,
    matrix_multiply_3x3,
    matrix_vector_multiply_3x3,
    apply_rotation_matrix,
)

# Backend manager for dual-pathway system
from periodica.utils.backend_manager import (
    BackendManager,
    use_pure_python,
    use_libraries,
    get_backend_status,
    validate_backends,
)

__all__ = [
    'PhysicsConstants',
    'AtomCalculator',
    'SubatomicCalculator',
    'MoleculeCalculator',
    # pure_math exports
    'factorial', 'double_factorial', 'genlaguerre', 'lpmv', 'GeneralizedLaguerre',
    'spherical_harmonic', 'spherical_harmonic_real', 'spherical_harmonic_prefactor',
    'binomial', 'gamma_half_integer',
    # pure_array exports
    'pi',
    'sqrt', 'cos', 'sin', 'acos', 'atan2',
    'random_uniform', 'random_seed',
    'Vec3',
    'generate_nucleon_positions',
    'generate_shell_positions',
    'lerp', 'clamp', 'smoothstep', 'distance',
    # 3D rotation matrices
    'rotation_matrix_x', 'rotation_matrix_y', 'rotation_matrix_z',
    'rotation_matrix_axis_angle', 'rotation_matrix_euler',
    'matrix_multiply_3x3', 'matrix_vector_multiply_3x3', 'apply_rotation_matrix',
    # Backend manager
    'BackendManager', 'use_pure_python', 'use_libraries',
    'get_backend_status', 'validate_backends',
    # Simulation schema exports
    'ParticleType', 'SpinType', 'LatticeType',
    'SimulationConstants',
    'Position3D', 'Momentum3D', 'QuantumState', 'FormFactors',
    'QuarkSimulationData', 'HadronSimulationData', 'AtomSimulationData',
    'MoleculeSimulationData', 'AlloySimulationData',
    'propagate_quark_to_hadron', 'propagate_hadrons_to_atom',
    'propagate_atoms_to_molecule', 'propagate_elements_to_alloy',
    'dict_to_quark', 'dict_to_atom',
]
