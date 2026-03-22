"""
Material Property Transform Module
==================================

Provides spatial sampling of material properties using:
- Fourier transforms for periodic variations
- Wavelet transforms for multi-scale variations
- Karhunen-Loeve expansion for stochastic fields
- Temperature and pressure dependencies via equations of state

Example:
    from periodica.utils.transforms import MaterialSampler

    sampler = MaterialSampler(material_json)
    props = sampler.sample(x=0.01, y=0.02, z=0.005, T=573.15, P=1e8)
"""

from .fourier_field import FourierFieldEvaluator, FourierCoefficient3D
from .wavelet_field import WaveletFieldEvaluator, WaveletConfig
from .stochastic_field import StochasticFieldEvaluator, CovarianceModel
from .thermo_pressure import (
    BirchMurnaghanEOS,
    TemperatureDependence,
    ThermoPressureSampler
)
from .material_sampler import MaterialSampler

__all__ = [
    'MaterialSampler',
    'FourierFieldEvaluator',
    'FourierCoefficient3D',
    'WaveletFieldEvaluator',
    'WaveletConfig',
    'StochasticFieldEvaluator',
    'CovarianceModel',
    'BirchMurnaghanEOS',
    'TemperatureDependence',
    'ThermoPressureSampler',
]
