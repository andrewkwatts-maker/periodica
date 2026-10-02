"""Semi-empirical predictor framework (Python, pending the native port).

The surviving models -- SEMF, Slater, constituent quark, VSEPR, Hückel,
Miedema/rule of mixtures, protein / amino-acid / nucleic-acid -- are the M6
port targets in the migration census. Importing this package registers the
nuclear, atomic and hadron predictors with ``PredictorRegistry``.
"""

from .registry import PredictorRegistry, register_predictor
from .base import (
    BasePredictor,
    BaseNuclearPredictor,
    BaseAtomicPredictor,
    BaseHadronPredictor,
    NuclearInput,
    NuclearResult,
    AtomicInput,
    AtomicResult,
    HadronInput,
    HadronResult,
)

# Import predictors to trigger registration
from .nuclear import SEMFNuclearPredictor
from .atomic import SlaterAtomicPredictor
from .hadron import ConstituentQuarkPredictor

__all__ = [
    'PredictorRegistry', 'register_predictor',
    'BasePredictor', 'BaseNuclearPredictor', 'BaseAtomicPredictor', 'BaseHadronPredictor',
    'NuclearInput', 'NuclearResult', 'AtomicInput', 'AtomicResult',
    'HadronInput', 'HadronResult',
    'SEMFNuclearPredictor', 'SlaterAtomicPredictor', 'ConstituentQuarkPredictor',
]
