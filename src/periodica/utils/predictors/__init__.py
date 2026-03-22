"""SOLID-compliant predictor framework for Periodics."""

from .protocols import (
    PredictorProtocol,
    NuclearPredictorProtocol,
    AtomicPredictorProtocol,
    HadronPredictorProtocol,
    PredictionConfidence,
)
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
from .chain import DerivationChain, ChainResult

# Import predictors to trigger registration
from .nuclear import SEMFNuclearPredictor
from .atomic import SlaterAtomicPredictor
from .hadron import ConstituentQuarkPredictor

# Legacy compatibility
from .compat import PredictionEngine, NuclearDerivation, AtomicDerivation

__all__ = [
    'PredictorProtocol', 'PredictionConfidence',
    'PredictorRegistry', 'register_predictor',
    'BasePredictor', 'BaseNuclearPredictor', 'BaseAtomicPredictor', 'BaseHadronPredictor',
    'DerivationChain', 'ChainResult',
    'NuclearInput', 'NuclearResult', 'AtomicInput', 'AtomicResult',
    'SEMFNuclearPredictor', 'SlaterAtomicPredictor', 'ConstituentQuarkPredictor',
    'PredictionEngine', 'NuclearDerivation', 'AtomicDerivation',
]
