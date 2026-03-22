"""Backward compatibility layer for existing code."""

from .registry import PredictorRegistry
from .chain import DerivationChain
from .base import NuclearInput, AtomicInput

class PredictionEngine:
    """Legacy wrapper matching original API."""
    VERSION = "2.0.0"

    def __init__(self):
        self._chain = DerivationChain()

    def predict(self, Z: int, N: int = None):
        return self._chain.predict(Z, N)

    def predict_unknown(self, Z: int, N: int = None):
        result = self.predict(Z, N)
        if Z > 118:
            result.confidence['overall'] *= 0.5
        return result

class NuclearDerivation:
    """Legacy wrapper for nuclear predictors."""
    def __init__(self):
        self._predictor = PredictorRegistry().get('nuclear', 'default')

    def calculate(self, Z: int, N: int):
        return self._predictor.predict(NuclearInput(Z=Z, N=N))

class AtomicDerivation:
    """Legacy wrapper for atomic predictors."""
    def __init__(self):
        self._predictor = PredictorRegistry().get('atomic', 'default')

    def calculate(self, Z: int, nuclear_mass_mev=None):
        return self._predictor.predict(AtomicInput(Z=Z, nuclear_mass_mev=nuclear_mass_mev))
