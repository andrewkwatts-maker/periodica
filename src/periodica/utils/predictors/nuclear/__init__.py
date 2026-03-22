"""
Nuclear Predictors Package
==========================

This package provides nuclear physics predictors for calculating nuclear properties
such as binding energy, nuclear radius, and stability.

Available Predictors:
    - SEMFNuclearPredictor: Uses the Semi-Empirical Mass Formula (SEMF) for
      predicting nuclear binding energy and related properties.

Usage:
    from periodica.utils.predictors.nuclear import SEMFNuclearPredictor
    from periodica.utils.predictors.base import NuclearInput

    predictor = SEMFNuclearPredictor()
    input_data = NuclearInput(Z=26, N=30)  # Iron-56
    result = predictor.predict(input_data)
    print(f"Binding energy: {result.binding_energy_mev:.2f} MeV")
"""

from .semf_predictor import SEMFNuclearPredictor

__all__ = ['SEMFNuclearPredictor']
