"""
Hadron Predictors Package.

This package contains predictor implementations for calculating hadron
properties from quark compositions.

Available predictors:
    - ConstituentQuarkPredictor: Uses the constituent quark model with
      experimental values for known hadrons and QCD-corrected calculations
      for unknown combinations.
"""

from .constituent_predictor import ConstituentQuarkPredictor

__all__ = ['ConstituentQuarkPredictor']
