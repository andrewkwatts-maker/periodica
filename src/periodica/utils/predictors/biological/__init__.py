"""
Biological predictors for the Periodics application.

This module provides predictors for biological entities from amino acids
through to biological materials, implementing Henderson-Hasselbalch equations,
Chou-Fasman secondary structure prediction, and other biochemical calculations.
"""

from .amino_acid_predictor import AminoAcidPredictor

__all__ = ['AminoAcidPredictor']
