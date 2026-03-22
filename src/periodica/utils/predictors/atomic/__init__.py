"""
Atomic Predictors Package
=========================

This package contains predictor implementations for atomic properties.

Available predictors:
- SlaterAtomicPredictor: Uses Slater's rules and Aufbau principle
"""

from .slater_predictor import SlaterAtomicPredictor

__all__ = ['SlaterAtomicPredictor']
