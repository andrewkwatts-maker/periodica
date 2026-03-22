"""
Alloy Property Predictors

This package provides predictors for calculating alloy properties from
constituent element compositions.

Available Predictors:
- RuleOfMixturesPredictor: Uses weighted averaging schemes appropriate for
  each property type (inverse rule for density, linear for melting point, etc.)
"""

from .rule_of_mixtures import RuleOfMixturesPredictor

__all__ = ['RuleOfMixturesPredictor']
