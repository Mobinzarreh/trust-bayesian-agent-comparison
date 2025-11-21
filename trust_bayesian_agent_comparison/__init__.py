"""Trust-based Bayesian agent comparison framework."""

from .config import (
    PAYOFF_MATRIX,
    stag_indifference_threshold,
    get_payoff,
    DECISION_THRESHOLD,
)
from .agents import FocalAgent, BayesianFocalAgent

__version__ = "0.1.0"

__all__ = [
    "FocalAgent",
    "BayesianFocalAgent",
    "PAYOFF_MATRIX",
    "stag_indifference_threshold",
    "get_payoff",
    "DECISION_THRESHOLD",
]
