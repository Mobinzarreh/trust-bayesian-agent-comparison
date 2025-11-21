"""Analysis module for trust-bayesian agent comparison."""
from .sensitivity import SensitivityAnalysisManager
from .monte_carlo import MonteCarloManager
from .metrics import (
    agent_coop_rate,
    mutual_coop_rate,
    betrayal_rate,
    final_decision,
    calculate_payoffs,
    total_payoff,
    time_to_threshold,
    compute_strategy_statistics
)

__all__ = [
    'SensitivityAnalysisManager',
    'MonteCarloManager',
    'agent_coop_rate',
    'mutual_coop_rate',
    'betrayal_rate',
    'final_decision',
    'calculate_payoffs',
    'total_payoff',
    'time_to_threshold',
    'compute_strategy_statistics',
]
