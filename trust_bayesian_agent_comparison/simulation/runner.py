"""Core simulation runner for agent-partner interactions.

Reproducibility:
- Seeds both NumPy and Python's `random` per simulation call to ensure
    deterministic behavior when agents/partners use either RNG.

Note:
- Both agent and partner consume from the same global RNG streams. If the two
    agents consume different numbers of random draws, their partner's draws could
    differ across runs even with identical seeds. For strict isolation, consider
    refactoring to pass dedicated RNG objects. For now, we reset the RNGs at the
    start of each simulation to provide consistent conditions within a run.
"""

import pandas as pd
import numpy as np
import random
from typing import Any
from ..config import NUM_ROUNDS, DEFAULT_SEED, PAYOFF_MATRIX


def run_agent_simulation(
    agent: Any,
    partner: Any,
    num_rounds: int = NUM_ROUNDS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Run simulation between agent and partner.
    
    Timeline per round:
    1. Record agent's current belief/state
    2. Both players decide simultaneously
    3. Actions executed, outcomes recorded
    4. Both players update based on observed actions
    
    Args:
        agent: Agent instance with decide() and update() methods
        partner: Partner instance with decide() and update() methods
        num_rounds: Number of interaction rounds
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns:
            - round: Round number (0-indexed)
            - agent_belief: Agent's belief before decision
            - agent_action: Agent's action (1=cooperate, 0=defect)
            - partner_action: Partner's action
            - agent_payoff: Agent's payoff this round
            - partner_payoff: Partner's payoff this round
    """
    # Seed both RNGs for reproducibility across NumPy and Python random users
    np.random.seed(seed)
    random.seed(seed)
    
    records = []
    
    for r in range(num_rounds):
        # 1. Record agent's current belief (before decision)
        belief = agent.get_belief()
        
        # Get additional state variables if available
        trust_level = getattr(agent, 't', None)
        signal = getattr(agent, 'x', None)
        alpha = getattr(agent, 'alpha', None)
        beta = getattr(agent, 'beta', None)
        
        # 2. Both decide simultaneously
        agent_action = agent.decide()
        partner_action = partner.decide(r)
        
        # 3. Calculate payoffs
        agent_payoff = PAYOFF_MATRIX[agent_action, partner_action, 0]
        partner_payoff = PAYOFF_MATRIX[agent_action, partner_action, 1]
        
        # 4. Record round data with all available state
        record = {
            'round': r,
            'agent_belief': belief,
            'agent_action': agent_action,
            'partner_action': partner_action,
            'agent_payoff': agent_payoff,
            'partner_payoff': partner_payoff,
        }
        
        # Add optional state variables
        if trust_level is not None:
            record['trust_level'] = trust_level
        if signal is not None:
            record['signal'] = signal
        if alpha is not None:
            record['alpha'] = alpha
        if beta is not None:
            record['beta'] = beta
            
        records.append(record)
        
        # 5. Update both players
        agent.update(partner_action)
        partner.update(agent_action)
    
    return pd.DataFrame(records)


def run_paired_simulation(
    agent1: Any,
    agent2: Any,
    partner_factory: Any,
    num_rounds: int = NUM_ROUNDS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run paired simulations with identical conditions per agent.

    Creates two fresh partner instances via `partner_factory()` so that each
    agent faces an identical but isolated opponent. Each simulation run
    re-seeds RNGs to the same `seed` to provide comparable stochastic
    conditions for both agents.

    Args:
        agent1: First agent instance
        agent2: Second agent instance
        partner_factory: Callable that creates fresh partner instances (picklable)
        num_rounds: Number of rounds per simulation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (df1, df2) where:
            - df1: Results for agent1
            - df2: Results for agent2
    """
    # Run agent1 vs partner
    partner1 = partner_factory()
    df1 = run_agent_simulation(agent1, partner1, num_rounds, seed)
    
    # Run agent2 vs identical partner (same seed)
    partner2 = partner_factory()
    df2 = run_agent_simulation(agent2, partner2, num_rounds, seed)
    
    return df1, df2
