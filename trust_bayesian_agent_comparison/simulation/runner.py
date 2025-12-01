"""Core simulation runner for agent-partner interactions.

Reproducibility:
- Each simulation receives an isolated RNG (numpy.random.Generator) to ensure
  deterministic and independent behavior across parallel runs.
- The RNG is passed to agents/partners that support it, or used to seed global
  RNGs for backward compatibility.

RNG Isolation:
- Each simulation creates its own numpy.random.Generator from the seed.
- This ensures that Agent1 and Agent2 simulations are completely independent,
  even if they consume different numbers of random draws.
"""

import pandas as pd
import numpy as np
import random
from typing import Any, Optional
from ..config import NUM_ROUNDS, DEFAULT_SEED, PAYOFF_MATRIX


def run_agent_simulation(
    agent: Any,
    partner: Any,
    num_rounds: int = NUM_ROUNDS,
    seed: int = DEFAULT_SEED,
    rng: Optional[np.random.Generator] = None,
    notebook_compatible_seeding: bool = False,
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
        seed: Random seed for reproducibility (used if rng not provided)
        rng: Optional numpy random Generator for isolated randomness
        notebook_compatible_seeding: If True, use direct seeding like notebook
            (np.random.seed(seed)) for validation. If False, use isolated RNG
            for proper Monte Carlo independence.
        
    Returns:
        DataFrame with columns:
            - round: Round number (0-indexed)
            - agent_belief: Agent's belief before decision
            - agent_action: Agent's action (1=cooperate, 0=defect)
            - partner_action: Partner's action
            - agent_payoff: Agent's payoff this round
            - partner_payoff: Partner's payoff this round
    """
    if notebook_compatible_seeding:
        # Notebook-style: direct seeding for exact reproducibility with notebook
        np.random.seed(seed)
        random.seed(seed)
        rng = None  # Don't use isolated RNG
    else:
        # Production-style: isolated RNG for Monte Carlo independence
        # Create isolated RNG if not provided
        if rng is None:
            rng = np.random.default_rng(seed)
        
        # Seed global RNGs for backward compatibility with code using np.random or random
        # Use rng to generate seeds to maintain isolation
        np.random.seed(int(rng.integers(0, 2**31)))
        random.seed(int(rng.integers(0, 2**31)))
    
    # Pass RNG to agent/partner if they support it
    if hasattr(agent, 'set_rng'):
        agent.set_rng(rng)
    if hasattr(partner, 'set_rng'):
        partner.set_rng(rng)
    
    records = []
    last_agent_action = None  # Track for reactive partners
    
    for r in range(num_rounds):
        # 1. Record agent's current belief (before decision)
        belief = agent.get_belief()
        
        # Get additional state variables if available
        trust_level = getattr(agent, 't', None)
        signal = getattr(agent, 'x', None)
        alpha = getattr(agent, 'alpha', None)
        beta = getattr(agent, 'beta', None)
        
        # 2. Both decide simultaneously
        # Agent decides based on its belief
        agent_action = agent.decide()
        
        # Partner decides - pass last_agent_action for reactive partners
        partner_action = partner.decide(r, last_agent_action)
        
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
        
        # 6. Track last agent action for next round (reactive partners)
        last_agent_action = agent_action
    
    return pd.DataFrame(records)


def run_paired_simulation(
    agent1: Any,
    agent2: Any,
    partner_factory: Any,
    num_rounds: int = NUM_ROUNDS,
    seed: int = DEFAULT_SEED,
    notebook_compatible_seeding: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run paired simulations with identical conditions per agent.

    Creates two fresh partner instances via `partner_factory()` so that each
    agent faces an identical but isolated opponent. Each agent gets its own
    independent RNG stream derived from the same seed, ensuring:
    1. Both agents face statistically identical conditions
    2. Different random draw counts don't affect the other agent's simulation

    Args:
        agent1: First agent instance
        agent2: Second agent instance
        partner_factory: Callable that creates fresh partner instances (picklable)
        num_rounds: Number of rounds per simulation
        seed: Random seed for reproducibility
        notebook_compatible_seeding: If True, use direct seeding like notebook.
            Note: In notebook mode, agent2 will NOT have identical RNG conditions
            as agent1, which matches notebook behavior but sacrifices isolation.

    Returns:
        Tuple of (df1, df2) where:
            - df1: Results for agent1
            - df2: Results for agent2
    """
    if notebook_compatible_seeding:
        # Notebook-style: sequential simulation with shared RNG state
        # This matches notebook behavior but agent2 sees different RNG state than agent1
        partner1 = partner_factory()
        df1 = run_agent_simulation(
            agent1, partner1, num_rounds, seed, 
            notebook_compatible_seeding=True
        )
        
        # Agent2 runs with same seed but RNG state has advanced from agent1's simulation
        # To match notebook behavior where each simulation re-seeds:
        partner2 = partner_factory()
        df2 = run_agent_simulation(
            agent2, partner2, num_rounds, seed,
            notebook_compatible_seeding=True
        )
    else:
        # Production-style: isolated RNG for each agent
        # Create independent RNGs for each agent's simulation
        # Using different but deterministic sub-seeds ensures isolation
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1_000_000)  # Offset to ensure independence
        
        # Run agent1 vs partner with isolated RNG
        partner1 = partner_factory()
        df1 = run_agent_simulation(agent1, partner1, num_rounds, seed, rng=rng1)
        
        # Run agent2 vs identical partner with its own isolated RNG
        partner2 = partner_factory()
        df2 = run_agent_simulation(agent2, partner2, num_rounds, seed, rng=rng2)
    
    return df1, df2
