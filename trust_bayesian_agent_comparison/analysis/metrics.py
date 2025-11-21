"""
Metrics and analysis utilities for simulation results.
"""
import pandas as pd
import numpy as np
from typing import Optional

from ..config import DECISION_THRESHOLD, PAYOFF_MATRIX


def get_payoff(player1_strategy: int, player2_strategy: int, player_id: int) -> float:
    """Extract payoff from matrix."""
    return float(PAYOFF_MATRIX[int(player1_strategy), int(player2_strategy), int(player_id)])


def agent_coop_rate(df: pd.DataFrame) -> float:
    """Calculate fraction of rounds the agent chose to cooperate."""
    return float(df["agent_action"].mean())


def mutual_coop_rate(df: pd.DataFrame) -> float:
    """Calculate fraction of rounds with mutual cooperation."""
    return float(((df["agent_action"] == 1) & (df["partner_action"] == 1)).mean())


def betrayal_rate(df: pd.DataFrame) -> float:
    """Calculate fraction of rounds where agent cooperated but partner defected."""
    return float(((df["agent_action"] == 1) & (df["partner_action"] == 0)).mean())


def final_decision(df: pd.DataFrame) -> int:
    """Return final decision (0 or 1)."""
    return int(df["agent_action"].iloc[-1])


def calculate_payoffs(df: pd.DataFrame) -> pd.DataFrame:
    """Add payoff columns to simulation results."""
    df = df.copy()
    
    agent_payoffs = []
    partner_payoffs = []
    
    for _, row in df.iterrows():
        agent_choice = int(row["agent_action"])
        partner_choice = int(row["partner_action"])
        
        agent_payoffs.append(get_payoff(agent_choice, partner_choice, 0))
        partner_payoffs.append(get_payoff(agent_choice, partner_choice, 1))
    
    df["agent_payoff"] = agent_payoffs
    df["partner_payoff"] = partner_payoffs
    
    return df


def total_payoff(df: pd.DataFrame) -> float:
    """Calculate total agent payoff across all rounds."""
    if "agent_payoff" not in df.columns:
        df = calculate_payoffs(df)
    return float(df["agent_payoff"].sum())


def time_to_threshold(
    df: pd.DataFrame,
    p_star: Optional[float] = None,
    direction: Optional[str] = None
) -> Optional[float]:
    """
    Find first round when E[p] crosses threshold.
    
    Args:
        df: Simulation results
        p_star: Threshold value (default: DECISION_THRESHOLD)
        direction: 'up' (E[p] >= threshold), 'down' (E[p] <= threshold), or None (any crossing)
        
    Returns:
        Round number (1-indexed) or None if no crossing
    """
    if p_star is None:
        p_star = DECISION_THRESHOLD
    
    # Use agent_belief column
    E = df["agent_belief"].to_numpy()
    
    if len(E) == 0:
        return None
    
    if direction == "up":
        idx = np.where(E >= p_star)[0]
        return int(df["round"].iloc[idx[0]]) if idx.size > 0 else None
    elif direction == "down":
        idx = np.where(E <= p_star)[0]
        return int(df["round"].iloc[idx[0]]) if idx.size > 0 else None
    else:
        # Any crossing
        sign0 = np.sign(E[0] - p_star)
        for i in range(1, len(E)):
            if np.sign(E[i] - p_star) != sign0:
                return int(df["round"].iloc[i])
        return None


def compute_strategy_statistics(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive statistics for a single simulation.
    
    Returns dictionary with:
        - Cooperation rates
        - Payoff statistics
        - Final beliefs
        - Threshold metrics
    """
    df_with_payoffs = calculate_payoffs(df)
    
    stats = {
        "agent_coop_rate": agent_coop_rate(df),
        "mutual_coop_rate": mutual_coop_rate(df),
        "betrayal_rate": betrayal_rate(df),
        "total_agent_payoff": total_payoff(df_with_payoffs),
        "avg_agent_payoff": df_with_payoffs["agent_payoff"].mean(),
        "total_partner_payoff": df_with_payoffs["partner_payoff"].sum(),
        "avg_partner_payoff": df_with_payoffs["partner_payoff"].mean(),
        "final_decision": final_decision(df),
        "final_belief": float(df["agent_belief"].iloc[-1]),
    }
    
    return stats
