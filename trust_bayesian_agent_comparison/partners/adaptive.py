"""Adaptive and deceptive strategy partners."""

import numpy as np
from .base import BasePartner


class StrategicCheaterPartner(BasePartner):
    """
    Strategic cheater with pattern: C^n, D, C^n, D, ...
    
    Cooperates for n rounds, defects once, repeats.
    Tests agent's forgiveness and memory.
    """
    
    def __init__(self, coop_length: int = 5):
        """
        Initialize strategic cheater.
        
        Args:
            coop_length: Number of cooperation rounds between defections
        """
        super().__init__()
        self.coop_length = coop_length
    
    def decide(self, round_num: int) -> int:
        """Cooperate for n rounds, defect once, repeat."""
        position = round_num % (self.coop_length + 1)
        return 0 if position == self.coop_length else 1


class AdaptiveStrategicPartner(BasePartner):
    """
    Adaptive strategic partner that learns agent's threshold.
    
    Starts cooperating, occasionally tests with defection,
    adapts defection frequency based on agent's responses.
    """
    
    def __init__(self, exploration_rate: float = 0.1):
        """
        Initialize adaptive strategic partner.
        
        Args:
            exploration_rate: Probability of testing with defection
        """
        super().__init__()
        self.exploration_rate = exploration_rate
        self.cooperation_count = 0
        self.defection_count = 0
    
    def decide(self, round_num: int) -> int:
        """Mostly cooperate, occasionally test with defection."""
        if round_num < 5:
            return 1  # Build trust initially
        
        # Explore with probability
        if np.random.rand() < self.exploration_rate:
            return 0
        return 1
    
    def update(self, agent_action: int):
        """Track agent's responses."""
        super().update(agent_action)
        if agent_action == 1:
            self.cooperation_count += 1
        else:
            self.defection_count += 1


class BayesianDeceptivePartner(BasePartner):
    """
    Bayesian deceptive partner that models agent's trust.
    
    Cooperates to build trust, then exploits when belief is high.
    Uses simple heuristic: defect after k consecutive agent cooperations.
    """
    
    def __init__(self, exploit_threshold: int = 7):
        """
        Initialize Bayesian deceptive partner.
        
        Args:
            exploit_threshold: Consecutive agent cooperations before exploiting
        """
        super().__init__()
        self.exploit_threshold = exploit_threshold
        self.consecutive_coop = 0
    
    def decide(self, round_num: int) -> int:
        """Exploit when agent shows high trust."""
        if self.consecutive_coop >= self.exploit_threshold:
            self.consecutive_coop = 0  # Reset after exploitation
            return 0
        return 1
    
    def update(self, agent_action: int):
        """Track consecutive agent cooperations."""
        super().update(agent_action)
        if agent_action == 1:
            self.consecutive_coop += 1
        else:
            self.consecutive_coop = 0


class ExploitativePartner(BasePartner):
    """
    Exploitative partner that defects when agent cooperates.
    
    Anti-tit-for-tat: does opposite of agent's previous action.
    """
    
    def decide(self, round_num: int) -> int:
        """Do opposite of agent's previous action."""
        if round_num == 0:
            return 1  # Start cooperating to lure
        
        # Do opposite
        return 1 - self.history[-1]
