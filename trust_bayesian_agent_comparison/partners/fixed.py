"""Fixed strategy partners."""

import numpy as np
import random

from .base import BasePartner


class AlwaysCooperatePartner:
    """Always cooperates regardless of agent's actions."""
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Always return cooperate."""
        return 1
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


# Alias for compatibility
AlwaysCollaboratePartner = AlwaysCooperatePartner


class AlwaysDefectPartner:
    """Always defects regardless of agent's actions."""
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Always return defect."""
        return 0
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class RandomPartner:
    """Random partner with cooperation probability p."""
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        self.p = p
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Randomly choose cooperation or defection."""
        return 1 if random.random() < self.p else 0
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class PeriodicCheaterPartner:
    """
    Cooperate for `cycle_length` rounds, then defect for `cheat_duration` rounds, repeat.
    
    Creates predictable betrayal cycles to test agent's pattern recognition.
    """
    
    def __init__(self, cycle_length: int = 6, cheat_duration: int = 2):
        """
        Initialize periodic cheater.
        
        Args:
            cycle_length: Number of cooperation rounds in each cycle
            cheat_duration: Number of defection rounds in each cycle
        """
        self.cycle_length = cycle_length
        self.cheat_duration = cheat_duration
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate for cycle_length rounds, defect for cheat_duration rounds."""
        return 1 if (round_num % (self.cycle_length + self.cheat_duration)) < self.cycle_length else 0
    
    def observe(self, focal_agent_action: int):
        pass  # No learning
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class SingleCyclePartner(BasePartner):
    """
    Cooperate for the first fraction of rounds, then defect thereafter.
    
    Tests agent's response to sudden, permanent strategy change.
    Cooperates for first 40% of total rounds, then permanently defects.
    """
    
    def __init__(self, num_rounds: int = 100, cooperate_fraction: float = 0.4):
        """
        Initialize single cycle partner.
        
        Args:
            num_rounds: Total number of rounds in the simulation
            cooperate_fraction: Fraction of rounds to cooperate before switching (default: 0.4 = 40%)
        """
        super().__init__()
        self.num_rounds = num_rounds
        self.cooperate_fraction = cooperate_fraction
        self.cooperate_rounds = int(num_rounds * cooperate_fraction)
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate for first cooperate_fraction of rounds, then always defect."""
        return 1 if round_num < self.cooperate_rounds else 0


class ProbabilisticPartner:
    """
    Cooperate with a fixed probability.
    """
    
    def __init__(self, cooperate_prob: float = 0.7):
        """
        Initialize probabilistic partner.
        
        Args:
            cooperate_prob: Probability of cooperation (constant)
        """
        if not 0 <= cooperate_prob <= 1:
            raise ValueError("cooperate_prob must be in [0,1]")
        self.cooperate_prob = cooperate_prob
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate with fixed probability."""
        return 1 if random.random() < self.cooperate_prob else 0
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class GradualDeteriorationPartner:
    """
    Starts fully cooperative, then gradually reduces cooperation probability linearly over time.
    cooperation_prob(t) = 1.0 - (deterioration_rate * t / num_rounds)
    """
    
    def __init__(self, deterioration_rate: float = 0.8, num_rounds: int = 70):
        """
        Initialize gradual deterioration partner.
        
        Args:
            deterioration_rate: How much to deteriorate (0.8 = drops to 20% by end)
            num_rounds: Total rounds for full deterioration
        """
        self.deterioration_rate = deterioration_rate  # how much to deteriorate
        self.num_rounds = num_rounds
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Linear deterioration: starts at 1.0, decreases over time."""
        # Linear deterioration: starts at 1.0, decreases over time
        coop_prob = 1.0 - (self.deterioration_rate * round_num / self.num_rounds)
        coop_prob = max(0.0, min(1.0, coop_prob))  # clip to [0,1]
        return 1 if random.random() < coop_prob else 0
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass
