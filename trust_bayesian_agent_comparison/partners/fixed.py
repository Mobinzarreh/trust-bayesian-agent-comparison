"""Fixed strategy partners."""

import numpy as np
from .base import BasePartner


class AlwaysCooperatePartner(BasePartner):
    """Always cooperates regardless of agent's actions."""
    
    def decide(self, round_num: int) -> int:
        """Always return cooperate."""
        return 1


class AlwaysDefectPartner(BasePartner):
    """Always defects regardless of agent's actions."""
    
    def decide(self, round_num: int) -> int:
        """Always return defect."""
        return 0


class RandomPartner(BasePartner):
    """Cooperates with fixed probability p."""
    
    def __init__(self, p: float = 0.5):
        """
        Initialize random partner.
        
        Args:
            p: Probability of cooperation
        """
        super().__init__()
        self.p = p
    
    def decide(self, round_num: int) -> int:
        """Cooperate with probability p."""
        return 1 if np.random.rand() < self.p else 0


class PeriodicCheaterPartner(BasePartner):
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
        super().__init__()
        self.cycle_length = cycle_length
        self.cheat_duration = cheat_duration
    
    def decide(self, round_num: int) -> int:
        """Cooperate for cycle_length rounds, defect for cheat_duration rounds."""
        position = round_num % (self.cycle_length + self.cheat_duration)
        return 1 if position < self.cycle_length else 0


class SingleCyclePartner(BasePartner):
    """
    Cooperate for the first `cooperate_rounds` rounds, then defect thereafter.
    
    Tests agent's response to sudden, permanent strategy change.
    """
    
    def __init__(self, cooperate_rounds: int = 30):
        """
        Initialize single cycle partner.
        
        Args:
            cooperate_rounds: Number of rounds to cooperate before switching to permanent defection
        """
        super().__init__()
        self.cooperate_rounds = cooperate_rounds
    
    def decide(self, round_num: int) -> int:
        """Cooperate until threshold, then always defect."""
        return 1 if round_num < self.cooperate_rounds else 0


class ProbabilisticPartner(BasePartner):
    """
    Partner with time-varying cooperation probability.
    
    Can be configured for gradual deterioration, improvement, or custom patterns.
    """
    
    def __init__(self, p_start: float = 1.0, p_end: float = 0.2, num_rounds: int = 70):
        """
        Initialize probabilistic partner.
        
        Args:
            p_start: Initial cooperation probability
            p_end: Final cooperation probability
            num_rounds: Total number of rounds for interpolation
        """
        super().__init__()
        self.p_start = p_start
        self.p_end = p_end
        self.num_rounds = num_rounds
    
    def decide(self, round_num: int) -> int:
        """Linearly interpolate cooperation probability over time."""
        t = min(round_num / self.num_rounds, 1.0)
        coop_prob = self.p_start + (self.p_end - self.p_start) * t
        return 1 if np.random.rand() < coop_prob else 0


class GradualDeteriorationPartner(BasePartner):
    """
    Starts fully cooperative, then gradually reduces cooperation probability linearly.
    
    cooperation_prob(t) = 1.0 - (deterioration_rate * t / num_rounds)
    """
    
    def __init__(self, deterioration_rate: float = 0.8, num_rounds: int = 70):
        """
        Initialize gradual deterioration partner.
        
        Args:
            deterioration_rate: How much to deteriorate (0.8 = drops to 20% by end)
            num_rounds: Total rounds for full deterioration
        """
        super().__init__()
        self.deterioration_rate = deterioration_rate
        self.num_rounds = num_rounds
    
    def decide(self, round_num: int) -> int:
        """Linear deterioration: starts at 1.0, decreases over time."""
        coop_prob = 1.0 - (self.deterioration_rate * round_num / self.num_rounds)
        coop_prob = max(0.0, min(1.0, coop_prob))  # clip to [0,1]
        return 1 if np.random.rand() < coop_prob else 0
