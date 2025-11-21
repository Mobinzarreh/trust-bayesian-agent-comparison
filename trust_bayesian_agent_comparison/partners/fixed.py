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
