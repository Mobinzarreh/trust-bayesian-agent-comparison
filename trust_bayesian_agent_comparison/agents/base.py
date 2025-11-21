"""Base agent class with shared decision logic."""

import numpy as np
from ..config import PAYOFF_MATRIX, STOCHASTIC, INVERSE_TEMPERATURE


class BaseAgent:
    """Base agent with shared decision-making logic."""
    
    def __init__(self, stochastic: bool = STOCHASTIC, inv_temp: float = INVERSE_TEMPERATURE):
        """
        Initialize base agent.
        
        Args:
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
        """
        self.name = self.__class__.__name__
        self.stochastic = stochastic
        self.inv_temp = inv_temp
    
    def expected_p(self) -> float:
        """
        Get expected probability that partner will cooperate.
        
        Returns:
            float: Probability that partner will cooperate
        """
        raise NotImplementedError("Subclasses must implement expected_p()")
    
    def get_belief(self) -> float:
        """
        Alias for expected_p() for backwards compatibility.
        
        Returns:
            float: Probability that partner will cooperate
        """
        return self.expected_p()
    
    def decide(self) -> int:
        """
        Make cooperation decision using logit/softmax rule or deterministic threshold.
        
        Computes expected utilities and chooses action:
        - EV(defect) = (1-p)*payoff[0,0] + p*payoff[0,1]
        - EV(cooperate) = (1-p)*payoff[1,0] + p*payoff[1,1]
        - Δ = EV(cooperate) - EV(defect)
        
        If stochastic: P(cooperate) = 1 / (1 + exp(-β*Δ))
        If deterministic: cooperate if Δ > 0
        
        Returns:
            int: 1 for cooperate, 0 for defect
        """
        p = self.expected_p()
        
        # Expected values from payoff matrix [agent_action, partner_action, player_id]
        EV_defect = (1 - p) * PAYOFF_MATRIX[0, 0, 0] + p * PAYOFF_MATRIX[0, 1, 0]
        EV_cooperate = (1 - p) * PAYOFF_MATRIX[1, 0, 0] + p * PAYOFF_MATRIX[1, 1, 0]
        
        delta = EV_cooperate - EV_defect
        
        if not self.stochastic:
            # Deterministic threshold rule
            return 1 if delta > 0 else 0
        
        # Stochastic logit/softmax choice
        prob_cooperate = 1.0 / (1.0 + np.exp(-self.inv_temp * delta))
        return 1 if np.random.random() < prob_cooperate else 0
    
    def update(self, partner_choice: int):
        """
        Update internal state based on observed partner choice.
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        raise NotImplementedError("Subclasses must implement update()")
