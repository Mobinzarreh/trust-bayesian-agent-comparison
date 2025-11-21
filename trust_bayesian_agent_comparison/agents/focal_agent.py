"""Trust-based focal agent with dual-state representation."""

import numpy as np
from .base import BaseAgent
from ..config import (
    TRUST_MIN,
    TRUST_MAX,
    LOSS_AVERSION,
    LEARNING_RATE_X,
    LEARNING_RATE_T
)


class FocalAgent(BaseAgent):
    """
    Trust-based agent with asymmetric trust dynamics.
    
    Uses dual-state representation:
    - x: normalized expectation ∈ [0, 1]
    - t: trust level ∈ [0, TRUST_MAX]
    
    Features asymmetric updates:
    - Loss aversion (λ): betrayal hurts more than reward
    - Asymmetry parameter (μ): cooperation updates vs defection updates
    """
    
    def __init__(
        self,
        loss_aversion: float = LOSS_AVERSION,
        mu: float = 0.5,
        eta_x: float = LEARNING_RATE_X,
        eta_t: float = LEARNING_RATE_T,
    ):
        """
        Initialize focal agent.
        
        Args:
            loss_aversion: Multiplier for negative prediction errors (λ > 1 for loss aversion)
            mu: Asymmetry parameter for trust updates (μ < 0.5 means defection hits harder)
            eta_x: Learning rate for expectation updates
            eta_t: Learning rate for trust updates
        """
        super().__init__()
        self.loss_aversion = loss_aversion
        self.mu = mu
        self.eta_x = eta_x
        self.eta_t = eta_t
        
        # Initialize state at midpoints
        self.x = 0.5  # Neutral expectation
        self.t = TRUST_MAX / 2  # Moderate trust
        
        # Store history
        self.history_x = [self.x]
        self.history_t = [self.t]
    
    def get_belief(self) -> float:
        """
        Get current belief (normalized expectation).
        
        Returns:
            float: Current value of x ∈ [0, 1]
        """
        return self.x
    
    def update(self, partner_action: int):
        """
        Update expectation and trust based on partner's action.
        
        Updates:
        1. Expectation (x): δx = η_x * (y - x) * [λ if y < x else 1]
        2. Trust (t): δt = η_t * (δx / (t + ε)) * [μ if y=1 else (1-μ)]
        
        Args:
            partner_action: Partner's action (1=cooperate, 0=defect)
        """
        y = partner_action
        
        # Compute prediction error
        delta = y - self.x
        
        # Asymmetric loss weighting
        weight = self.loss_aversion if delta < 0 else 1.0
        
        # Update expectation with loss aversion
        delta_x = self.eta_x * delta * weight
        self.x = np.clip(self.x + delta_x, 0.0, 1.0)
        
        # Update trust with asymmetry
        eps = 1e-6  # Prevent division by zero
        trust_change_base = delta_x / (self.t + eps)
        
        # Asymmetric trust dynamics
        if y == 1:  # Partner cooperated
            trust_multiplier = self.mu
        else:  # Partner defected
            trust_multiplier = 1.0 - self.mu
        
        delta_t = self.eta_t * trust_change_base * trust_multiplier
        self.t = np.clip(self.t + delta_t, TRUST_MIN, TRUST_MAX)
        
        # Record history
        self.history_x.append(self.x)
        self.history_t.append(self.t)
