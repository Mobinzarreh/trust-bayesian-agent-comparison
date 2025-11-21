"""Trust-based focal agent with dual-state representation."""

import numpy as np
from .base import BaseAgent
from ..config import (
    TRUST_MIN,
    TRUST_MAX,
    LOSS_AVERSION,
    MEMORY_DISCOUNT,
    TRUST_DISCOUNT,
    TRUST_SMOOTHING,
    ETA,
    NOISE_SIGMA,
    STOCHASTIC,
    INVERSE_TEMPERATURE,
    EPS,
    LAMBDA_SURPRISE,
    stag_indifference_threshold,
)


class FocalAgent(BaseAgent):
    """
    Trust-based agent with dual-state belief representation (signal x, trust t).
    Updates beliefs using asymmetric penalties for betrayal and surprise.
    """
    
    def __init__(
        self,
        u_i: float = None,
        t_init: float = 0.1,
        eta: float = ETA,
        noise_sigma: float = NOISE_SIGMA,
        memory_discount: float = MEMORY_DISCOUNT,
        trust_discount: float = TRUST_DISCOUNT,
        trust_smoothing: float = TRUST_SMOOTHING,
        stochastic: bool = STOCHASTIC,
        inv_temp: float = INVERSE_TEMPERATURE,
        loss_aversion: float = LOSS_AVERSION,
        lambda_surprise: float = LAMBDA_SURPRISE,
    ):
        """
        Initialize focal agent.
        
        Args:
            u_i: Initial signal value (if None, uses 1 - decision_threshold)
            t_init: Initial trust level
            eta: Learning rate for signal update
            noise_sigma: Gaussian noise std for signal exploration
            memory_discount: Discount factor for signal history
            trust_discount: Discount factor for trust history
            trust_smoothing: Smoothing factor for trust convergence
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
            loss_aversion: λ ≥ 1: betrayal penalty multiplier
            lambda_surprise: μ ∈ [0,λ]: surprise penalty weight
        """
        super().__init__(stochastic=stochastic, inv_temp=inv_temp)
        
        # Validations
        if not 0 <= memory_discount <= 1:
            raise ValueError("memory_discount must be in [0,1]")
        if not 0 <= trust_discount <= 1:
            raise ValueError("trust_discount must be in [0,1]")
        if not 0 <= trust_smoothing <= 1:
            raise ValueError("trust_smoothing must be in [0,1]")
        if u_i is not None and not 0 <= u_i <= 1:
            raise ValueError("u_i must be between 0 and 1")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0,1]")
        if noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if loss_aversion < 1:
            raise ValueError("loss_aversion must be at least 1")
        if not 0 <= lambda_surprise <= loss_aversion:
            raise ValueError("lambda_surprise must be in [0, loss_aversion]")
        
        # State variables
        self.x = u_i if u_i is not None else 1 - stag_indifference_threshold()
        self.t = t_init
        
        # Parameters
        self.eta = eta
        self.noise_sigma = noise_sigma
        self.memory_discount = memory_discount
        self.trust_discount = trust_discount
        self.trust_smoothing = trust_smoothing
        self.loss_aversion = loss_aversion
        self.lambda_surprise = lambda_surprise
        
        # History tracking
        self.trust_match_hist = []
        self.trust_betrayal_hist = []
        self.trust_surprise_hist = []
        self.action_history = []
        
        # For compatibility
        self.history_x = [self.x]
        self.history_t = [self.t]
    
    def expected_p(self) -> float:
        """
        Expected cooperation probability from dual-state belief.
        
        Uses Beta distribution parameterization:
        α = EPS + x * t
        β = EPS + (1 - x) * t
        E[p] = α / (α + β)
        
        Returns:
            float: Expected probability of partner cooperation
        """
        alpha = EPS + self.x * self.t
        beta = EPS + (1 - self.x) * self.t
        return alpha / (alpha + beta)
    
    def update(self, partner_choice: int):
        """
        Update both trust and signal based on observed partner choice.
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        new_trust = self._compute_new_trust(partner_choice)
        new_signal = self._compute_new_signal(partner_choice)
        self.t = new_trust
        self.x = new_signal
        
        # Record history
        self.history_x.append(self.x)
        self.history_t.append(self.t)
    
    def _compute_new_trust(self, partner_choice: int) -> float:
        """
        Asymmetric trust update with loss aversion.
        
        Categories:
        - match (expected == observed)     → +1 to numerator & denominator
        - betrayal (expected C, saw D)     → +λ to denominator (penalty)
        - surprise (expected D, saw C)     → +μ to denominator (smaller penalty)
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
            
        Returns:
            float: Updated trust level
        """
        expected = 1 if self.x > 0.5 else 0
        obs = int(partner_choice)
        
        is_match = 1.0 if (obs == expected) else 0.0
        is_betrayal = 1.0 if (expected == 1 and obs == 0) else 0.0
        is_surprise = 1.0 if (expected == 0 and obs == 1) else 0.0
        
        # Include-current sequences
        M = (self.trust_match_hist + [is_match]) if self.trust_match_hist else [is_match]
        B = (self.trust_betrayal_hist + [is_betrayal]) if self.trust_betrayal_hist else [is_betrayal]
        U = (self.trust_surprise_hist + [is_surprise]) if self.trust_surprise_hist else [is_surprise]
        
        # Apply exponential discounting
        nM = len(M)
        wM = [self.trust_discount ** (nM - 1 - i) for i in range(nM)]
        nB = len(B)
        wB = [self.trust_discount ** (nB - 1 - i) for i in range(nB)]
        nU = len(U)
        wU = [self.trust_discount ** (nU - 1 - i) for i in range(nU)]
        
        WM = sum(e * w for e, w in zip(M, wM))
        WB = sum(e * w for e, w in zip(B, wB))
        WU = sum(e * w for e, w in zip(U, wU))
        
        # Asymmetric consistency measure
        consistency = WM / (WM + self.loss_aversion * WB + self.lambda_surprise * WU + 1e-9)
        
        target_trust = TRUST_MIN + (TRUST_MAX - TRUST_MIN) * consistency
        new_trust = (1 - self.trust_smoothing) * self.t + self.trust_smoothing * target_trust
        new_trust = float(np.clip(new_trust, TRUST_MIN, TRUST_MAX))
        
        # Record for next iteration
        self.trust_match_hist.append(is_match)
        self.trust_betrayal_hist.append(is_betrayal)
        self.trust_surprise_hist.append(is_surprise)
        
        return new_trust
    
    def _compute_new_signal(self, partner_choice: int) -> float:
        """
        Recency-weighted EWMA update with noise.
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
            
        Returns:
            float: Updated signal value
        """
        obs = int(partner_choice)
        
        if self.action_history:
            actions = self.action_history + [obs]
            n = len(actions)
            weights = [self.memory_discount ** (n - 1 - i) for i in range(n)]
            P_obs = sum(a * w for a, w in zip(actions, weights)) / sum(weights)
        else:
            P_obs = float(obs)
        
        noise = np.random.normal(0, self.noise_sigma)
        new_signal = float(np.clip(self.x + self.eta * (P_obs - self.x) + noise, 0.0, 1.0))
        
        self.action_history.append(obs)
        return new_signal
