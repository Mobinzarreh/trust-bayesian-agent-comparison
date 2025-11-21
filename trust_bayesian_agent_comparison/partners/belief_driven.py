"""Belief-driven partners that maintain trust dynamics like focal agents."""

import numpy as np
from ..config import (
    TRUST_MIN,
    TRUST_MAX,
    MEMORY_DISCOUNT,
    TRUST_DISCOUNT,
    TRUST_SMOOTHING,
    ETA,
    NOISE_SIGMA,
    stag_indifference_threshold,
)


class BeliefDrivenPartnerBase:
    """
    Partner maintains its own beliefs (x_hat in [0,1], t_hat in [t_min,t_max]),
    updates TRUST with asymmetric penalties (λ for betrayal, μ for surprise),
    and updates SIGNAL via include-current EWMA (no caution).
    
    Mirrors the focal agent's trust dynamics but from partner's perspective.
    """
    
    def __init__(
        self,
        u_i: float = None,
        t0: float = 0.1,
        eta: float = ETA,
        noise_sigma: float = NOISE_SIGMA,
        memory_discount: float = MEMORY_DISCOUNT,
        trust_discount: float = TRUST_DISCOUNT,
        trust_smoothing: float = TRUST_SMOOTHING,
        t_min: float = TRUST_MIN,
        t_max: float = TRUST_MAX,
        loss_aversion_hat: float = 2.0,  # λ ≥ 1: betrayal penalty
        lambda_surprise_hat: float = 0.5,  # μ ∈ [0,λ]: surprise penalty
    ):
        """
        Initialize belief-driven partner.
        
        Args:
            u_i: Initial signal value (if None, uses 1 - threshold)
            t0: Initial trust level
            eta: Learning rate for signal update
            noise_sigma: Gaussian noise std for exploration
            memory_discount: Discount factor for signal history
            trust_discount: Discount factor for trust history
            trust_smoothing: Smoothing factor for trust convergence
            t_min: Minimum trust level
            t_max: Maximum trust level
            loss_aversion_hat: λ ≥ 1, betrayal penalty multiplier
            lambda_surprise_hat: μ ∈ [0,λ], surprise penalty weight
        """
        # Validations
        if u_i is not None and not 0 <= u_i <= 1:
            raise ValueError("u_i must be in [0,1]")
        if not t_min <= t0 <= t_max:
            raise ValueError("t0 must be between t_min and t_max")
        if loss_aversion_hat < 1:
            raise ValueError("loss_aversion_hat (λ) must be ≥ 1")
        if not 0 <= lambda_surprise_hat <= loss_aversion_hat:
            raise ValueError("lambda_surprise_hat (μ) must be in [0, loss_aversion_hat]")
        
        # Priors
        self.x_hat = u_i if u_i is not None else 1 - stag_indifference_threshold()
        self.t_hat = float(t0)
        
        # Process params
        self.eta = eta
        self.noise_sigma = noise_sigma
        self.memory_discount = memory_discount
        self.trust_discount = trust_discount
        self.trust_smoothing = trust_smoothing
        self.t_min = t_min
        self.t_max = t_max
        
        # Asymmetry params
        self.loss_aversion_hat = loss_aversion_hat
        self.lambda_surprise_hat = lambda_surprise_hat
        
        # Histories
        self.partner_match_history = []
        self.partner_betrayal_history = []
        self.partner_surprise_history = []
        self.obs_actions = []
        
        # Temporary storage
        self._last_observed_action = None
    
    def _expected_choice_from_signal(self, x: float) -> int:
        """Predict action from signal (threshold at 0.5)."""
        return 1 if x > 0.5 else 0
    
    def _update_trust(self, observed_action: int):
        """
        Asymmetric trust update with current event (no lag).
        
        Categories:
        - match (expected == observed)     → contributes to numerator & denominator
        - betrayal (expected C, observed D) → penalty λ in denominator
        - surprise (expected D, observed C) → penalty μ in denominator
        """
        expected = self._expected_choice_from_signal(self.x_hat)
        obs = int(observed_action)
        
        # Classify current outcome
        is_match = 1.0 if (obs == expected) else 0.0
        is_betrayal = 1.0 if (expected == 1 and obs == 0) else 0.0  # expected C, saw D
        is_surprise = 1.0 if (expected == 0 and obs == 1) else 0.0  # expected D, saw C
        
        # Build sequences including current event (newest weight = 1)
        M = (self.partner_match_history + [is_match]) if self.partner_match_history else [is_match]
        B = (self.partner_betrayal_history + [is_betrayal]) if self.partner_betrayal_history else [is_betrayal]
        U = (self.partner_surprise_history + [is_surprise]) if self.partner_surprise_history else [is_surprise]
        
        # Apply exponential discounting
        nM = len(M)
        wM = [self.trust_discount ** (nM - 1 - i) for i in range(nM)]
        nB = len(B)
        wB = [self.trust_discount ** (nB - 1 - i) for i in range(nB)]
        nU = len(U)
        wU = [self.trust_discount ** (nU - 1 - i) for i in range(nU)]
        
        WM = sum(e * w for e, w in zip(M, wM))  # matches
        WB = sum(e * w for e, w in zip(B, wB))  # betrayals
        WU = sum(e * w for e, w in zip(U, wU))  # surprises
        
        lam = self.loss_aversion_hat
        mu = self.lambda_surprise_hat
        consistency = WM / (WM + lam * WB + mu * WU + 1e-9)  # in [0,1]
        
        # Update t_hat
        T_target = self.t_min + (self.t_max - self.t_min) * consistency
        new_t = (1 - self.trust_smoothing) * self.t_hat + self.trust_smoothing * T_target
        self.t_hat = float(np.clip(new_t, self.t_min, self.t_max))
        
        # Record current event for next round
        self.partner_match_history.append(is_match)
        self.partner_betrayal_history.append(is_betrayal)
        self.partner_surprise_history.append(is_surprise)
    
    def _update_signal(self):
        """Recency-weighted EWMA update with noise."""
        obs = int(self._last_observed_action)
        
        if self.obs_actions:
            actions = self.obs_actions + [obs]
            n = len(actions)
            weights = [self.memory_discount ** (n - 1 - i) for i in range(n)]
            P_obs = sum(a * w for a, w in zip(actions, weights)) / sum(weights)
        else:
            P_obs = float(obs)
        
        noise = np.random.normal(0, self.noise_sigma)
        new_signal = float(np.clip(self.x_hat + self.eta * (P_obs - self.x_hat) + noise, 0.0, 1.0))
        self.x_hat = new_signal
        return new_signal
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Make decision (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement decide()")
    
    def observe(self, focal_agent_action: int):
        """
        Observe focal agent's action and update beliefs.
        
        1) Set current observed action
        2) Update TRUST with asymmetric penalties
        3) Update SIGNAL with EWMA
        4) Append to history
        
        Args:
            focal_agent_action: Agent's action (1=cooperate, 0=defect)
        """
        self._last_observed_action = int(focal_agent_action)
        
        # TRUST update (include-current)
        self._update_trust(self._last_observed_action)
        
        # SIGNAL update (include-current, no caution)
        self._update_signal()
        
        # Record for next round
        self.obs_actions.append(self._last_observed_action)


class AdaptivePartner(BeliefDrivenPartnerBase):
    """
    Adaptive partner that mirrors agent's expected behavior.
    
    Cooperates when it predicts agent will cooperate (x_hat > 0.5),
    defects when it predicts agent will defect (x_hat ≤ 0.5).
    """
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate if predicting agent will cooperate."""
        return 1 if self.x_hat > 0.5 else 0


class StrategicCheaterPartner(BeliefDrivenPartnerBase):
    """
    Strategic cheater that exploits when agent's trust is high.
    
    Defects when trust exceeds threshold, cooperates otherwise.
    Tests whether agent can recover from betrayal at high trust.
    """
    
    def __init__(self, t_threshold: float = 5.0, **kwargs):
        """
        Initialize strategic cheater.
        
        Args:
            t_threshold: Trust threshold above which to defect
            **kwargs: Arguments for BeliefDrivenPartnerBase
        """
        super().__init__(**kwargs)
        self.t_threshold = t_threshold
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Defect when trust exceeds threshold."""
        return 0 if self.t_hat > self.t_threshold else 1


class ExpectationViolationPartner(BeliefDrivenPartnerBase):
    """
    Expectation violation partner that does the opposite of prediction.
    
    Defects when predicting agent will cooperate,
    cooperates when predicting agent will defect.
    Maximizes agent's prediction errors.
    """
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Choose opposite of predicted agent action."""
        return 0 if self.x_hat > 0.5 else 1
