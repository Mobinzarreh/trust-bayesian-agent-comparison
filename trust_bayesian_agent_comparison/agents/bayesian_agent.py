"""Bayesian agent with Beta-Bernoulli learning."""

from .base import BaseAgent
from ..config import EPS, STOCHASTIC, INVERSE_TEMPERATURE


class BayesianFocalAgent(BaseAgent):
    """
    Bayesian learning agent using Beta-Bernoulli conjugate prior.
    
    Models partner cooperation probability with Beta distribution:
    - Prior: Beta(α₀, β₀)
    - Updates: α += 1 when cooperation observed, β += 1 when defection observed
    - Belief: E[θ] = α / (α + β)
    """
    
    def __init__(
        self,
        alpha0: float = EPS,
        beta0: float = EPS,
        stochastic: bool = STOCHASTIC,
        inv_temp: float = INVERSE_TEMPERATURE,
    ):
        """
        Initialize Bayesian agent.
        
        Args:
            alpha0: Prior pseudo-count for cooperation (α₀)
            beta0: Prior pseudo-count for defection (β₀)
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
        """
        super().__init__(stochastic=stochastic, inv_temp=inv_temp)
        
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be > 0")
        
        self.alpha = float(alpha0)
        self.beta = float(beta0)
        
        # Store history
        self.history_alpha = [self.alpha]
        self.history_beta = [self.beta]
    
    def expected_p(self) -> float:
        """
        Expected cooperation probability = posterior mean of Beta distribution.
        
        Returns:
            float: E[θ] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, partner_choice: int):
        """
        Bayesian update: increment alpha for cooperate, beta for defect.
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        if partner_choice not in (0, 1):
            raise ValueError("partner_choice must be 0 or 1")
        
        self.alpha += partner_choice
        self.beta += (1 - partner_choice)
        
        # Record history
        self.history_alpha.append(self.alpha)
        self.history_beta.append(self.beta)
    
    def posterior_mean(self) -> float:
        """Backwards compatibility: same as expected_p()."""
        return self.expected_p()
