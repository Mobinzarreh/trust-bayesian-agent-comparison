"""Bayesian agent with Beta-Bernoulli learning."""

from .base import BaseAgent


class BayesianFocalAgent(BaseAgent):
    """
    Bayesian learning agent using Beta-Bernoulli conjugate prior.
    
    Models partner cooperation probability with Beta distribution:
    - Prior: Beta(α₀, β₀)
    - Updates: α += 1 when cooperation observed, β += 1 when defection observed
    - Belief: E[θ] = α / (α + β)
    """
    
    def __init__(self, alpha_0: float = 0.5, beta_0: float = 0.5):
        """
        Initialize Bayesian agent.
        
        Args:
            alpha_0: Prior pseudo-count for cooperation (α₀)
            beta_0: Prior pseudo-count for defection (β₀)
        """
        super().__init__()
        self.alpha = alpha_0
        self.beta = beta_0
        
        # Store history
        self.history_alpha = [self.alpha]
        self.history_beta = [self.beta]
    
    def get_belief(self) -> float:
        """
        Get current belief (posterior mean).
        
        Returns:
            float: E[θ] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, partner_action: int):
        """
        Update Beta posterior based on partner's action.
        
        Args:
            partner_action: Partner's action (1=cooperate, 0=defect)
        """
        if partner_action == 1:
            self.alpha += 1  # Observed cooperation
        else:
            self.beta += 1   # Observed defection
        
        # Record history
        self.history_alpha.append(self.alpha)
        self.history_beta.append(self.beta)
