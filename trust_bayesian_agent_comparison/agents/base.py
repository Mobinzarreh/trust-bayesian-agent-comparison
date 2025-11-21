"""Base agent class with shared decision logic."""

from ..config import DECISION_THRESHOLD


class BaseAgent:
    """Base agent with shared decision-making logic."""
    
    def __init__(self):
        """Initialize base agent."""
        self.name = self.__class__.__name__
    
    def get_belief(self) -> float:
        """
        Get current belief/expectation about partner cooperation.
        
        Returns:
            float: Probability that partner will cooperate
        """
        raise NotImplementedError("Subclasses must implement get_belief()")
    
    def decide(self) -> int:
        """
        Make cooperation decision based on belief and threshold.
        
        Uses the stag hunt indifference threshold (2/3 for standard payoffs)
        to determine action: cooperate if belief >= threshold, else defect.
        
        Returns:
            int: 1 for cooperate, 0 for defect
        """
        belief = self.get_belief()
        return 1 if belief >= DECISION_THRESHOLD else 0
    
    def update(self, partner_action: int):
        """
        Update internal state based on observed partner action.
        
        Args:
            partner_action: Partner's action (1=cooperate, 0=defect)
        """
        raise NotImplementedError("Subclasses must implement update()")
