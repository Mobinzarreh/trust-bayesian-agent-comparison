"""Base partner class."""


class BasePartner:
    """Base class for all partner strategies."""
    
    def __init__(self):
        """Initialize base partner."""
        self.name = self.__class__.__name__
        self.history = []
    
    def decide(self, round_num: int) -> int:
        """
        Decide action for current round.
        
        Args:
            round_num: Current round number (0-indexed)
            
        Returns:
            int: 1 for cooperate, 0 for defect
        """
        raise NotImplementedError("Subclasses must implement decide()")
    
    def update(self, agent_action: int):
        """
        Update internal state based on agent's action.
        
        Args:
            agent_action: Agent's action (1=cooperate, 0=defect)
        """
        self.history.append(agent_action)
    
    def reset(self):
        """Reset partner state for new simulation."""
        self.history = []
