"""Reactive strategy partners that respond to agent's previous actions."""

from .base import BasePartner


class TitForTatCooperatePartner(BasePartner):
    """
    Tit-for-tat starting with cooperation.
    
    Cooperates on first round, then mirrors agent's previous action.
    """
    
    def decide(self, round_num: int) -> int:
        """Cooperate first, then copy agent's last action."""
        if round_num == 0:
            return 1  # Start cooperating
        return self.history[-1]


class TitForTatDefectPartner(BasePartner):
    """
    Tit-for-tat starting with defection.
    
    Defects on first round, then mirrors agent's previous action.
    """
    
    def decide(self, round_num: int) -> int:
        """Defect first, then copy agent's last action."""
        if round_num == 0:
            return 0  # Start defecting
        return self.history[-1]


class SuspiciousTitForTatPartner(BasePartner):
    """
    Suspicious tit-for-tat: defects first, then copies.
    
    Alias for TitForTatDefectPartner with more descriptive name.
    """
    
    def decide(self, round_num: int) -> int:
        """Defect first, then copy agent's last action."""
        if round_num == 0:
            return 0
        return self.history[-1]


class GrimTriggerPartner(BasePartner):
    """
    Grim trigger strategy.
    
    Cooperates until agent defects once, then defects forever.
    """
    
    def __init__(self):
        """Initialize grim trigger partner."""
        super().__init__()
        self.triggered = False
    
    def decide(self, round_num: int) -> int:
        """Cooperate until triggered, then defect forever."""
        if self.triggered:
            return 0
        return 1
    
    def update(self, agent_action: int):
        """Check if agent defected."""
        super().update(agent_action)
        if agent_action == 0:
            self.triggered = True
    
    def reset(self):
        """Reset trigger state."""
        super().reset()
        self.triggered = False


class PavlovPartner(BasePartner):
    """
    Pavlov (win-stay, lose-shift) strategy.
    
    Repeats action if payoff was good (4 or 3), switches if bad (0 or 2).
    Requires tracking own actions and outcomes.
    """
    
    def __init__(self):
        """Initialize Pavlov partner."""
        super().__init__()
        self.last_action = 1  # Start with cooperation
        self.last_payoff = None
    
    def decide(self, round_num: int) -> int:
        """Win-stay, lose-shift logic."""
        if round_num == 0:
            return self.last_action
        
        # Switch if last payoff was bad (0 or 2)
        if self.last_payoff in [0, 2]:
            self.last_action = 1 - self.last_action
        # Stay if payoff was good (3 or 4)
        
        return self.last_action
    
    def update(self, agent_action: int):
        """Update state and calculate payoff."""
        super().update(agent_action)
        
        # Calculate payoff (simplified for partner's perspective)
        # [agent_action, partner_action] -> payoff for partner
        if agent_action == 1 and self.last_action == 1:
            self.last_payoff = 4  # Mutual cooperation
        elif agent_action == 0 and self.last_action == 1:
            self.last_payoff = 0  # Sucker
        elif agent_action == 1 and self.last_action == 0:
            self.last_payoff = 3  # Temptation
        else:  # agent_action == 0 and self.last_action == 0
            self.last_payoff = 2  # Mutual defection
    
    def reset(self):
        """Reset Pavlov state."""
        super().reset()
        self.last_action = 1
        self.last_payoff = None
