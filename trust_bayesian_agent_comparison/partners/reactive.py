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
    Pavlov (Win-Stay, Lose-Shift) strategy.
    
    - Win (stay): both cooperated (1,1) or both defected (0,0) → repeat
    - Lose (shift): mismatch (1,0) or (0,1) → switch
    
    Simpler than payoff-based: just looks at whether actions matched.
    """
    
    def __init__(self):
        """Initialize Pavlov partner."""
        super().__init__()
        self.last_own_choice = 1  # Start with cooperation
        self.last_agent_choice = None
        self.current_choice = None
    
    def decide(self, round_num: int) -> int:
        """Win-stay, lose-shift based on action matching."""
        if round_num == 0:
            self.current_choice = 1  # Start with cooperation
            return self.current_choice
        
        # Win-Stay: if last round matched (both same action), repeat
        # Lose-Shift: if mismatch, switch
        if self.last_agent_choice is not None and self.last_own_choice is not None:
            if self.last_own_choice == self.last_agent_choice:
                # Win: both chose same → stay
                self.current_choice = self.last_own_choice
            else:
                # Lose: mismatch → shift
                self.current_choice = 1 - self.last_own_choice
        else:
            self.current_choice = self.last_own_choice
        
        return self.current_choice
    
    def update(self, agent_action: int):
        """Track what happened for next decision."""
        super().update(agent_action)
        self.last_agent_choice = agent_action
        self.last_own_choice = self.current_choice
    
    def reset(self):
        """Reset Pavlov state."""
        super().reset()
        self.last_own_choice = 1
        self.last_agent_choice = None
        self.current_choice = None
