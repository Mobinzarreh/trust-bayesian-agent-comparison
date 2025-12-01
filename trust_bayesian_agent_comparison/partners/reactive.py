"""Reactive strategy partners that respond to agent's previous actions."""

import random


class TitForTatCooperatePartner:
    """
    Start by cooperating; afterwards mirror the focal agent's last action.
    """
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate first, then copy agent's last action."""
        return 1 if last_agent_choice is None else int(last_agent_choice)
    
    def observe(self, focal_agent_action: int):
        pass  # No internal state needed
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class TitForTatDefectPartner:
    """
    Start by defecting; afterwards mirror the focal agent's last action.
    """
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Defect first, then copy agent's last action."""
        return 0 if last_agent_choice is None else int(last_agent_choice)
    
    def observe(self, focal_agent_action: int):
        pass  # No internal state needed
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class SuspiciousTitForTatPartner:
    """
    Suspicious tit-for-tat: defects first, then copies.
    
    Alias for TitForTatDefectPartner with more descriptive name.
    """
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Defect first, then copy agent's last action."""
        return 0 if last_agent_choice is None else int(last_agent_choice)
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class GrimTriggerPartner:
    """
    Grim Trigger (Grudger) - Unforgiving Partner Strategy
    
    Behavior:
    - Starts by cooperating unconditionally
    - Continues cooperating as long as the agent NEVER defects
    - Once the agent defects even ONCE, switches to permanent defection forever
    - Never forgives - holds an infinite grudge after betrayal
    """
    
    def __init__(self):
        """Initialize grim trigger partner."""
        self.triggered = False  # Has the agent ever defected?
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate until triggered, then defect forever."""
        # If already triggered, defect forever
        if self.triggered:
            return 0
        
        # First round: cooperate
        if last_agent_choice is None:
            return 1
        
        # Check if agent defected in the last round
        if last_agent_choice == 0:
            self.triggered = True  # Trigger activated
            return 0  # Immediately retaliate
        
        # Agent cooperated, continue cooperating
        return 1
    
    def observe(self, focal_agent_action: int):
        """Update trigger state based on agent's action."""
        if focal_agent_action == 0:
            self.triggered = True
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        self.observe(agent_action)


class PavlovPartner:
    """
    Win-Stay, Lose-Shift (Pavlov):
    - If last round was mutually beneficial (CC or DD), repeat
    - If last round was not mutually beneficial (CD or DC), switch
    
    For stag hunt:
    - Win (stay): both cooperated (1,1) or both defected (0,0) 
    - Lose (shift): mismatch (1,0) or (0,1)
    """
    
    def __init__(self):
        """Initialize Pavlov partner."""
        self.last_own_choice = 1  # Start with cooperation
        self.last_agent_choice = None
        self.current_choice = None
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Win-stay, lose-shift based on action matching."""
        if round_num == 0:
            self.current_choice = 1  # Start with cooperation
            return self.current_choice
        
        # Win-Stay: if last round matched (both same action), repeat
        # Lose-Shift: if mismatch, switch
        if last_agent_choice is not None and self.last_own_choice is not None:
            if self.last_own_choice == last_agent_choice:
                # Win: both chose same → stay
                self.current_choice = self.last_own_choice
            else:
                # Lose: mismatch → shift
                self.current_choice = 1 - self.last_own_choice
        else:
            self.current_choice = self.last_own_choice
        
        return self.current_choice
    
    def observe(self, focal_agent_action: int):
        """Track what happened in this round for next decision."""
        self.last_agent_choice = focal_agent_action
        self.last_own_choice = self.current_choice
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        self.observe(agent_action)
