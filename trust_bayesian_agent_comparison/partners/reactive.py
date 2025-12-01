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
    Win-Stay, Lose-Shift (Pavlov) - Payoff-based implementation.
    
    Uses myopic best response logic based on actual payoffs received:
    - If opponent cooperated last → Cooperate (payoff 4 > 3)
    - If opponent defected last → Defect (payoff 2 > 0)
    
    This is equivalent to myopic best response in Stag Hunt:
    Choose the action that maximizes payoff assuming opponent repeats.
    
    For Stag Hunt payoffs [[2,3],[0,4]]:
    - vs Cooperate: Coop gets 4, Defect gets 3 → Cooperate
    - vs Defect: Coop gets 0, Defect gets 2 → Defect
    """
    
    def __init__(self):
        """Initialize Pavlov partner."""
        self.last_own_choice = 1  # Start with cooperation
        self.last_agent_choice = None
        self.current_choice = None
        # Stag Hunt payoffs: PAYOFFS[my_action][opponent_action]
        # 0 = Defect, 1 = Cooperate
        self.payoffs = [[2, 3], [0, 4]]  # [[DD, DC], [CD, CC]]
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """
        Myopic best response: choose action that maximizes payoff
        assuming opponent repeats their last action.
        """
        if round_num == 0 or last_agent_choice is None:
            self.current_choice = 1  # Start with cooperation
            return self.current_choice
        
        # Myopic best response: maximize payoff given opponent's last action
        payoff_if_coop = self.payoffs[1][last_agent_choice]  # If I cooperate
        payoff_if_def = self.payoffs[0][last_agent_choice]   # If I defect
        
        if payoff_if_coop > payoff_if_def:
            self.current_choice = 1  # Cooperate
        elif payoff_if_def > payoff_if_coop:
            self.current_choice = 0  # Defect
        else:
            # Tie: stay with last choice
            self.current_choice = self.last_own_choice
        
        return self.current_choice
    
    def observe(self, focal_agent_action: int):
        """Track what happened in this round for next decision."""
        self.last_agent_choice = focal_agent_action
        self.last_own_choice = self.current_choice
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        self.observe(agent_action)
