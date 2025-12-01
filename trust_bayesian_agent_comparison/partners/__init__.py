"""Partner strategies for agent interactions."""

from .base import BasePartner

# Fixed strategies
from .fixed import (
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    RandomPartner,
    PeriodicCheaterPartner,
    SingleCyclePartner,
    GradualDeteriorationPartner,
)

# Reactive strategies  
from .reactive import (
    TitForTatCooperatePartner,
    GrimTriggerPartner,
    SuspiciousTitForTatPartner,
)

# Belief-driven strategies (mirror focal agent dynamics)
from .belief_driven import (
    BeliefDrivenPartnerBase,
    StrategicCheaterPartner,
    ExpectationViolationPartner,
)

__all__ = [
    'BasePartner',
    # Fixed
    'AlwaysCooperatePartner',
    'AlwaysDefectPartner', 
    'RandomPartner',
    'PeriodicCheaterPartner',
    'SingleCyclePartner',
    'GradualDeteriorationPartner',
    # Reactive
    'TitForTatCooperatePartner',
    'GrimTriggerPartner',
    'SuspiciousTitForTatPartner',
    # Belief-driven
    'BeliefDrivenPartnerBase',
    'StrategicCheaterPartner',
    'ExpectationViolationPartner',
]
