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
    PavlovPartner,
    TitForTatDefectPartner,
)

# Belief-driven strategies (mirror focal agent dynamics)
from .belief_driven import (
    BeliefDrivenPartnerBase,
    StrategicCheaterPartner,
    ExpectationViolationPartner,
    AdaptivePartner,
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
    'PavlovPartner',
    'TitForTatDefectPartner',
    # Belief-driven
    'BeliefDrivenPartnerBase',
    'StrategicCheaterPartner',
    'ExpectationViolationPartner',
    'AdaptivePartner',
]
