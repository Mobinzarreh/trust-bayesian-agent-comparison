"""Partner strategies for agent interactions."""

from .base import BasePartner

# Fixed strategies
from .fixed import (
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    RandomPartner,
    PeriodicCheaterPartner,
    SingleCyclePartner,
    ProbabilisticPartner,
    GradualDeteriorationPartner,
)

# Reactive strategies  
from .reactive import (
    TitForTatCooperatePartner,
    TitForTatDefectPartner,
    GrimTriggerPartner,
    PavlovPartner,
    SuspiciousTitForTatPartner,
)

# Belief-driven strategies (mirror focal agent dynamics)
from .belief_driven import (
    BeliefDrivenPartnerBase,
    AdaptivePartner,
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
    'ProbabilisticPartner',
    'GradualDeteriorationPartner',
    # Reactive
    'TitForTatCooperatePartner',
    'TitForTatDefectPartner',
    'GrimTriggerPartner',
    'PavlovPartner',
    'SuspiciousTitForTatPartner',
    # Belief-driven
    'BeliefDrivenPartnerBase',
    'AdaptivePartner',
    'StrategicCheaterPartner',
    'ExpectationViolationPartner',
]
