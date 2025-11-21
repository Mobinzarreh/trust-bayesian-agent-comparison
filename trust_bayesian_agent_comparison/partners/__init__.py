"""Partner strategies for agent interactions."""

from .base import BasePartner

# Fixed strategies
from .fixed import (
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    RandomPartner,
)

# Reactive strategies  
from .reactive import (
    TitForTatCooperatePartner,
    TitForTatDefectPartner,
    GrimTriggerPartner,
    PavlovPartner,
    SuspiciousTitForTatPartner,
)

# Adaptive strategies
from .adaptive import (
    StrategicCheaterPartner,
    AdaptiveStrategicPartner,
    BayesianDeceptivePartner,
    ExploitativePartner,
)

__all__ = [
    'BasePartner',
    # Fixed
    'AlwaysCooperatePartner',
    'AlwaysDefectPartner', 
    'RandomPartner',
    # Reactive
    'TitForTatCooperatePartner',
    'TitForTatDefectPartner',
    'GrimTriggerPartner',
    'PavlovPartner',
    'SuspiciousTitForTatPartner',
    # Adaptive
    'StrategicCheaterPartner',
    'AdaptiveStrategicPartner',
    'BayesianDeceptivePartner',
    'ExploitativePartner',
]
