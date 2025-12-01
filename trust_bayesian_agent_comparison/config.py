"""
Configuration module for trust-bayesian agent comparison study.
Centralized constants and default parameters.
"""
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_timestamped_path(base_dir: Path, prefix: str = "", suffix: str = ".csv") -> Path:
    """Generate timestamped file path for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{prefix}{timestamp}{suffix}"

# ============================================================================
# GAME PARAMETERS
# ============================================================================
# Stag Hunt Payoff Matrix
PAYOFF_MATRIX = np.array([
    [[2, 2], [3, 0]],  # [Hare][Partner action][player_id]
    [[0, 3], [4, 4]]   # [Stag][Partner action][player_id]
])

def get_payoff(player1_strategy: int, player2_strategy: int, player_id: int) -> float:
    """Get payoff for a given strategy combination."""
    return float(PAYOFF_MATRIX[int(player1_strategy), int(player2_strategy), int(player_id)])

def stag_indifference_threshold() -> float:
    """
    Calculate the normalized deviation loss (threshold) for cooperation.
    
    Returns the probability threshold at which the agent is indifferent
    between cooperating (Stag) and defecting (Hare) based on the payoff matrix.
    
    Formula: p* = (a00 - a10) / [(a00 - a10) + (a11 - a01)]
    where aij is the payoff for (agent_action=i, partner_action=j)
    
    Returns:
        float: Threshold probability in [0, 1]
    """
    a00 = get_payoff(0, 0, 0)
    a01 = get_payoff(0, 1, 0)
    a10 = get_payoff(1, 0, 0)
    a11 = get_payoff(1, 1, 0)
    
    den = (a00 - a10) + (a11 - a01)
    return float((a00 - a10) / den) if den != 0 else 0.5

# Decision threshold for cooperation (dynamically calculated)
DECISION_THRESHOLD = stag_indifference_threshold()

# ============================================================================
# AGENT PARAMETERS
# ============================================================================
# Learning parameters for dual-state agent
# The agent maintains two state variables:
#   - x (signal): Expected cooperation rate, updated via EWMA
#   - t (trust): Confidence/precision in the signal, updated via consistency measure
ETA = 0.3                  # Learning rate for signal (x) update via EWMA: x_new = x + η*(obs - x)
LEARNING_RATE_T = 0.2      # Learning rate for trust (t) update: t_new = (1-α)*t + α*target
                           # Note: This is implemented as 'trust_smoothing' parameter in FocalAgent
NOISE_SIGMA = 0.02         # Gaussian noise for exploration

# Trust-based agent defaults
MEMORY_DISCOUNT = 0.9      # Recency weighting for signal update (γ_x)
TRUST_DISCOUNT = 0.80       # Recency weighting for trust update (γ_t)
TRUST_SMOOTHING = LEARNING_RATE_T  # Alias: smoothing factor = learning rate for trust
TRUST_MIN = 0.0            # Minimum trust level
TRUST_MAX = 10.0           # Maximum trust level
EPS = 1                  # Small constant for Beta distribution

# Asymmetric loss parameters
LOSS_AVERSION = 2.0        # Betrayal penalty multiplier (λ)
LAMBDA_SURPRISE = 0.5      # Surprise penalty multiplier (μ)

# Decision-making
STOCHASTIC = True          # Use probabilistic (logit) choice
INVERSE_TEMPERATURE = 2.0  # Exploration-exploitation trade-off

# Bayesian agent defaults
ALPHA_0 = EPS              # Prior cooperation count
BETA_0 = EPS               # Prior defection count

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
NUM_ROUNDS = 70            # Default rounds per simulation
DEFAULT_SEED = 42          # Reproducibility seed

# ============================================================================
# SENSITIVITY ANALYSIS PARAMETERS
# ============================================================================
SENSITIVITY_SEEDS = (42, 43, 44)  # Multiple seeds for robustness

# Parameter grids
ETA_GRID = np.linspace(0.0, 1.0, 6)
MEMORY_DISCOUNT_GRID = np.linspace(0.0, 1.0, 4)
TRUST_DISCOUNT_GRID = np.linspace(0.0, 1.0, 4)
TRUST_SMOOTHING_GRID = np.linspace(0.0, 1.0, 4)
LOSS_AVERSION_GRID = np.linspace(1.0, 5.0, 5)
LAMBDA_SURPRISE_GRID = np.linspace(0.0, 1.0, 4)

# Parallel processing
N_JOBS = -1                # Use all available cores
VERBOSE = 10               # Progress reporting level

# ============================================================================
# MONTE CARLO PARAMETERS
# ============================================================================
NUM_MONTE_CARLO_RUNS = 300  # Simulations per partner
MC_BASE_SEED = 42           # Base seed for MC runs

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
# Standard figure size
FIGURE_SIZE = (14, 8)
GRID_FIGURE_SIZE = (16, 12)

# Color palettes
TRUST_COLOR = '#2E86AB'
BAYESIAN_COLOR = '#A23B72'
PARTNER_COLOR = '#F18F01'
THRESHOLD_COLOR = '#C73E1D'

# Plot styling
DPI = 100
SAVE_FORMAT = 'png'
TRANSPARENT_BG = False
