# Trust-Based vs Bayesian Agent Comparison - Refactored

## 🎯 Project Overview

This study compares trust-based and Bayesian learning agents in repeated stag hunt games across diverse partner strategies.

## 📁 New Project Structure

```
trust-bayesian-agent-comparison/
├── trust_bayesian_agent_comparison/    # Main package
│   ├── agents/                          # Agent implementations
│   ├── partners/                        # Partner strategies  
│   ├── simulation/                      # Simulation engine
│   ├── analysis/                        # Analysis tools
│   │   ├── sensitivity.py               # Parameter sweeps
│   │   ├── monte_carlo.py               # MC simulations
│   │   └── metrics.py                   # KPI calculations
│   ├── visualization/                   # Plotting utilities
│   └── config.py                        # Centralized configuration
├── notebooks/                           # Clean analysis notebooks
│   ├── 01_single_run_analysis.ipynb
│   ├── 02_sensitivity_analysis.ipynb
│   └── 03_monte_carlo_comparison.ipynb
├── scripts/                             # CLI tools
│   ├── run_sensitivity.py
│   └── run_monte_carlo.py
├── results/                             # Auto-organized outputs
│   ├── sensitivity/
│   │   └── 2025-11-21/                  # Dated subdirectories
│   └── monte_carlo/
└── tests/                               # Unit tests
```

## 🚀 Quick Start

### Option 1: Using Notebooks (Recommended for Exploration)

```python
# In notebook
from trust_bayesian_agent_comparison.analysis.sensitivity import SensitivityAnalysisManager

# Initialize manager
manager = SensitivityAnalysisManager()

# Run analysis (auto-saves with timestamp)
results = manager.run_analysis(
    partner_name="TitForTatCoop",
    partner_factory=lambda: TitForTatCooperatePartner(),
    threshold_direction="up",
    overwrite=False  # Load existing if available
)
```

### Option 2: Using CLI (Recommended for Batch Jobs)

```bash
# Run sensitivity for specific partners
python scripts/run_sensitivity.py --partners TitForTatCoop AlwaysDefect

# Run for all partners
python scripts/run_sensitivity.py --all

# Force re-run with custom seeds
python scripts/run_sensitivity.py --all --overwrite --seeds 42 43 44 45 46

# Use more parameter points
python scripts/run_sensitivity.py --partners Random --eta-points 10
```

## 📊 Key Features

### 1. **Automatic Result Management**
- Results saved in dated folders: `results/sensitivity/2025-11-21/`
- No accidental overwrites
- Timestamped filenames for version control
- Load existing results with `overwrite=False`

### 2. **Centralized Configuration**
All constants in one place (`config.py`):
```python
LOSS_AVERSION = 2.0
SENSITIVITY_SEEDS = (42, 43, 44)
NUM_ROUNDS = 70
```

### 3. **Modular Design**
```python
# Analysis logic separated from notebooks
from trust_bayesian_agent_comparison.analysis import (
    agent_coop_rate,
    mutual_coop_rate,
    betrayal_rate,
    compute_strategy_statistics
)
```

### 4. **Parallel Processing**
```python
# Automatically uses all CPU cores
results = sweep_learning_params(
    partner_factory=partner_factory,
    n_jobs=-1  # Use all cores
)
```

## 📝 Usage Examples

### Running Sensitivity Analysis

```python
from trust_bayesian_agent_comparison.analysis.sensitivity import SensitivityAnalysisManager
from trust_bayesian_agent_comparison.partners import TitForTatCooperatePartner

manager = SensitivityAnalysisManager()

# Single partner
result = manager.run_analysis(
    partner_name="TitForTatCoop",
    partner_factory=lambda: TitForTatCooperatePartner(),
    threshold_direction="up",
    overwrite=False
)

# Multiple partners
partner_configs = [
    ("TitForTatCoop", lambda: TitForTatCooperatePartner(), "up"),
    ("AlwaysDefect", lambda: AlwaysDefectPartner(), "down"),
]

results = manager.run_multiple(
    partner_configs=partner_configs,
    overwrite=False
)
```

### Computing Metrics

```python
from trust_bayesian_agent_comparison.analysis import (
    compute_strategy_statistics,
    calculate_payoffs
)

# Run simulation
df = run_agent_simulation(agent, partner, num_rounds=70)

# Get all statistics
stats = compute_strategy_statistics(df)
print(stats)
# {'agent_coop_rate': 0.85, 'mutual_coop_rate': 0.80, ...}
```

### Custom Parameter Grids

```python
import numpy as np

# Custom grid for deeper analysis
results = manager.run_analysis(
    partner_name="Strategic",
    partner_factory=lambda: StrategicCheaterPartner(),
    threshold_direction="down",
    # Override defaults
    loss_aversion_grid=np.linspace(1.0, 5.0, 20),  # More points
    seeds=(42, 43, 44, 45, 46, 47),  # More seeds
    overwrite=True
)
```

## 🔄 Migration from Old Notebook

### Before (Old Notebook):
```python
# Scattered code in cells
df_sens_TitForTatCoop = get_sweep_results(
    "results_TitForTatCoop.csv",  # Manual filename
    lambda: TitForTatCooperatePartner(),
    "up",
)
# Results might overwrite accidentally
```

### After (New Structure):
```python
# Clean, managed workflow
manager = SensitivityAnalysisManager()
df_sens_TitForTatCoop = manager.run_analysis(
    partner_name="TitForTatCoop",
    partner_factory=lambda: TitForTatCooperatePartner(),
    threshold_direction="up",
    overwrite=False  # Explicitly control
)
# Auto-saved in: results/sensitivity/2025-11-21/TitForTatCoop.csv
```

## 📈 Benefits

| Aspect | Old Approach | New Approach |
|--------|-------------|--------------|
| **Organization** | Scattered CSV files | Dated subdirectories |
| **Overwrites** | Easy to lose data | Explicit control |
| **Reproducibility** | Manual seed tracking | Timestamped results |
| **Code Reuse** | Copy-paste cells | Import modules |
| **Scalability** | Hard to add partners | Easy registry |
| **Maintenance** | Notebook sprawl | Modular codebase |

## 🔧 Configuration

Edit `trust_bayesian_agent_comparison/config.py`:

```python
# Adjust defaults
SENSITIVITY_SEEDS = (42, 43, 44, 45)  # Add more seeds
NUM_ROUNDS = 100  # Longer simulations
LOSS_AVERSION = 2.5  # Different default

# Change result directories
RESULTS_DIR = PROJECT_ROOT / "my_results"
```

## 📚 Next Steps

1. **Move partner classes** from notebook to `partners/` module
2. **Create plotting utilities** in `visualization/` module
3. **Add unit tests** for agents and partners
4. **Implement Monte Carlo manager** similar to sensitivity
5. **Create comparison notebook** using both managers

## 🤝 Contributing

To add a new partner strategy:

1. Add class to `partners/` module
2. Register in `scripts/run_sensitivity.py`
3. Use in notebooks via manager

## 📄 License

TBD
