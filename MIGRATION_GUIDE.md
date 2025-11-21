# Migration Guide: Old Notebook → Refactored Structure

## Overview

This guide shows how to migrate from the old monolithic notebook to the new modular structure.

## What Changed?

### Old Structure (Monolithic)
```
notebooks/
└── Trust_updated_focal_stoch_Betray.ipynb  (4091 lines, 65 cells)
    - Cell definitions of agents/partners
    - Inline simulation functions
    - Hardcoded paths and filenames
    - Risk of accidental overwrites
    - Difficult to reuse code
```

### New Structure (Modular)
```
trust_bayesian_agent_comparison/
├── agents/           # Agent implementations
├── partners/         # Partner strategies
├── simulation/       # Simulation engine
├── analysis/         # Analysis tools
│   ├── sensitivity.py
│   ├── monte_carlo.py
│   └── metrics.py
├── visualization/    # (TODO) Plotting utilities
└── config.py         # Centralized configuration

notebooks/            # Clean analysis notebooks
scripts/              # CLI tools
results/              # Auto-organized outputs
```

## Migration Examples

### 1. Creating Agents

#### Before (in notebook):
```python
# Cell 3: Define FocalAgent class (80+ lines)
class FocalAgent:
    def __init__(self, loss_aversion=2.0, mu=0.5, ...):
        # ... implementation ...
```

#### After (import from module):
```python
from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent

trust_agent = FocalAgent(loss_aversion=2.0, mu=0.5)
bayesian_agent = BayesianFocalAgent(alpha_0=0.5, beta_0=0.5)
```

### 2. Creating Partners

#### Before:
```python
# Cell 4: Define TitForTat (20 lines)
class TitForTatCooperatePartner:
    # ... implementation ...
```

#### After:
```python
from trust_bayesian_agent_comparison.partners import (
    TitForTatCooperatePartner,
    AlwaysDefectPartner,
    StrategicCheaterPartner,
)

partner = TitForTatCooperatePartner()
```

### 3. Running Simulations

#### Before:
```python
# Cell 6: run_agent_simulation function (50+ lines)
def run_agent_simulation(agent, partner, num_rounds, seed):
    # ... implementation ...
    
df = run_agent_simulation(trust_agent, partner, 70, 42)
```

#### After:
```python
from trust_bayesian_agent_comparison.simulation import run_agent_simulation

df = run_agent_simulation(trust_agent, partner, num_rounds=70, seed=42)
```

### 4. Sensitivity Analysis

#### Before (Cell 29):
```python
# Manual file naming, risk of overwrites
def sweep_learning_params(...):
    # ... 100+ lines ...
    
df_sens = get_sweep_results(
    "results_TitForTatCoop.csv",  # Manual filename
    lambda: TitForTatCooperatePartner(),
    "up",
)
# Might overwrite existing results accidentally!
```

#### After:
```python
from trust_bayesian_agent_comparison.analysis import SensitivityAnalysisManager

manager = SensitivityAnalysisManager()
df_sens = manager.run_analysis(
    partner_name="TitForTatCoop",
    partner_factory=lambda: TitForTatCooperatePartner(),
    threshold_direction="up",
    overwrite=False,  # Explicit control
)
# Auto-saved in: results/sensitivity/2025-11-21/TitForTatCoop.csv
```

### 5. Computing Metrics

#### Before (scattered in cells):
```python
# Cell 15: Define agent_coop_rate
def agent_coop_rate(df):
    return df['agent_action'].mean()

# Cell 16: Define mutual_coop_rate
def mutual_coop_rate(df):
    # ...
    
# Cell 17: Define betrayal_rate
# ... etc.
```

#### After (single import):
```python
from trust_bayesian_agent_comparison.analysis import (
    agent_coop_rate,
    mutual_coop_rate,
    betrayal_rate,
    compute_strategy_statistics,
)

stats = compute_strategy_statistics(df)  # All metrics at once
```

### 6. Configuration

#### Before (scattered constants):
```python
# Cell 1
TRUST_MAX = 10.0
LOSS_AVERSION = 2.0

# Cell 20
NUM_ROUNDS = 70

# Cell 29
seeds = (42, 43, 44)

# Cell 35
ETA_GRID = np.linspace(0.0, 1.0, 6)
```

#### After (centralized):
```python
from trust_bayesian_agent_comparison.config import (
    TRUST_MAX,
    LOSS_AVERSION,
    NUM_ROUNDS,
    SENSITIVITY_SEEDS,
    ETA_GRID,
)
# All constants in one place!
```

## New Capabilities

### 1. Command-Line Interface

Now you can run experiments without opening notebooks:

```bash
# Run sensitivity for specific partners
python scripts/run_sensitivity.py --partners TitForTatCoop AlwaysDefect

# Run for all partners
python scripts/run_sensitivity.py --all

# Force re-run with more seeds
python scripts/run_sensitivity.py --all --overwrite --seeds 42 43 44 45 46
```

### 2. Automatic Result Organization

Results are now organized by date:

```
results/
├── sensitivity/
│   ├── 2025-11-20/
│   │   ├── TitForTatCoop.csv
│   │   └── AlwaysDefect.csv
│   └── 2025-11-21/
│       ├── TitForTatCoop.csv  # Today's runs
│       └── StrategicCheat.csv
└── monte_carlo/
    └── 2025-11-21/
        ├── TitForTatCoop_agent1.csv
        └── TitForTatCoop_agent2.csv
```

### 3. Overwrite Protection

```python
# First run: computes and saves
manager.run_analysis(..., overwrite=False)  # Takes 10 minutes

# Second run: loads existing
manager.run_analysis(..., overwrite=False)  # Takes 1 second!

# Force recompute
manager.run_analysis(..., overwrite=True)   # Takes 10 minutes
```

### 4. Parallel Batch Processing

```python
# Run multiple partners in one call
partner_configs = [
    ("TitForTatCoop", lambda: TitForTatCooperatePartner(), "up"),
    ("AlwaysDefect", lambda: AlwaysDefectPartner(), "down"),
    ("StrategicCheat", lambda: StrategicCheaterPartner(), "down"),
]

results = manager.run_multiple(
    partner_configs=partner_configs,
    overwrite=False,
)
# Returns: {'TitForTatCoop': df1, 'AlwaysDefect': df2, ...}
```

## Step-by-Step Migration

### Phase 1: Start Using Modules (Keep Old Notebook)

1. Keep your existing notebook as reference
2. Create new notebook that imports from modules
3. Test with one partner/agent combination
4. Verify results match old notebook

### Phase 2: Migrate Analysis Workflows

1. Replace sensitivity analysis cells with `SensitivityAnalysisManager`
2. Replace Monte Carlo cells with `MonteCarloManager`
3. Update plotting to use new result structure

### Phase 3: Clean Up

1. Archive old notebook
2. Delete redundant cells
3. Use new notebooks as primary interface

## File Comparison

### Old vs New: Same Analysis, Less Code

**Old notebook cell count:**
- Agent definitions: 3 cells (~150 lines)
- Partner definitions: 1 cell (~200 lines)
- Simulation logic: 1 cell (~50 lines)
- Sensitivity analysis: 2 cells (~150 lines)
- Metrics: 5 cells (~100 lines)
- Total: **12+ cells, ~650 lines** (just for setup!)

**New notebook cell count:**
```python
# Cell 1: All imports (10 lines)
from trust_bayesian_agent_comparison.agents import FocalAgent
from trust_bayesian_agent_comparison.partners import TitForTatCooperatePartner
from trust_bayesian_agent_comparison.analysis import SensitivityAnalysisManager

# Cell 2: Run analysis (5 lines)
manager = SensitivityAnalysisManager()
results = manager.run_analysis(...)

# Cell 3: Plot results (10 lines)
plt.plot(...)
```

**Total: 3 cells, ~25 lines** for the same functionality!

## Benefits Summary

| Aspect | Old Notebook | New Structure |
|--------|-------------|---------------|
| **Code reuse** | Copy-paste cells | Import modules |
| **Organization** | 4091 lines, 65 cells | Modular, <100 lines per file |
| **Result management** | Manual filenames | Automatic dated folders |
| **Overwrites** | Easy to lose data | Explicit control |
| **Scalability** | Hard to add partners | Registry system |
| **Collaboration** | Merge conflicts | Separate modules |
| **Testing** | Difficult | Unit tests per module |
| **CLI support** | None | Full CLI interface |
| **Reproducibility** | Manual tracking | Timestamped results |

## Common Patterns

### Pattern 1: Quick Single Run
```python
from trust_bayesian_agent_comparison import *

agent = FocalAgent()
partner = TitForTatCooperatePartner()
df = run_agent_simulation(agent, partner)
stats = compute_strategy_statistics(df)
```

### Pattern 2: Parameter Sweep
```python
manager = SensitivityAnalysisManager()
results = manager.run_analysis(
    partner_name="TitForTatCoop",
    partner_factory=lambda: TitForTatCooperatePartner(),
    threshold_direction="up",
    overwrite=False,
)
```

### Pattern 3: Monte Carlo Comparison
```python
mc_manager = MonteCarloManager()
df1, df2 = mc_manager.run_monte_carlo(
    agent1_factory=lambda: FocalAgent(),
    agent2_factory=lambda: BayesianFocalAgent(),
    partner_factory=lambda: TitForTatCooperatePartner(),
    partner_name="TitForTatCoop",
    overwrite=False,
)
```

## Next Steps

1. **Try the quick start notebook**: `00_quick_start_refactored.ipynb`
2. **Run CLI experiments**: `python scripts/run_sensitivity.py --help`
3. **Read the refactoring guide**: `REFACTORING_GUIDE.md`
4. **Explore modules**: Browse `trust_bayesian_agent_comparison/`

## Questions?

- See `REFACTORING_GUIDE.md` for detailed documentation
- Check `00_quick_start_refactored.ipynb` for examples
- Original notebook preserved as `Trust_updated_focal_stoch_Betray.ipynb`
