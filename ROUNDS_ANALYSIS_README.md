# Rounds Analysis: Effect of Simulation Length on Agent Performance

## Overview

This analysis examines how the **number of rounds** (game length) affects the performance of Trust-based vs Bayesian agents across different partner types. This is critical for demonstrating when and why trust-based learning has advantages over Bayesian approaches.

## Key Questions

1. **Does Trust-based agent show stronger advantages with more rounds?**
   - Hypothesis: Yes, because more rounds = more opportunity for non-stationary patterns
   
2. **How quickly do agents adapt to different partner types?**
   - Measured by first-quarter performance (first 25% of rounds)
   
3. **At what round count do we see significant divergence?**
   - Critical for determining optimal experiment length
   
4. **Does Bayesian agent suffer more from non-stationarity over time?**
   - Hypothesis: Yes, cumulative effect of treating all observations equally

## Rounds Tested

- **10 rounds**: Very short-term, initial adaptation
- **50 rounds**: Short-term, rapid learning phase
- **70 rounds**: Early stabilization
- **100 rounds**: Standard experiment length (current baseline)
- **200 rounds**: Medium-term patterns
- **300 rounds**: Long-term trends
- **500 rounds**: Extended interaction
- **1000 rounds**: Very long-term convergence

## Partner Categories

### 1. Belief-Driven Partners (Non-Stationary)
These partners maintain beliefs about the agent and adapt strategically:
- **AdaptivePartner**: Mirrors agent's trust dynamics
- **StrategicCheaterPartner**: Exploits when trust is high
- **ExpectationViolationPartner**: Violates established patterns

**Expected Result**: Trust-based should show **strong advantages** that **increase with rounds**

### 2. Reactive Partners (Conditional but Stationary)
These partners respond to agent actions following fixed rules:
- **TitForTatCooperatePartner**: Mirrors last action (starts cooperative)
- **TitForTatDefectPartner**: Mirrors last action (starts suspicious)
- **GrimTriggerPartner**: Cooperates until first defection, then always defects
- **PavlovPartner**: Win-stay, lose-shift strategy

**Expected Result**: Both agents should perform **similarly** (strategies are stationary)

### 3. Fixed Partners (Baseline)
These partners follow predetermined patterns:
- **AlwaysCooperatePartner**: Always cooperates
- **AlwaysDefectPartner**: Always defects
- **PeriodicCheaterPartner**: Cycles between cooperation and defection
- **SingleCyclePartner**: Cooperates for first 40% of rounds, then defects (scales with round count)
- **GradualDeteriorationPartner**: Linear decline in cooperation (scales with round count)

**Expected Result**: Trust-based shows **moderate advantages** on non-stationary fixed partners

**Note**: SingleCyclePartner and GradualDeteriorationPartner are now **round-adaptive** - they adjust their behavior based on the total number of rounds to maintain consistent patterns across different simulation lengths.

## Usage

### Option 1: Quick Interactive Exploration (Recommended for Initial Testing)

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/Rounds_Analysis.ipynb
```

This notebook allows you to:
- Test individual partners with different round counts
- Visualize results in real-time
- Adjust parameters interactively
- Start with fewer runs (10-20) for quick testing

### Option 2: Full Comprehensive Analysis (For Publication Results)

Run the complete analysis script:

```bash
python run_rounds_analysis.py
```

This will:
1. Test **all partner types** across **all round counts**
2. Run **100 Monte Carlo simulations** per configuration
3. Generate comprehensive visualizations
4. Create statistical comparison tables
5. Save all results to CSV files

**Estimated Runtime**: 30-60 minutes (depending on hardware)

## Output Files

### Results Directory: `results/rounds_analysis/`

- `all_rounds_analysis.csv`: Combined results for all configurations
- `belief_driven_results.csv`: Belief-driven partners only
- `reactive_results.csv`: Reactive partners only
- `fixed_results.csv`: Fixed partners only

### Figures Directory: `results/figures/rounds_analysis/`

#### Summary Visualizations
- `summary_advantage_heatmap.png`: Payoff advantage across all partners and rounds
- `summary_win_rates.png`: Trust-based win rate vs Bayesian across rounds

#### Belief-Driven Partners
- `belief_driven/belief_driven_payoff_vs_rounds.png`: Total payoff comparison
- `belief_driven/belief_driven_cooperation_vs_rounds.png`: Mutual cooperation rates
- `belief_driven/belief_driven_betrayal_vs_rounds.png`: Betrayal rates
- `belief_driven/belief_driven_adaptation_speed.png`: First-quarter performance
- `belief_driven/belief_driven_comparison_table.csv`: Statistical comparison

#### Reactive Partners
- `reactive/reactive_payoff_vs_rounds.png`
- `reactive/reactive_cooperation_vs_rounds.png`
- `reactive/reactive_betrayal_vs_rounds.png`
- `reactive/reactive_adaptation_speed.png`
- `reactive/reactive_comparison_table.csv`

#### Fixed Partners
- `fixed/fixed_payoff_vs_rounds.png`
- `fixed/fixed_cooperation_vs_rounds.png`
- `fixed/fixed_betrayal_vs_rounds.png`
- `fixed/fixed_adaptation_speed.png`
- `fixed/fixed_comparison_table.csv`

## Key Metrics Reported

### Primary Metrics
1. **Total Payoff**: Cumulative payoff over all rounds
2. **Mutual Cooperation Rate**: % of rounds with successful collaboration
3. **Betrayal Rate**: % of rounds where agent cooperated but partner defected
4. **Payoff Advantage**: Trust payoff - Bayesian payoff

### Secondary Metrics
5. **First Quarter Payoff**: Mean payoff in first 25% of rounds (adaptation speed)
6. **Final Quarter Payoff**: Mean payoff in last 25% of rounds (convergence)
7. **Cooperation Rate**: % of rounds agent chose to cooperate
8. **Payoff Variance**: Stability of payoffs
9. **Win Rate**: % of runs where Trust > Bayesian

### For Trust-Based Agent (Additional)
10. **Final Trust Level**: Trust state at end of simulation
11. **Mean Trust**: Average trust throughout simulation
12. **Trust Variance**: Stability of trust dynamics

## Expected Findings for Publication

### Hypothesis 1: Trust-based advantages **increase** with more rounds
- **Rationale**: More rounds reveal non-stationary patterns that Bayesian can't handle
- **Evidence**: Payoff advantage should grow (possibly logarithmically) with rounds
- **Partners**: Strongest effect with belief-driven partners

### Hypothesis 2: Bayesian suffers from **cumulative stationarity violations**
- **Rationale**: Beta prior accumulates all observations equally (no recency weighting)
- **Evidence**: Betrayal rate increases with rounds for Bayesian
- **Partners**: Most evident with GradualDeterioration, ExpectationViolation

### Hypothesis 3: Trust-based shows **faster adaptation**
- **Rationale**: EWMA + trust discounting = faster response to changes
- **Evidence**: Higher first-quarter payoffs for trust-based
- **Partners**: All non-stationary partners

### Hypothesis 4: Reactive partners show **minimal difference**
- **Rationale**: TitForTat, Pavlov are stationary conditional strategies
- **Evidence**: Similar performance for both agents (except GrimTrigger)
- **Partners**: TitForTat, Pavlov converge to similar outcomes

## Statistical Significance

The script automatically computes:
- **Mean and standard deviation** for each metric
- **Win rates**: % of Monte Carlo runs where Trust > Bayesian
- **Effect sizes**: Magnitude of differences

For publication, you should additionally report:
- **Wilcoxon signed-rank test**: p-values for paired comparisons
- **Cohen's d**: Standardized effect size
- **Confidence intervals**: 95% CI for advantages

See `notebooks/Rounds_Analysis.ipynb` for interactive statistical testing.

## Customization

### Adjust Round Counts

Edit `run_rounds_analysis.py`:

```python
ROUNDS_LIST = [10, 50, 70, 100, 200, 300, 500, 1000]  # Modify as needed
```

### Adjust Number of Runs

For quick testing:
```python
NUM_RUNS = 20  # Fast, less reliable
```

For publication:
```python
NUM_RUNS = 100  # Robust, publication-quality
NUM_RUNS = 500  # Very robust, computationally expensive
```

### Add Custom Partners

Add to the partner dictionaries in `run_rounds_analysis.py`:

```python
BELIEF_DRIVEN_PARTNERS = {
    'MyCustomPartner': lambda: MyCustomPartner(param=value),
    # ... existing partners
}
```

## Parallel Execution

The script uses all available CPU cores by default. To limit:

```python
# Add to script or modify config
from joblib import Parallel, delayed
# Change n_jobs parameter in parallel sections
```

## Interpreting Results

### Good Signs for Publication
1. **Payoff advantage grows with rounds** (especially for belief-driven)
2. **Betrayal reduction increases with rounds** (Trust-based learns to avoid exploitation)
3. **Faster adaptation** (higher first-quarter payoffs for Trust-based)
4. **Win rates > 60%** for non-stationary partners
5. **Similar performance** on reactive partners (shows both are competent)

### Red Flags
1. **No clear advantage** at any round count → Parameter tuning needed
2. **Advantage decreases with rounds** → Trust mechanism may be unstable
3. **High variance** → Need more Monte Carlo runs or more stable parameters
4. **Bayesian wins on belief-driven** → Core hypothesis violated, major revision needed

## Next Steps After Analysis

1. **Identify optimal round count** for your experiments (where differences are most pronounced)
2. **Focus on partners** where Trust-based shows strongest advantages
3. **Create publication figures** using the generated plots
4. **Report statistical significance** with proper tests
5. **Write discussion** explaining why certain patterns emerge

## Publication Framing

### Key Message
"Trust-based learning demonstrates **increasing advantages** over Bayesian approaches as interaction length grows, particularly in **non-stationary environments** where partner behavior evolves over time. This advantage stems from the trust mechanism's ability to **weight recent observations more heavily** and **adapt beliefs dynamically**, overcoming the Bayesian assumption of **stationarity**."

### Evidence to Highlight
1. Payoff advantage vs rounds (Figure: heatmap)
2. Betrayal rate reduction with belief-driven partners
3. Faster adaptation (first-quarter comparison)
4. Similar performance on stationary partners (shows competence baseline)

## Questions?

Check the Jupyter notebook for interactive examples and additional analysis options.
