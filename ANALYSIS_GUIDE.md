# Monte Carlo Analysis Guide

## Overview

This guide describes all the comprehensive visualizations and analyses available for the Trust-Based vs. Bayesian agent comparison study.

## Quick Start

### Demo Mode (Fast Testing)
```bash
python run_demo.py
```
- 5 representative partners
- 10 Monte Carlo runs per partner
- Results in `results/figures/demo/`
- Takes ~5 minutes

### Full Study (Publication Quality)
```bash
python run_full_study.py
```
- 13 partners (all types)
- 300 Monte Carlo runs per partner
- Results in `results/figures/` and `results/`
- Takes ~30-60 minutes

## Generated Visualizations

### 1. Trust-Based Agent Analysis

#### A. Trust Evolution (`focal_trust_evolution.png`)
- **Description**: Evolution of trust level (t) over rounds for focal agent
- **Shows**: Mean trust ± std across all Monte Carlo runs
- **Overlays**: All partner types on same plot
- **Use for**: Understanding how trust develops with different partners

#### B. Signal Evolution (`focal_signal_evolution.png`)
- **Description**: Evolution of signal (x) over rounds for focal agent
- **Shows**: Mean signal ± std across all Monte Carlo runs
- **Overlays**: All partner types + decision threshold (0.5)
- **Use for**: Understanding how beliefs about partner cooperation evolve

#### C. Expected Cooperation Probability (`focal_expected_p_evolution.png`)
- **Description**: Evolution of E[p] derived from Beta(α, β) where α=εt+xt, β=εt+(1-x)t
- **Shows**: Mean E[p] ± std across all Monte Carlo runs
- **Overlays**: All partner types + decision threshold (~0.667)
- **Use for**: Understanding decision-making basis across partners

#### D. Final Round Distributions (`focal_final_distributions.png`)
- **Description**: Kernel density estimate of E[p] in final round
- **Shows**: Distribution of final beliefs across all runs
- **Overlays**: All partner types + decision threshold
- **Use for**: Understanding convergence and final states

#### E. Decision-Making Trends (`focal_final_decisions.csv`)
- **Columns**: Partner, Mean E[p], Cooperation Rate, Choice
- **Shows**: Last round behavior summary
- **Use for**: Quick summary of final decisions

### 2. Bayesian Agent Analysis

#### A. Expected Cooperation Probability (`bayesian_expected_p_evolution.png`)
- **Description**: Evolution of E[p] = α/(α+β) for Bayesian agent
- **Shows**: Mean E[p] ± std across all Monte Carlo runs
- **Overlays**: All partner types + decision threshold
- **Use for**: Comparing belief evolution with focal agent

#### B. Final Round Distributions (`bayesian_final_distributions.png`)
- **Description**: Beta distribution PDFs with final (α, β) parameters
- **Shows**: Posterior distributions for each partner
- **Overlays**: All partner types + decision threshold + posterior means
- **Use for**: Understanding Bayesian convergence patterns

#### C. Decision-Making Trends (`bayesian_final_decisions.csv`)
- **Columns**: Partner, Mean E[p], Cooperation Rate, Choice
- **Shows**: Last round behavior summary
- **Use for**: Quick summary of final decisions

### 3. Comparative Analysis

#### A. Mutual Cooperation Rate (`mutual_cooperation_comparison.csv`)
- **Columns**: 
  - Partner name
  - Focal/Bayesian mean ± std
  - Difference (Focal - Bayesian)
  - p-value (paired t-test)
  - Significance level (***/** /*/ ns)
- **Shows**: Rate of (C, C) outcomes
- **Use for**: Identifying which agent achieves more mutual cooperation

#### B. Failed Collaboration Rate (`failed_collaboration_comparison.csv`)
- **Columns**: Same structure as mutual cooperation
- **Shows**: Rate of (Agent=C, Partner=D) - agent gets betrayed
- **Use for**: Identifying which agent is more vulnerable to exploitation

#### C. Total Payoff Comparison (`total_payoff_comparison.csv`)
- **Columns**: Same structure + "% Focal Wins"
- **Shows**: Total game payoff across all rounds
- **Use for**: Overall performance comparison
- **Summary**: Counts of wins/losses/ties at bottom

## Interpretation Guidelines

### Trust vs. Signal
- **Trust (t)**: Confidence level in beliefs (higher = more certain)
- **Signal (x)**: Belief about cooperation probability
- **Relationship**: E[p] ≈ x for focal agent (transformed through Beta distribution)

### Statistical Significance
- `***`: p < 0.001 (highly significant)
- `**`: p < 0.01 (very significant)  
- `*`: p < 0.05 (significant)
- `ns`: Not significant

### Decision Threshold
- **Focal agent**: Threshold ≈ 0.667 (computed from game parameters)
- **Bayesian agent**: Same threshold applied to E[p]
- Cooperate if E[p] > threshold, defect otherwise

## Partner Types

### Fixed Strategies
1. **AlwaysCooperate**: Always cooperates
2. **AlwaysDefect**: Always defects
3. **Random**: Random 50-50
4. **PeriodicCheater**: Cycles of cooperation/defection
5. **SingleCycle**: Switches from cooperate to defect once
6. **GradualDeterioration**: Cooperation probability decays exponentially

### Reactive Strategies
7. **TitForTat**: Copies agent's previous action (start cooperate)
8. **SuspiciousTitForTat**: Copies agent's previous action (start defect)
9. **GrimTrigger**: Cooperates until first defection, then always defects
10. **Pavlov**: Win-stay, lose-shift strategy

### Belief-Driven Strategies
11. **Adaptive**: Mirrors predicted agent behavior
12. **StrategicCheater**: Exploits when agent's trust is high
13. **ExpectationViolation**: Violates agent's expectations strategically

## File Structure

```
results/
├── figures/                           # Main visualizations
│   ├── focal_trust_evolution.png
│   ├── focal_signal_evolution.png
│   ├── focal_expected_p_evolution.png
│   ├── focal_final_distributions.png
│   ├── bayesian_expected_p_evolution.png
│   ├── bayesian_final_distributions.png
│   ├── payoff_comparison.png          # Basic bar charts
│   ├── cooperation_rates.png
│   ├── payoff_advantage.png
│   └── belief_trajectories.png
│   
├── focal_final_decisions.csv         # Last round decisions
├── bayesian_final_decisions.csv
├── mutual_cooperation_comparison.csv  # Comparison tables
├── failed_collaboration_comparison.csv
├── total_payoff_comparison.csv
├── summary_statistics.csv             # Basic summary
│
└── monte_carlo/                       # Raw data
    └── YYYY-MM-DD/
        ├── FocalAgent_vs_Partner/
        └── BayesianFocalAgent_vs_Partner/
```

## Customization

### Modify Monte Carlo Runs
Edit `trust_bayesian_agent_comparison/config.py`:
```python
NUM_MONTE_CARLO_RUNS = 300  # Change to 100, 500, etc.
```

### Change Partners
Edit `run_full_study.py` or `run_demo.py`:
```python
PARTNERS = {
    'MyPartner': lambda: MyPartnerClass(param=value),
    # Add/remove partners here
}
```

### Adjust Visualizer
Edit `trust_bayesian_agent_comparison/visualization/__init__.py`:
- Color schemes: Line 28 `sns.color_palette()`
- Figure sizes: Various `figsize=(width, height)`
- Statistical tests: Methods in `create_*_table()`

## Tips for Analysis

1. **Start with demo**: Run `run_demo.py` to verify everything works
2. **Check console output**: Tables are printed during execution
3. **Review significance**: Focus on statistically significant differences
4. **Compare evolution plots**: Look for divergence patterns
5. **Examine final distributions**: Check for multi-modal or skewed distributions
6. **Cross-reference tables**: Compare mutual cooperation with total payoffs

## Common Questions

### Q: Why does Bayesian agent win more often?
A: Check the trust evolution and signal evolution plots. Bayesian agent may converge faster to optimal beliefs due to simpler update rule.

### Q: Which visualization shows decision timing?
A: Expected cooperation probability evolution shows when agent crosses decision threshold.

### Q: How to identify exploitation patterns?
A: Compare failed collaboration rates. High values indicate agent cooperates while partner defects.

### Q: What if results vary between runs?
A: Increase `NUM_MONTE_CARLO_RUNS` for more stable estimates. Check standard deviations in tables.

## Citation

If using these analyses in research, please cite both the original stag-hunt model and this implementation:
- Original model: [Add reference]
- This implementation: [Add reference]

## Contact

For questions or issues:
- Email: mzarreh@asu.edu
- Issues: [GitHub repository URL]
