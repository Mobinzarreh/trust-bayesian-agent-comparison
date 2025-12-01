# Trust-Based vs Bayesian Agent Comparison Study

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive computational study comparing trust-based and Bayesian learning agents in repeated stag hunt games, with extensive sensitivity analysis of agent parameters.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Analysis](#detailed-analysis)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Results Organization](#results-organization)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Authors](#authors)

## 🎯 Overview

This study implements and compares two types of learning agents in repeated stag hunt games:

### 🤝 Trust-Based Agent (Focal Agent)
- **Dual-state model**: Signal (x) and Trust (t) with asymmetric penalties
- **Loss aversion**: Different weights for betrayal vs. surprise
- **Adaptive learning**: Updates beliefs based on trust dynamics
- **Stochastic decisions**: Logit choice with inverse temperature parameter

### 🎲 Bayesian Agent
- **Conjugate prior**: Beta-Bernoulli updating
- **Probabilistic reasoning**: Updates beliefs using Bayes' theorem
- **Memory-less**: Each observation weighted equally

## 🏆 Key Findings

### Agent Performance Comparison
- **Trust-based agent excels** against non-stationary partners (SingleCycle, GradualDeterioration, ExpectationViolation)
- **Bayesian agent performs better** against stationary partners (TitForTat, GrimTrigger)
- **Trust-based agent shows advantage** with increased interaction rounds

### Parameter Sensitivity Results
Complete sensitivity analysis of 7 parameters across 5 partner types:
- **eta** (learning rate): [0.1, 0.3, 0.5, 0.7, 0.9]
- **memory_discount**: [0.5, 0.7, 0.8, 0.9, 0.95]
- **trust_discount**: [0.5, 0.7, 0.8, 0.9, 0.95]
- **trust_smoothing**: [0.1, 0.2, 0.3, 0.5, 0.7]
- **loss_aversion**: [1.0, 1.5, 2.0, 3.0, 5.0]
- **lambda_surprise**: [0.0, 0.25, 0.5, 0.75, 1.0]
- **inverse_temperature**: [0.5, 2.0, 5.0, 10.0]

## 📁 Project Structure

```
trust-bayesian-agent-comparison/
├── 📓 notebooks/                    # Jupyter notebooks
│   ├── Parameter_Sensitivity_Analysis.ipynb    # ⭐ Main sensitivity analysis
│   ├── Trust_Agent_Excellence_Analysis.ipynb   # Winning partners analysis
│   ├── Rounds_Analysis.ipynb                   # Rounds effect analysis
│   ├── Trust_updated_focal_stoch_Betray.ipynb  # Original implementation
│   └── 00_quick_start_refactored.ipynb         # Tutorial
├── 📜 scripts/                      # Analysis scripts
│   ├── run_all_sensitivity.py       # ⭐ Run all 7 parameters
│   ├── run_inverse_temp_sensitivity.py        # Quick inverse temp test
│   ├── run_demo.py                  # Fast demo (10 runs)
│   ├── run_full_study.py            # Production study (300 runs)
│   ├── run_rounds_analysis.py       # Rounds effect analysis
│   └── view_all_tables.py           # Results viewer
├── 📊 results/                      # Analysis results
│   ├── experiments/                 # Individual experiment CSVs
│   ├── summaries/                   # Summary comparison tables
│   ├── sensitivity_*/               # Parameter sensitivity results
│   ├── figures/                     # Generated visualizations
│   └── rounds_analysis/             # Rounds effect analysis
├── 🏗️ trust_bayesian_agent_comparison/  # Source code
│   ├── agents/                      # Agent implementations
│   ├── partners/                    # Partner strategy implementations
│   ├── simulation/                  # Game simulation engine
│   ├── analysis/                    # Analysis and metrics
│   └── visualization/               # Plotting utilities
├── 📚 Documentation
│   ├── ANALYSIS_GUIDE.md            # How to interpret results
│   ├── ROUNDS_ANALYSIS_README.md    # Rounds analysis guide
│   └── CLEANUP_SUMMARY.txt          # Project cleanup history
└── ⚙️ Configuration
    ├── pyproject.toml               # Project dependencies
    ├── poetry.lock                  # Locked dependencies
    └── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### Option 1: Demo Mode (Fast, 5 minutes)
```bash
# Run quick demo with 5 partners, 10 Monte Carlo runs each
python run_demo.py
```

### Option 2: Full Study (Complete, 30-60 minutes)
```bash
# Run comprehensive study with 13 partners, 300 Monte Carlo runs each
python run_full_study.py
```

### Option 3: Sensitivity Analysis (Complete parameter sweep)
```bash
# Run sensitivity analysis for all 7 parameters
python scripts/run_all_sensitivity.py
```

## 📊 Detailed Analysis

### Monte Carlo Comparison
- **300 runs** per agent-partner combination
- **13 partner types**: Fixed, reactive, adaptive, and cheating strategies
- **3 metrics**: Mutual cooperation rate, betrayal rate, total payoff

### Rounds Effect Analysis
- **7 round counts**: 10, 50, 70, 100, 200, 300, 500, 1000
- **3 partner categories**: Belief-driven, Fixed, Reactive
- **Performance tracking**: Adaptation speed, convergence stability

### Parameter Sensitivity Analysis
- **7 parameters** varied independently
- **5 representative partners**: SingleCycle, GradualDeterioration, ExpectationViolation, StrategicCheater, TitForTat
- **Optimal parameter identification** for each metric and partner

## 🔬 Sensitivity Analysis

### Parameters Analyzed

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| **eta** | Learning rate for signal updates | [0.1, 0.9] | 0.3 |
| **memory_discount** | Recency weighting for signals | [0.5, 0.95] | 0.9 |
| **trust_discount** | Recency weighting for trust | [0.5, 0.95] | 0.8 |
| **trust_smoothing** | Trust update smoothing factor | [0.1, 0.7] | 0.2 |
| **loss_aversion** | Betrayal penalty multiplier (λ) | [1.0, 5.0] | 2.0 |
| **lambda_surprise** | Surprise penalty multiplier (μ) | [0.0, 1.0] | 0.5 |
| **inverse_temperature** | Exploration-exploitation balance | [0.5, 10.0] | 2.0 |

### Key Insights
- **Trust-based agent** shows robust performance across parameter ranges
- **Parameter interactions** exist but single-parameter optimization provides good baselines
- **Partner-specific optimization** often yields better results than universal settings

## 📁 Results Organization

```
results/
├── experiments/           # Raw experiment data (26 CSV files)
│   ├── Adaptive_agent1.csv
│   ├── SingleCycle_agent2.csv
│   └── ... (all agent-partner combinations)
├── summaries/             # Aggregated comparison tables
│   ├── mutual_cooperation_comparison.csv
│   ├── total_payoff_comparison.csv
│   └── summary_statistics.csv
├── sensitivity_*/         # Parameter sensitivity results
│   ├── eta/ExpectationViolation_results.csv
│   ├── memory_discount/SingleCycle_results.csv
│   └── ... (all parameter-partner combinations)
├── figures/               # Generated visualizations
│   ├── focal_signal_evolution.png
│   ├── payoff_comparison.png
│   └── rounds_analysis/
└── rounds_analysis/       # Rounds effect analysis
    ├── all_rounds_analysis.csv
    └── belief_driven_results.csv
```

## 💻 Installation

### Prerequisites
- Python 3.8+
- Poetry (recommended) or pip

### Using Poetry (Recommended)
```bash
# Clone repository
git clone https://github.com/Mobinzarreh/trust-bayesian-agent-comparison.git
cd trust-bayesian-agent-comparison

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scipy joblib
```

## 🎮 Usage

### Running Analyses

```bash
# Quick demo (5 minutes)
python run_demo.py

# Full production study (30-60 minutes)
python run_full_study.py

# Sensitivity analysis for all parameters (15-20 minutes)
python scripts/run_all_sensitivity.py

# Analyze effect of interaction rounds (10-15 minutes)
python run_rounds_analysis.py

# View all comparison tables
python view_all_tables.py
```

### Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open main notebooks:
# - notebooks/Parameter_Sensitivity_Analysis.ipynb (⭐ Main analysis)
# - notebooks/Trust_Agent_Excellence_Analysis.ipynb
# - notebooks/Rounds_Analysis.ipynb
```

## 🔬 Methodology

### Game Setup
- **Stag Hunt Game**: Payoff matrix with coordination incentives
- **Repeated interactions**: 100 rounds per simulation
- **Stochastic decisions**: Logit choice with inverse temperature
- **Monte Carlo analysis**: Multiple random seeds for robustness

### Agent Models

#### Trust-Based Agent
```
Signal Update: x ← η * x + (1-η) * observed_action
Trust Update: t ← trust_discount * t + trust_smoothing * (signal - t)
Decision: P(cooperate) = 1 / (1 + exp(-(expected_utility * inverse_temperature)))
```

#### Bayesian Agent
```
Prior: Beta(α, β)
Likelihood: Bernoulli(p)
Posterior: Beta(α + successes, β + failures)
Decision: P(cooperate) = 1 / (1 + exp(-(expected_p * inverse_temperature)))
```

### Partner Strategies
- **Fixed**: AlwaysCooperate, AlwaysDefect, Random
- **Reactive**: TitForTat, GrimTrigger, Pavlov
- **Adaptive**: Belief-driven partners that learn opponent behavior
- **Cheating**: SingleCycle, StrategicCheater, PeriodicCheater

## 📈 Results Summary

### Performance Comparison
```
Mutual Cooperation Rate:
Trust-based: 0.72 ± 0.15    Bayesian: 0.68 ± 0.18

Betrayal Rate:
Trust-based: 0.18 ± 0.12    Bayesian: 0.22 ± 0.14

Total Payoff:
Trust-based: 285 ± 45       Bayesian: 275 ± 52
```

### Optimal Parameters (Example for SingleCycle partner)
- **eta**: 0.7 (best for mutual cooperation)
- **memory_discount**: 0.95 (best for payoff)
- **inverse_temperature**: 2.0 (balanced performance)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research advisor for guidance on trust modeling
- Open-source community for scientific computing tools
- Contributors to the stag hunt game theory literature

## 📧 Contact

**Mobin Zarreh**
- GitHub: [@Mobinzarreh](https://github.com/Mobinzarreh)
- Email: [mobin.zarreh@asu.edu]

---

**⭐ Star this repository** if you find it useful for your research on trust modeling, reinforcement learning, or game theory!
## License

TBD
