# Trust-Based vs Bayesian Agent Comparison Study

This repository contains the code for comparing trust-based and Bayesian learning agents in stag hunt games.

## Project Structure

- `notebooks/Trust_updated_focal_stoch_Betray.ipynb` - Primary analysis comparing trust-based (focal) and Bayesian agents
- `trust_bayesian_agent_comparison/` - Python package containing agent implementations
- `data/` - Directory for data files and results
- `pyproject.toml` - Project configuration and dependencies
- `poetry.lock` - Locked dependencies for reproducible installs

## Overview

This study implements and compares two types of learning agents:

1. **Trust-Based Agent (Focal Agent)**: Uses a dual-state model with signal (x) and trust (t) that incorporates asymmetric penalties for betrayal and surprise
2. **Bayesian Agent**: Uses Beta-Bernoulli conjugate prior updating

Both agents play repeated stag hunt games against various partner strategies including:
- Fixed strategies (Always Cooperate, Always Defect, Random)
- Reactive strategies (Tit-for-Tat variants)
- Adaptive strategies (Belief-driven partners)
- Cheating strategies (Periodic, Strategic, Single-cycle)

## Key Features

- Stochastic decision-making with logit choice functions
- Multiple partner types with different behavioral patterns
- Asymmetric trust updating with loss aversion
- Comparative performance analysis

## Installation

This project uses Poetry for dependency management. To install:

```bash
poetry install
```

## Requirements

The main dependencies are:
- numpy
- pandas
- matplotlib
- joblib

## Authors

Mobin Zarreh
## License

TBD
