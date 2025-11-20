# Trust-Based vs Bayesian Agent Comparison Study

This repository contains the code for comparing trust-based and Bayesian learning agents in stag hunt games.

## Main Notebook

- `Trust_updated_focal_stoch_Betray.ipynb` - Primary analysis comparing trust-based (focal) and Bayesian agents

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

## Requirements

```python
numpy
pandas
matplotlib
joblib
```

## Authors

Mobin Zarreh
## License

TBD
