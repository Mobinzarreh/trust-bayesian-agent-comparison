# Agent Implementation Update - Alignment with Original

## Summary of Changes

### 1. BaseAgent (base.py)
**Changes:**
- Added `stochastic` and `inv_temp` parameters to `__init__`
- Added abstract `expected_p()` method (replaces direct `get_belief()`)
- Added `get_belief()` as alias to `expected_p()` for backwards compatibility
- Moved `decide()` logic from subclasses to base class (shared implementation)
- Changed parameter name from `partner_action` to `partner_choice` in `update()`

**Original Logic:**
```python
class BaseAgent:
    def __init__(self, stochastic: bool = True, inv_temp: float = 2.0)
    def expected_p(self) -> float  # Abstract method
    def decide(self) -> int  # Uses expected_p() with logit or threshold
    def update(self, partner_choice: int)  # Abstract method
```

### 2. FocalAgent (focal_agent.py)
**MAJOR REWRITE - Previously Oversimplified**

**New Parameters (matching original):**
- `u_i`: Initial signal value (defaults to 1 - decision_threshold)
- `t_init`: Initial trust level
- `eta`: Learning rate for signal update
- `noise_sigma`: Gaussian noise std for exploration
- `memory_discount`: Discount factor for signal history
- `trust_discount`: Discount factor for trust history  
- `trust_smoothing`: Smoothing factor for trust convergence
- `loss_aversion`: λ ≥ 1, betrayal penalty multiplier
- `lambda_surprise`: μ ∈ [0,λ], surprise penalty weight

**New Implementation:**
```python
def expected_p(self) -> float:
    """Beta distribution parameterization."""
    alpha = EPS + self.x * self.t
    beta = EPS + (1 - self.x) * self.t
    return alpha / (alpha + beta)

def _compute_new_trust(self, partner_choice: int) -> float:
    """
    Asymmetric trust update with history tracking:
    - match: expected == observed
    - betrayal: expected C, saw D (penalty λ)
    - surprise: expected D, saw C (penalty μ)
    
    Uses exponential discounting on history lists.
    Computes: consistency = WM / (WM + λ*WB + μ*WU)
    """
    
def _compute_new_signal(self, partner_choice: int) -> float:
    """
    Recency-weighted EWMA with exponential discounting.
    Adds Gaussian noise for exploration.
    """
```

**History Tracking:**
- `trust_match_hist`: Records whether expectation matched observation
- `trust_betrayal_hist`: Records betrayals (expected C, saw D)
- `trust_surprise_hist`: Records surprises (expected D, saw C)
- `action_history`: Partner actions for EWMA calculation

**Removed (oversimplified):**
- Simple delta-based updates
- Direct parameter η_x and η_t
- Basic asymmetry parameter μ
- Missing complex trust dynamics

### 3. BayesianFocalAgent (bayesian_agent.py)
**Changes:**
- Changed parameter names: `alpha_0` → `alpha0`, `beta_0` → `beta0` (match original)
- Removed unused `numpy` import
- Changed `partner_action` → `partner_choice` in `update()`
- Added `posterior_mean()` method for backwards compatibility
- Update logic: `self.alpha += partner_choice; self.beta += (1 - partner_choice)`

**Original Logic Preserved:**
```python
def expected_p(self) -> float:
    return self.alpha / (self.alpha + self.beta)

def update(self, partner_choice: int):
    self.alpha += partner_choice
    self.beta += (1 - partner_choice)
```

## Verification

### Test Results
✅ FocalAgent:
- Initializes with all 11 parameters
- `expected_p()` uses Beta distribution
- Updates trust with asymmetric penalties
- Updates signal with EWMA and noise
- Tracks match/betrayal/surprise history

✅ BayesianFocalAgent:
- Initializes with alpha0, beta0
- `expected_p()` returns posterior mean
- Updates via Bayesian conjugate prior
- `posterior_mean()` alias works

✅ Shared Logic:
- Both use `BaseAgent.decide()` with logit/threshold
- Both override `expected_p()`
- Both override `update()`

### Key Differences from Previous (Oversimplified) Version

| Aspect | Old (Wrong) | New (Correct) |
|--------|-------------|---------------|
| FocalAgent parameters | 6 params | 11 params (matches original) |
| Trust update | Simple delta | Complex asymmetric with history |
| Signal update | Basic EWMA | Discounted EWMA with noise |
| expected_p() | Just returns x | Beta distribution formula |
| History tracking | None | 4 history lists |
| Trust dynamics | Simplified | Match/betrayal/surprise categorization |
| Penalties | Single λ | λ for betrayal, μ for surprise |

## Files Modified
1. `/trust_bayesian_agent_comparison/agents/base.py`
2. `/trust_bayesian_agent_comparison/agents/focal_agent.py` (complete rewrite)
3. `/trust_bayesian_agent_comparison/agents/bayesian_agent.py`

## Next Steps
1. ✅ Verify agents work correctly (test_agents_updated.py passed)
2. ⏳ Run comparison with original notebook outputs
3. ⏳ Update git commit with corrected implementation
4. ⏳ Verify simulation outputs match original
