# How to View Results and Tables

## 📊 Viewing Comparison Tables

### Option 1: Console Output (Recommended for Quick View)
When you run the scripts, the tables are **printed to the console** in a nicely formatted way:

```bash
# Run demo (fast, 5 partners, 10 runs)
python run_demo.py

# OR run full study (all partners, 300 runs)
python run_full_study.py
```

**Example output you'll see:**
```
==================================================================================================================================
MONTE CARLO: SUCCESSFUL COLLABORATION RATE COMPARISON
Based on 10 runs with paired seeds per partner
KPI: Mutual Cooperation Rate (both agent and partner choose to collaborate)
==================================================================================================================================
Partner Strategy Trust-Based Mean Trust-Based Std Bayesian Mean Bayesian Std Difference (Trust - Bayes) Sig % Trust Better
 AlwaysCooperate            74.3%            7.2%         84.3%         4.7%                     -10.0%  **             0%
       TitForTat             2.7%            3.1%          7.9%        12.6%                      -5.1%  ns            30%
...
```

### Option 2: CSV Files (For Analysis/Excel)
All tables are automatically saved as CSV files in the `results/` directory:

```bash
# View in terminal
cat results/mutual_cooperation_comparison.csv
cat results/failed_collaboration_comparison.csv
cat results/total_payoff_comparison.csv

# Or use column for better formatting
column -t -s, results/mutual_cooperation_comparison.csv

# Open in Excel/LibreOffice
xdg-open results/mutual_cooperation_comparison.csv
```

### Option 3: Using Python/Pandas
```python
import pandas as pd

# Load any comparison table
df = pd.read_csv('results/mutual_cooperation_comparison.csv')
print(df)

# Or load all tables
mutual_coop = pd.read_csv('results/mutual_cooperation_comparison.csv')
betrayal = pd.read_csv('results/failed_collaboration_comparison.csv')
payoff = pd.read_csv('results/total_payoff_comparison.csv')

# Display with nice formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(mutual_coop)
```

### Option 4: Using VS Code
1. Navigate to `results/` folder in VS Code Explorer
2. Click on any `.csv` file
3. VS Code will display it in a nice table format

## 📈 Viewing Visualizations

### Image Files (PNG)
All visualizations are saved in `results/figures/`:

```bash
# List all figures
ls results/figures/*.png

# Open specific visualization
xdg-open results/figures/focal_trust_evolution.png
xdg-open results/figures/bayesian_final_distributions.png
xdg-open results/figures/payoff_comparison.png

# Open all visualizations at once
xdg-open results/figures/*.png
```

### Available Visualizations
- **Trust evolution**: `focal_trust_evolution.png`
- **Signal evolution**: `focal_signal_evolution.png`
- **Expected p evolution**: `focal_expected_p_evolution.png`, `bayesian_expected_p_evolution.png`
- **Final distributions**: `focal_final_distributions.png`, `bayesian_final_distributions.png`
- **Basic comparisons**: `payoff_comparison.png`, `cooperation_rates.png`, `payoff_advantage.png`
- **Sample trajectories**: `belief_trajectories.png`

## 📁 Complete Results Structure

```
results/
├── figures/                                    # All visualizations
│   ├── focal_trust_evolution.png
│   ├── focal_signal_evolution.png
│   ├── focal_expected_p_evolution.png
│   ├── focal_final_distributions.png
│   ├── bayesian_expected_p_evolution.png
│   ├── bayesian_final_distributions.png
│   ├── payoff_comparison.png
│   ├── cooperation_rates.png
│   ├── payoff_advantage.png
│   └── belief_trajectories.png
│
├── mutual_cooperation_comparison.csv          # TABLE 1: Successful collaboration
├── failed_collaboration_comparison.csv        # TABLE 2: Unsuccessful collaboration (betrayal)
├── total_payoff_comparison.csv               # TABLE 3: Total payoffs
├── focal_final_decisions.csv                 # Trust-based final round decisions
├── bayesian_final_decisions.csv              # Bayesian final round decisions
├── summary_statistics.csv                    # Basic summary stats
│
└── monte_carlo/                              # Raw simulation data
    └── YYYY-MM-DD/
        ├── FocalAgent_vs_Partner.parquet
        └── BayesianFocalAgent_vs_Partner.parquet
```

## 🔍 Quick Commands

```bash
# View all comparison tables at once
echo "=== MUTUAL COOPERATION ===" && cat results/mutual_cooperation_comparison.csv && \
echo -e "\n=== BETRAYAL RATE ===" && cat results/failed_collaboration_comparison.csv && \
echo -e "\n=== TOTAL PAYOFF ===" && cat results/total_payoff_comparison.csv

# Count how many visualizations were generated
ls results/figures/*.png | wc -l

# View summary of results
cat results/summary_statistics.csv

# Check Monte Carlo data
ls -lh results/monte_carlo/*/
```

## 💡 Tips

1. **For Quick Testing**: Use `run_demo.py` first (takes ~5 minutes)
2. **For Publication**: Use `run_full_study.py` (takes ~30-60 minutes)
3. **Console vs Files**: Console output is formatted for readability; CSV files are for analysis
4. **Table Interpretation**: 
   - Look for the `Sig` column (statistical significance)
   - Check `% Trust Better` to see win rates
   - Read the `INTERPRETATION` section at the bottom of each table
5. **Visualizations**: Open PNG files in any image viewer or include in papers/presentations

## 📊 Creating Custom Views

### Python Script to View All Tables
```python
#!/usr/bin/env python3
"""View all comparison tables nicely formatted."""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

tables = {
    'SUCCESSFUL COLLABORATION': 'mutual_cooperation_comparison.csv',
    'BETRAYAL RATE': 'failed_collaboration_comparison.csv',
    'TOTAL PAYOFF': 'total_payoff_comparison.csv'
}

for title, filename in tables.items():
    print(f"\n{'='*130}")
    print(f"{title}")
    print('='*130)
    df = pd.read_csv(f'results/{filename}')
    print(df.to_string(index=False))
    print()
```

Save as `view_tables.py` and run: `python view_tables.py`

## 🎯 What to Look For

### In Console Output:
- ✅ Three main comparison tables with proper headers
- ✅ Summary statistics showing wins/losses
- ✅ Interpretation notes explaining positive/negative differences
- ✅ Significance levels (*** / ** / * / ns)

### In CSV Files:
- ✅ Column headers: "Partner Strategy", "Trust-Based Mean/Std", "Bayesian Mean/Std", etc.
- ✅ "% Trust Better" column showing percentage of runs where Trust-Based wins
- ✅ "Difference (Trust - Bayes)" showing the gap
- ✅ "Sig" column showing statistical significance

### In Visualizations:
- ✅ Clear legends distinguishing Trust-Based vs Bayesian agents
- ✅ All partner types shown in evolution plots
- ✅ Decision thresholds marked on relevant plots
- ✅ Mean trajectories with standard deviation bands
