#!/usr/bin/env python3
"""
View comparison tables for selected partners in a nicely formatted way.

Usage:
    python view_selected_partners.py
"""

import pandas as pd
from pathlib import Path

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', None)

results_dir = Path('results')

# Define selected partners (mapped to actual partner names in results)
# Note: Joss Strategy is not implemented in the current study
selected_partners = [
    'SingleCycle',        # Trust-Building Betrayer (cooperates for 30 rounds, then defects)
    'StrategicCheater',   # Strategic Cheater
    'SuspiciousTFT',      # Generous TFT (starts with defect, then Tit-for-Tat)
    'TitForTat',          # Tit-for-Tat (Cooperate)
    'Pavlov',             # Pavlov (Win-Stay-Lose-Shift)
    'PeriodicCheater',    # Periodic Cheater
    'GrimTrigger'         # Grim Trigger
]

# Define tables to display
tables = [
    {
        'title': 'SUCCESSFUL COLLABORATION RATE COMPARISON',
        'file': 'mutual_cooperation_comparison.csv',
        'description': 'KPI: Mutual Cooperation Rate (both agents cooperate) - HIGHER IS BETTER'
    },
    {
        'title': 'UNSUCCESSFUL COLLABORATION (BETRAYAL) RATE COMPARISON',
        'file': 'failed_collaboration_comparison.csv',
        'description': 'KPI: Betrayal Rate (agent cooperates, partner defects) - LOWER IS BETTER'
    },
    {
        'title': 'TOTAL PAYOFF COMPARISON',
        'file': 'total_payoff_comparison.csv',
        'description': 'KPI: Sum of Payoffs over all rounds - HIGHER IS BETTER'
    }
]

def main():
    print("\n" + "="*130)
    print("MONTE CARLO COMPARISON TABLES - SELECTED PARTNERS")
    print(f"Partners: {', '.join(selected_partners)}")
    print("="*130)
    
    for table_info in tables:
        filepath = results_dir / table_info['file']
        
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            print("Run 'python run_demo.py' or 'python run_full_study.py' first to generate results.")
            continue
        
        print("\n" + "="*130)
        print(f"{table_info['title']}")
        print(table_info['description'])
        print("="*130)
        
        df = pd.read_csv(filepath)
        
        # Filter to selected partners
        filtered_df = df[df['Partner Strategy'].isin(selected_partners)]
        
        if filtered_df.empty:
            print("No data found for selected partners.")
        else:
            print(filtered_df.to_string(index=False))
        print()
    
    # Display summary for selected partners
    print("\n" + "="*130)
    print("SUMMARY - SELECTED PARTNERS")
    print("="*130)
    
    if (results_dir / 'total_payoff_comparison.csv').exists():
        payoff_df = pd.read_csv(results_dir / 'total_payoff_comparison.csv')
        filtered_payoff = payoff_df[payoff_df['Partner Strategy'].isin(selected_partners)]
        
        if not filtered_payoff.empty:
            # Count wins by checking the difference column
            # Extract numeric values from the formatted strings
            differences = []
            for val in filtered_payoff['Difference (Trust - Bayes)']:
                # Handle both string and numeric types
                if isinstance(val, str):
                    numeric_val = float(val.replace('%', '').replace('+', ''))
                else:
                    numeric_val = float(val)
                differences.append(numeric_val)
            
            trust_wins = sum(1 for d in differences if d > 0)
            bayes_wins = sum(1 for d in differences if d < 0)
            ties = sum(1 for d in differences if d == 0)
            
            print(f"Overall Performance (Total Payoff) - Selected Partners:")
            print(f"  Trust-Based Wins: {trust_wins} partners")
            print(f"  Bayesian Wins:    {bayes_wins} partners")
            print(f"  Ties:             {ties} partners")
            print()
    
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("\nNOTE:")
    print("  - For Collaboration & Payoff: Positive difference means Trust-Based performs better")
    print("  - For Betrayal: Negative difference means Trust-Based performs better (less exploited)")
    print("\nSelected Partners Mapping:")
    print("  - Trust-Building Betrayer → SingleCycle")
    print("  - Strategic Cheater → StrategicCheater")
    print("  - Generous TFT → SuspiciousTFT")
    print("  - Tit-for-Tat (Cooperate) → TitForTat")
    print("  - Pavlov (Win-Stay-Lose-Shift) → Pavlov")
    print("  - Periodic Cheater → PeriodicCheater")
    print("  - Grim Trigger → GrimTrigger")
    print("  - Joss Strategy: Not implemented in current study")

if __name__ == "__main__":
    main()