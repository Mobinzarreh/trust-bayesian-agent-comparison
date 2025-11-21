#!/usr/bin/env python3
"""
View all comparison tables in a nicely formatted way.

Usage:
    python view_all_tables.py
"""

import pandas as pd
from pathlib import Path

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', None)

results_dir = Path('results')

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
    print("MONTE CARLO COMPARISON TABLES")
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
        print(df.to_string(index=False))
        print()
    
    # Display summary
    print("\n" + "="*130)
    print("SUMMARY")
    print("="*130)
    
    if (results_dir / 'total_payoff_comparison.csv').exists():
        payoff_df = pd.read_csv(results_dir / 'total_payoff_comparison.csv')
        
        # Count wins by checking the difference column
        # Extract numeric values from the formatted strings
        differences = []
        for val in payoff_df['Difference (Trust - Bayes)']:
            # Handle both string and numeric types
            if isinstance(val, str):
                numeric_val = float(val.replace('%', '').replace('+', ''))
            else:
                numeric_val = float(val)
            differences.append(numeric_val)
        
        trust_wins = sum(1 for d in differences if d > 0)
        bayes_wins = sum(1 for d in differences if d < 0)
        ties = sum(1 for d in differences if d == 0)
        
        print(f"Overall Performance (Total Payoff):")
        print(f"  Trust-Based Wins: {trust_wins} partners")
        print(f"  Bayesian Wins:    {bayes_wins} partners")
        print(f"  Ties:             {ties} partners")
        print()
    
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("\nNOTE:")
    print("  - For Collaboration & Payoff: Positive difference means Trust-Based performs better")
    print("  - For Betrayal: Negative difference means Trust-Based performs better (less exploited)")
    print()

if __name__ == "__main__":
    main()
