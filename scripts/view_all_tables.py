#!/usr/bin/env python3
"""
View all comparison tables in a nicely formatted way.

Usage:
    python view_all_tables.py [--word] [--csv]

Options:
    --word: Output in tab-separated format for easy Word table import
    --csv: Output as CSV files for import
"""

import pandas as pd
import argparse
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

def output_word_format(df, title, description):
    """Output table in tab-separated format for Word import."""
    print(f"\n{title}")
    print(f"{description}")
    print("\n" + "\t".join(df.columns))
    for _, row in df.iterrows():
        print("\t".join(str(val) for val in row.values))
    print()

def output_csv_files():
    """Save tables as CSV files."""
    csv_dir = results_dir / 'csv_for_word'
    csv_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving CSV files to: {csv_dir}")
    print("You can import these CSV files directly into Word or Excel.")
    
    for table_info in tables:
        filepath = results_dir / table_info['file']
        if filepath.exists():
            df = pd.read_csv(filepath)
            csv_path = csv_dir / f"{table_info['file']}"
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='View comparison tables')
    parser.add_argument('--word', action='store_true', help='Output in tab-separated format for Word table import')
    parser.add_argument('--csv', action='store_true', help='Save tables as CSV files for import')
    
    args = parser.parse_args()
    
    if args.csv:
        output_csv_files()
        return
    
    print("\n" + "="*130)
    print("MONTE CARLO COMPARISON TABLES")
    if args.word:
        print("TAB-SEPARATED FORMAT - Copy and paste into Word, then use 'Convert Text to Table'")
    else:
        print("FORMATTED TEXT DISPLAY")
    print("="*130)
    
    for table_info in tables:
        filepath = results_dir / table_info['file']
        
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filepath}")
            print("Run 'python run_demo.py' or 'python run_full_study.py' first to generate results.")
            continue
        
        df = pd.read_csv(filepath)
        
        if args.word:
            output_word_format(df, table_info['title'], table_info['description'])
        else:
            print("\n" + "="*130)
            print(f"{table_info['title']}")
            print(table_info['description'])
            print("="*130)
            print(df.to_string(index=False))
            print()
    
    # Display summary
    print("\n" + "="*130)
    print("SUMMARY")
    print("="*130)
    
    if (results_dir / 'total_payoff_comparison.csv').exists():
        payoff_df = pd.read_csv(results_dir / 'total_payoff_comparison.csv')
        
        # Count wins by checking the difference column
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
    
    if args.word:
        print("INSTRUCTIONS FOR WORD:")
        print("1. Copy the tab-separated text above")
        print("2. Paste into Word document")
        print("3. Select the pasted text")
        print("4. Go to: Insert → Table → Convert Text to Table")
        print("5. Choose 'Tabs' as delimiter")
        print("6. Click OK")
    else:
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    print("\nNOTE:")
    print("  - For Collaboration & Payoff: Positive difference means Trust-Based performs better")
    print("  - For Betrayal: Negative difference means Trust-Based performs better (less exploited)")
    print()

if __name__ == "__main__":
    main()
