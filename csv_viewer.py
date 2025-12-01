#!/usr/bin/env python3
"""
CSV Viewer for Trust-Bayesian Agent Comparison Results
Run this script to view all Monte Carlo analysis results
"""

import pandas as pd
import os
from pathlib import Path

def main():
    results_dir = Path('results')
    
    print('TRUST-BAYESIAN AGENT COMPARISON RESULTS')
    print('=' * 60)
    
    # 1. Summary Statistics
    print('\n1. SUMMARY STATISTICS')
    print('-' * 30)
    df_summary = pd.read_csv(results_dir / 'summary_statistics.csv')
    print(df_summary.to_string(index=False))
    
    # 2. Failed Collaboration (Betrayal Rates)
    print('\n\n2. FAILED COLLABORATION (Betrayal Rates)')
    print('-' * 40)
    df_failed = pd.read_csv(results_dir / 'failed_collaboration_comparison.csv')
    print(df_failed.to_string(index=False))
    
    # 3. Mutual Cooperation
    print('\n\n3. MUTUAL COOPERATION RATES')
    print('-' * 30)
    df_mutual = pd.read_csv(results_dir / 'mutual_cooperation_comparison.csv')
    print(df_mutual.to_string(index=False))
    
    # 4. Total Payoff
    print('\n\n4. TOTAL PAYOFF COMPARISON')
    print('-' * 30)
    df_payoff = pd.read_csv(results_dir / 'total_payoff_comparison.csv')
    print(df_payoff.to_string(index=False))
    
    # 5. ExpectationViolation Deep Dive
    print('\n\n5. EXPECTATIONVIOLATION DETAILED ANALYSIS')
    print('-' * 45)
    
    # Load detailed data
    df_trust_ev = pd.read_csv(results_dir / 'ExpectationViolation_agent1.csv')
    df_bayes_ev = pd.read_csv(results_dir / 'ExpectationViolation_agent2.csv')
    
    print(f"Trust-based agent data: {len(df_trust_ev):,} rows")
    print(f"Bayesian agent data: {len(df_bayes_ev):,} rows")
    
    # Final round stats
    final_trust = df_trust_ev[df_trust_ev['round'] == 99]
    final_bayes = df_bayes_ev[df_bayes_ev['round'] == 99]
    
    print("\nFinal round cooperation rates:")
    print(f"Trust-based: {final_trust['agent_action'].mean():.3f}")
    print(f"Bayesian: {final_bayes['agent_action'].mean():.3f}")
    
    # Betrayal rate calculation
    trust_betrayal = df_trust_ev.groupby('run_id').apply(
        lambda x: ((x['agent_action'] == 1) & (x['partner_action'] == 0)).mean(), include_groups=False
    )
    bayes_betrayal = df_bayes_ev.groupby('run_id').apply(
        lambda x: ((x['agent_action'] == 1) & (x['partner_action'] == 0)).mean(), include_groups=False
    )
    
    print("\nBetrayal rates (lower = better):")
    print(f"Trust-based: {trust_betrayal.mean():.1%}")
    print(f"Bayesian: {bayes_betrayal.mean():.1%}")
    print(f"Difference: {(trust_betrayal.mean() - bayes_betrayal.mean()):+.1%}")

if __name__ == '__main__':
    main()
