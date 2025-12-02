#!/usr/bin/env python3
"""
Quick demo script - runs a smaller study for testing.

This is a faster version with fewer Monte Carlo runs (10 instead of 300)
and fewer partners (5 representative ones).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.partners import (
    AlwaysCooperatePartner,
    TitForTatCooperatePartner,
    StrategicCheaterPartner,
    GrimTriggerPartner,
    AdaptivePartner,
)
from trust_bayesian_agent_comparison.analysis import MonteCarloManager
from trust_bayesian_agent_comparison.visualization import MonteCarloVisualizer
from trust_bayesian_agent_comparison.config import NUM_ROUNDS, FIGURES_DIR


# Representative partners for quick demo
DEMO_PARTNERS = {
    'AlwaysCooperate': lambda: AlwaysCooperatePartner(),
    'TitForTat': lambda: TitForTatCooperatePartner(),
    'GrimTrigger': lambda: GrimTriggerPartner(),
    'Adaptive': lambda: AdaptivePartner(),
    'StrategicCheater': lambda: StrategicCheaterPartner(t_threshold=5.0),
}


def main():
    """Run quick demo study."""
    print("\n" + "="*80)
    print("QUICK DEMO: Trust vs Bayesian Agent Comparison")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Monte Carlo Runs: 10 (fast demo mode)")
    print(f"  - Rounds per Simulation: {NUM_ROUNDS}")
    print(f"  - Partner Types: {len(DEMO_PARTNERS)}")
    
    mc_manager = MonteCarloManager()
    
    # Agent factories
    focal_factory = lambda: FocalAgent()
    bayesian_factory = lambda: BayesianFocalAgent()
    
    all_results = {}
    summary_data = []
    
    # Run simulations
    print("\n" + "="*80)
    print("Running simulations...")
    print("="*80)
    
    for partner_name, partner_factory in DEMO_PARTNERS.items():
        print(f"\nPartner: {partner_name}")
        
        df_focal, df_bayesian = mc_manager.run_monte_carlo(
            agent1_factory=focal_factory,
            agent2_factory=bayesian_factory,
            partner_factory=partner_factory,
            partner_name=f"demo_{partner_name}",
            num_runs=10,  # Quick demo
            overwrite=True,  # Always fresh for demo
        )
        
        all_results[partner_name] = {'focal': df_focal, 'bayesian': df_bayesian}
        
        # Compute summary
        focal_payoff = df_focal.groupby('run_id')['agent_payoff'].sum().mean()
        bayesian_payoff = df_bayesian.groupby('run_id')['agent_payoff'].sum().mean()
        focal_coop = df_focal.groupby('run_id')['agent_action'].mean().mean()
        bayesian_coop = df_bayesian.groupby('run_id')['agent_action'].mean().mean()
        
        summary_data.append({
            'Partner': partner_name,
            'Focal_Payoff': focal_payoff,
            'Bayesian_Payoff': bayesian_payoff,
            'Difference': focal_payoff - bayesian_payoff,
            'Focal_Coop%': focal_coop * 100,
            'Bayesian_Coop%': bayesian_coop * 100,
        })
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format table
    display_df = summary_df.copy()
    display_df['Focal_Payoff'] = display_df['Focal_Payoff'].apply(lambda x: f"{x:.1f}")
    display_df['Bayesian_Payoff'] = display_df['Bayesian_Payoff'].apply(lambda x: f"{x:.1f}")
    display_df['Difference'] = display_df['Difference'].apply(lambda x: f"{x:+.1f}")
    display_df['Focal_Coop%'] = display_df['Focal_Coop%'].apply(lambda x: f"{x:.1f}%")
    display_df['Bayesian_Coop%'] = display_df['Bayesian_Coop%'].apply(lambda x: f"{x:.1f}%")
    
    print("\n" + display_df.to_string(index=False))
    
    # Winner summary
    print("\n" + "="*80)
    print("WINNER SUMMARY")
    print("="*80)
    
    focal_wins = (summary_df['Difference'] > 0).sum()
    bayesian_wins = (summary_df['Difference'] < 0).sum()
    
    print(f"\nTrust-Based Agent Wins: {focal_wins}")
    print(f"Bayesian Agent Wins: {bayesian_wins}")
    
    # Quick basic visualization
    print("\n" + "="*80)
    print("Creating basic visualization...")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Payoff comparison
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax1.bar(x - width/2, summary_df['Focal_Payoff'], width, label='Trust-Based', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, summary_df['Bayesian_Payoff'], width, label='Bayesian', color='#A23B72', alpha=0.8)
    ax1.set_xlabel('Partner Type', fontweight='bold')
    ax1.set_ylabel('Mean Total Payoff', fontweight='bold')
    ax1.set_title('Payoff Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['Partner'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Cooperation rate
    ax2.bar(x - width/2, summary_df['Focal_Coop%'], width, label='Trust-Based', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, summary_df['Bayesian_Coop%'], width, label='Bayesian', color='#A23B72', alpha=0.8)
    ax2.set_xlabel('Partner Type', fontweight='bold')
    ax2.set_ylabel('Cooperation Rate (%)', fontweight='bold')
    ax2.set_title('Cooperation Rates', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df['Partner'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=100, bbox_inches='tight')
    print("\nBasic visualization saved to: demo_results.png")
    plt.close()
    
    # Generate comprehensive Monte Carlo visualizations
    print("\n" + "="*80)
    print("Creating comprehensive Monte Carlo analysis...")
    print("="*80)
    
    visualizer = MonteCarloVisualizer(all_results, figures_dir=FIGURES_DIR / 'demo')
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nDemo visualizations saved to: figures/demo/")
    print("To run the full study with all partners and 300 Monte Carlo runs:")
    print("  python run_full_study.py")


if __name__ == "__main__":
    main()
