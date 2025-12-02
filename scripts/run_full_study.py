#!/usr/bin/env python3
"""
Main script to run the complete Trust vs Bayesian agent comparison study.

This script:
1. Runs Monte Carlo simulations for both agents against all partner types
2. Computes summary statistics and metrics
3. Generates comparison tables
4. Creates visualizations
5. Saves all results to the results/ directory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.partners import (
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    RandomPartner,
    TitForTatCooperatePartner,
    SuspiciousTitForTatPartner,
    GrimTriggerPartner,
    PeriodicCheaterPartner,
    SingleCyclePartner,
    GradualDeteriorationPartner,
    StrategicCheaterPartner,
    ExpectationViolationPartner,
)
from trust_bayesian_agent_comparison.analysis import MonteCarloManager
from trust_bayesian_agent_comparison.visualization import MonteCarloVisualizer
from trust_bayesian_agent_comparison.config import (
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    RESULTS_DIR,
    FIGURES_DIR,
)


# Define partner configurations
PARTNERS = {
    # Fixed strategies
    'AlwaysCooperate': lambda: AlwaysCooperatePartner(),
    'AlwaysDefect': lambda: AlwaysDefectPartner(),
    'Random': lambda: RandomPartner(p=0.5),
    'PeriodicCheater': lambda: PeriodicCheaterPartner(cycle_length=6, cheat_duration=2),
    'SingleCycle': lambda: SingleCyclePartner(num_rounds=100, cooperate_fraction=0.3),
    'GradualDeterioration': lambda: GradualDeteriorationPartner(deterioration_rate=0.8),
    
    # Reactive strategies
    'TitForTat': lambda: TitForTatCooperatePartner(),
    'SuspiciousTFT': lambda: SuspiciousTitForTatPartner(),
    'GrimTrigger': lambda: GrimTriggerPartner(),
    
    # Belief-driven strategies
    'StrategicCheater': lambda: StrategicCheaterPartner(t_threshold=5.0),
    'ExpectationViolation': lambda: ExpectationViolationPartner(),
}


def run_monte_carlo_simulations(overwrite=False, notebook_compatible_seeding=False):
    """Run Monte Carlo simulations for all partners.
    
    Args:
        overwrite: If True, regenerate results even if they exist
        notebook_compatible_seeding: If True, use notebook-style direct seeding
            for validation against notebook results. If False, use isolated RNG
            for proper Monte Carlo independence (recommended for production).
    """
    print("="*80)
    print("RUNNING MONTE CARLO SIMULATIONS")
    print("="*80)
    if notebook_compatible_seeding:
        print("Using NOTEBOOK-COMPATIBLE seeding (for validation)")
    else:
        print("Using ISOLATED RNG seeding (production mode)")
    
    mc_manager = MonteCarloManager()
    
    # Agent factories
    focal_factory = lambda: FocalAgent()
    bayesian_factory = lambda: BayesianFocalAgent()
    
    all_results = {}
    
    for partner_name, partner_factory in PARTNERS.items():
        print(f"\n{'='*80}")
        print(f"Partner: {partner_name}")
        print(f"{'='*80}")
        
        df_focal, df_bayesian = mc_manager.run_monte_carlo(
            agent1_factory=focal_factory,
            agent2_factory=bayesian_factory,
            partner_factory=partner_factory,
            partner_name=partner_name,
            num_runs=NUM_MONTE_CARLO_RUNS,
            overwrite=overwrite,
            notebook_compatible_seeding=notebook_compatible_seeding,
        )
        
        all_results[partner_name] = {
            'focal': df_focal,
            'bayesian': df_bayesian,
        }
    
    return all_results


def compute_summary_statistics(all_results):
    """Compute summary statistics for all simulations."""
    print("\n" + "="*80)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*80)
    
    summary_data = []
    
    for partner_name, data in all_results.items():
        df_focal = data['focal']
        df_bayesian = data['bayesian']
        
        # Aggregate by run
        focal_by_run = df_focal.groupby('run_id').agg({
            'agent_payoff': 'sum',
            'agent_action': 'mean',  # cooperation rate
        }).reset_index()
        
        bayesian_by_run = df_bayesian.groupby('run_id').agg({
            'agent_payoff': 'sum',
            'agent_action': 'mean',
        }).reset_index()
        
        # Summary statistics
        summary_data.append({
            'Partner': partner_name,
            'Focal_Mean_Payoff': focal_by_run['agent_payoff'].mean(),
            'Focal_Std_Payoff': focal_by_run['agent_payoff'].std(),
            'Focal_Cooperation_Rate': focal_by_run['agent_action'].mean(),
            'Bayesian_Mean_Payoff': bayesian_by_run['agent_payoff'].mean(),
            'Bayesian_Std_Payoff': bayesian_by_run['agent_payoff'].std(),
            'Bayesian_Cooperation_Rate': bayesian_by_run['agent_action'].mean(),
            'Payoff_Difference': focal_by_run['agent_payoff'].mean() - bayesian_by_run['agent_payoff'].mean(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_file = RESULTS_DIR / 'summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary statistics saved to: {summary_file}")
    
    return summary_df


def create_comparison_table(summary_df):
    """Create formatted comparison table."""
    print("\n" + "="*80)
    print("AGENT PERFORMANCE COMPARISON")
    print("="*80)
    
    # Format for display
    display_df = summary_df.copy()
    display_df['Trust-Based Payoff'] = display_df['Focal_Mean_Payoff'].apply(lambda x: f"{x:.1f}")
    display_df['Bayesian Payoff'] = display_df['Bayesian_Mean_Payoff'].apply(lambda x: f"{x:.1f}")
    display_df['Trust-Based Coop%'] = (display_df['Focal_Cooperation_Rate'] * 100).apply(lambda x: f"{x:.1f}%")
    display_df['Bayesian Coop%'] = (display_df['Bayesian_Cooperation_Rate'] * 100).apply(lambda x: f"{x:.1f}%")
    display_df['Δ Payoff'] = display_df['Payoff_Difference'].apply(lambda x: f"{x:+.1f}")
    
    # Select columns for display
    table = display_df[['Partner', 'Trust-Based Payoff', 'Bayesian Payoff', 'Δ Payoff', 
                        'Trust-Based Coop%', 'Bayesian Coop%']]
    
    print("\n" + table.to_string(index=False))
    
    # Identify best performer for each partner
    print("\n" + "="*80)
    print("WINNER SUMMARY")
    print("="*80)
    
    trust_wins = (summary_df['Payoff_Difference'] > 0).sum()
    bayesian_wins = (summary_df['Payoff_Difference'] < 0).sum()
    ties = (summary_df['Payoff_Difference'] == 0).sum()
    
    print(f"\nTrust-Based Agent Wins: {trust_wins}")
    print(f"Bayesian Agent Wins: {bayesian_wins}")
    print(f"Ties: {ties}")
    
    return table


def create_visualizations(all_results, summary_df):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # 1. Payoff comparison bar chart
    print("\n1. Creating payoff comparison chart...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    focal_payoffs = summary_df['Focal_Mean_Payoff'].values
    bayesian_payoffs = summary_df['Bayesian_Mean_Payoff'].values
    
    ax.bar(x - width/2, focal_payoffs, width, label='Trust-Based Agent', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, bayesian_payoffs, width, label='Bayesian Agent', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Partner Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Total Payoff', fontsize=12, fontweight='bold')
    ax.set_title('Agent Performance Comparison Across Partner Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Partner'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'payoff_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 2. Cooperation rate comparison
    print("2. Creating cooperation rate comparison...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    focal_coop = summary_df['Focal_Cooperation_Rate'].values * 100
    bayesian_coop = summary_df['Bayesian_Cooperation_Rate'].values * 100
    
    ax.bar(x - width/2, focal_coop, width, label='Trust-Based Agent', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, bayesian_coop, width, label='Bayesian Agent', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Partner Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cooperation Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Agent Cooperation Rates Across Partner Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Partner'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cooperation_rates.png', bbox_inches='tight')
    plt.close()
    
    # 3. Payoff difference (advantage)
    print("3. Creating payoff advantage chart...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    differences = summary_df['Payoff_Difference'].values
    colors = ['#2E86AB' if d > 0 else '#A23B72' for d in differences]
    
    ax.barh(summary_df['Partner'], differences, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Payoff Advantage (Focal - Bayesian)', fontsize=12, fontweight='bold')
    ax.set_title('Payoff Advantage by Partner Type', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'payoff_advantage.png', bbox_inches='tight')
    plt.close()
    
    # 4. Sample trajectory plots for selected partners
    print("4. Creating sample trajectory plots...")
    selected_partners = ['AlwaysCooperate', 'StrategicCheater', 'Adaptive', 'GrimTrigger']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, partner_name in enumerate(selected_partners):
        if partner_name not in all_results:
            continue
            
        ax = axes[idx]
        
        # Get first run data
        df_focal = all_results[partner_name]['focal']
        df_bayesian = all_results[partner_name]['bayesian']
        
        run_0_focal = df_focal[df_focal['run_id'] == 0]
        run_0_bayesian = df_bayesian[df_bayesian['run_id'] == 0]
        
        # Plot beliefs
        ax.plot(run_0_focal['round'], run_0_focal['agent_belief'], 
                label='Trust-Based Agent', color='#2E86AB', linewidth=2)
        ax.plot(run_0_bayesian['round'], run_0_bayesian['agent_belief'], 
                label='Bayesian Agent', color='#A23B72', linewidth=2)
        
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Belief (Cooperation Probability)', fontsize=10)
        ax.set_title(f'Partner: {partner_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Belief Evolution Trajectories (Sample Run)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'belief_trajectories.png', bbox_inches='tight')
    plt.close()
    
    print(f"\nAll visualizations saved to: {FIGURES_DIR}/")


def main(notebook_compatible_seeding=True):
    """Run complete study.
    
    Args:
        notebook_compatible_seeding: If True, use notebook-style seeding for validation
    """
    print("\n" + "="*80)
    print("TRUST-BASED VS BAYESIAN AGENT COMPARISON STUDY")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Monte Carlo Runs: {NUM_MONTE_CARLO_RUNS}")
    print(f"  - Rounds per Simulation: {NUM_ROUNDS}")
    print(f"  - Partner Types: {len(PARTNERS)}")
    print(f"  - Results Directory: {RESULTS_DIR}")
    print(f"  - Figures Directory: {FIGURES_DIR}")
    print(f"  - Seeding Mode: {'Notebook-Compatible' if notebook_compatible_seeding else 'Isolated RNG (Production)'}")
    
    # Run simulations
    all_results = run_monte_carlo_simulations(
        overwrite=True, 
        notebook_compatible_seeding=notebook_compatible_seeding
    )
    
    # Compute statistics (original summary)
    summary_df = compute_summary_statistics(all_results)
    
    # Create comparison table (original format)
    table = create_comparison_table(summary_df)
    
    # Generate basic visualizations (original)
    create_visualizations(all_results, summary_df)
    
    # Generate comprehensive Monte Carlo visualizations and analysis
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE MONTE CARLO ANALYSIS")
    print("="*80)
    
    visualizer = MonteCarloVisualizer(all_results)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*80)
    print("STUDY COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print("\nKey files:")
    print(f"  - Summary statistics: {RESULTS_DIR}/summary_statistics.csv")
    print(f"  - Monte Carlo data: {RESULTS_DIR}/*_agent[1|2].csv")
    print(f"  - Basic visualizations: {FIGURES_DIR}/payoff_*.png")
    print(f"  - Trust evolution: {FIGURES_DIR}/focal_trust_evolution.png")
    print(f"  - Signal evolution: {FIGURES_DIR}/focal_signal_evolution.png")
    print(f"  - Expected p evolution: {FIGURES_DIR}/*_expected_p_evolution.png")
    print(f"  - Final distributions: {FIGURES_DIR}/*_final_distributions.png")
    print(f"  - Comparison tables: {RESULTS_DIR}/*_comparison.csv")
    print(f"  - Decision trends: {RESULTS_DIR}/*_final_decisions.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Trust vs Bayesian agent comparison study"
    )
    parser.add_argument(
        "--notebook-seeding",
        action="store_true",
        help="Use notebook-compatible seeding for validation (default: use isolated RNG)"
    )
    
    args = parser.parse_args()
    main(notebook_compatible_seeding=args.notebook_seeding)
