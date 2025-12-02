#!/usr/bin/env python3
"""
Rounds Analysis: Effect of Simulation Length on Agent Performance

This script analyzes how the number of rounds affects:
1. Collaboration success rates
2. Payoff accumulation
3. Adaptation speed
4. Recovery from betrayal
5. Convergence stability

Compares Trust-based vs Bayesian agents across:
- Belief-driven partners (non-stationary)
- Reactive partners (stationary but conditional)
- Fixed partners (baseline)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.partners import (
    # Belief-driven (non-stationary)
    AdaptivePartner,
    StrategicCheaterPartner,
    ExpectationViolationPartner,
    # Reactive (conditional)
    TitForTatCooperatePartner,
    TitForTatDefectPartner,
    GrimTriggerPartner,
    PavlovPartner,
    # Fixed (for comparison)
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    PeriodicCheaterPartner,
    SingleCyclePartner,
    GradualDeteriorationPartner,
)
from trust_bayesian_agent_comparison.simulation import run_agent_simulation
from trust_bayesian_agent_comparison.config import RESULTS_DIR, FIGURES_DIR


# Rounds to test
ROUNDS_LIST = [10, 50, 70, 100, 200, 300, 500, 1000]

# Number of Monte Carlo runs per configuration
NUM_RUNS = 100

# Base seed
BASE_SEED = 200


# Partner configuration functions (called with num_rounds parameter)
def get_belief_driven_partners(num_rounds: int) -> Dict:
    """Get belief-driven partner factories for given number of rounds."""
    return {
        'Adaptive': lambda: AdaptivePartner(),
        'StrategicCheater': lambda: StrategicCheaterPartner(t_threshold=5.0),
        'ExpectationViolation': lambda: ExpectationViolationPartner(),
    }

def get_reactive_partners(num_rounds: int) -> Dict:
    """Get reactive partner factories for given number of rounds."""
    return {
        'TitForTat': lambda: TitForTatCooperatePartner(),
        'SuspiciousTFT': lambda: TitForTatDefectPartner(),
        'GrimTrigger': lambda: GrimTriggerPartner(),
        'Pavlov': lambda: PavlovPartner(),
    }

def get_fixed_partners(num_rounds: int) -> Dict:
    """Get fixed partner factories for given number of rounds."""
    return {
        'AlwaysCooperate': lambda: AlwaysCooperatePartner(),
        'AlwaysDefect': lambda: AlwaysDefectPartner(),
        'PeriodicCheater': lambda: PeriodicCheaterPartner(cycle_length=6, cheat_duration=2),
        'SingleCycle': lambda: SingleCyclePartner(num_rounds=num_rounds, cooperate_fraction=0.4),
        'GradualDeterioration': lambda: GradualDeteriorationPartner(deterioration_rate=0.8, num_rounds=num_rounds),
    }


def compute_metrics(df: pd.DataFrame, num_rounds: int) -> Dict:
    """
    Compute comprehensive metrics for a simulation.
    
    Args:
        df: Simulation results DataFrame
        num_rounds: Total number of rounds
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # 1. Total payoff
    metrics['total_payoff'] = df['agent_payoff'].sum()
    metrics['mean_payoff_per_round'] = df['agent_payoff'].mean()
    
    # 2. Cooperation rate
    metrics['cooperation_rate'] = df['agent_action'].mean()
    
    # 3. Mutual cooperation (successful collaboration)
    mutual_coop = ((df['agent_action'] == 1) & (df['partner_action'] == 1)).sum()
    metrics['mutual_cooperation_count'] = mutual_coop
    metrics['mutual_cooperation_rate'] = mutual_coop / num_rounds
    
    # 4. Collaboration failure (agent cooperates, partner defects)
    betrayal = ((df['agent_action'] == 1) & (df['partner_action'] == 0)).sum()
    metrics['betrayal_count'] = betrayal
    metrics['betrayal_rate'] = betrayal / num_rounds
    
    # 5. Mutual defection
    mutual_defect = ((df['agent_action'] == 0) & (df['partner_action'] == 0)).sum()
    metrics['mutual_defection_count'] = mutual_defect
    metrics['mutual_defection_rate'] = mutual_defect / num_rounds
    
    # 6. Exploitation (agent defects, partner cooperates)
    exploitation = ((df['agent_action'] == 0) & (df['partner_action'] == 1)).sum()
    metrics['exploitation_count'] = exploitation
    metrics['exploitation_rate'] = exploitation / num_rounds
    
    # 7. Final quarter performance (last 25% of rounds)
    quarter_idx = int(num_rounds * 0.75)
    final_quarter = df.iloc[quarter_idx:]
    metrics['final_quarter_payoff'] = final_quarter['agent_payoff'].mean()
    metrics['final_quarter_cooperation'] = final_quarter['agent_action'].mean()
    
    # 8. First quarter performance (adaptation speed)
    first_quarter = df.iloc[:int(num_rounds * 0.25)]
    metrics['first_quarter_payoff'] = first_quarter['agent_payoff'].mean()
    metrics['first_quarter_cooperation'] = first_quarter['agent_action'].mean()
    
    # 9. Variance (stability)
    metrics['payoff_variance'] = df['agent_payoff'].var()
    
    # 10. Trust-specific metrics (if available)
    if 'trust_level' in df.columns:
        metrics['final_trust'] = df['trust_level'].iloc[-1]
        metrics['mean_trust'] = df['trust_level'].mean()
        metrics['trust_variance'] = df['trust_level'].var()
    
    if 'signal' in df.columns:
        metrics['final_signal'] = df['signal'].iloc[-1]
        metrics['mean_signal'] = df['signal'].mean()
    
    # 11. Belief metrics (if available)
    if 'agent_belief' in df.columns:
        metrics['final_belief'] = df['agent_belief'].iloc[-1]
        metrics['belief_variance'] = df['agent_belief'].var()
    
    return metrics


def run_single_configuration(
    agent_factory,
    partner_factory,
    num_rounds: int,
    seed: int
) -> pd.DataFrame:
    """Run single simulation and compute metrics."""
    agent = agent_factory()
    partner = partner_factory()
    
    df = run_agent_simulation(agent, partner, num_rounds=num_rounds, seed=seed)
    
    return df


def run_rounds_comparison(
    agent_name: str,
    agent_factory,
    partner_configs_fn,
    rounds_list: List[int],
    num_runs: int,
    base_seed: int
) -> pd.DataFrame:
    """
    Run complete rounds analysis for one agent type.
    
    Args:
        agent_name: Name of the agent ('Trust-Based' or 'Bayesian')
        agent_factory: Factory function to create agent instances
        partner_configs_fn: Function that takes num_rounds and returns partner configs dict
        rounds_list: List of round counts to test
        num_runs: Number of Monte Carlo runs per configuration
        base_seed: Base random seed
        
    Returns:
        DataFrame with all results
    """
    results = []
    
    # Get partner names from first round count to calculate total
    sample_partners = partner_configs_fn(rounds_list[0])
    total_configs = len(sample_partners) * len(rounds_list) * num_runs
    completed = 0
    
    print(f"\n{'='*80}")
    print(f"Running {agent_name} Agent Analysis")
    print(f"{'='*80}")
    print(f"Total simulations: {total_configs}")
    
    for num_rounds in rounds_list:
        # Get partner configs for this specific round count
        partner_configs = partner_configs_fn(num_rounds)
        
        for partner_name, partner_factory in partner_configs.items():
            for run in range(num_runs):
                seed = base_seed + completed
                
                # Run simulation
                df = run_single_configuration(
                    agent_factory, partner_factory, num_rounds, seed
                )
                
                # Compute metrics
                metrics = compute_metrics(df, num_rounds)
                
                # Add metadata
                metrics['agent'] = agent_name
                metrics['partner'] = partner_name
                metrics['num_rounds'] = num_rounds
                metrics['run_id'] = run
                metrics['seed'] = seed
                
                results.append(metrics)
                
                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
    
    print(f"  Completed: {completed}/{total_configs}")
    
    return pd.DataFrame(results)


def analyze_and_visualize(
    results_df: pd.DataFrame,
    partner_category: str,
    output_dir: Path
):
    """
    Create comprehensive analysis and visualizations.
    
    Args:
        results_df: Combined results DataFrame
        partner_category: Category name ('belief_driven', 'reactive', 'fixed')
        output_dir: Output directory for figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    print(f"\n{'='*80}")
    print(f"Analyzing {partner_category.replace('_', ' ').title()} Partners")
    print(f"{'='*80}")
    
    # Group by agent, partner, and rounds
    grouped = results_df.groupby(['agent', 'partner', 'num_rounds']).agg({
        'total_payoff': ['mean', 'std'],
        'cooperation_rate': ['mean', 'std'],
        'mutual_cooperation_rate': ['mean', 'std'],
        'betrayal_rate': ['mean', 'std'],
        'final_quarter_payoff': ['mean', 'std'],
        'first_quarter_payoff': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # 1. Total Payoff vs Rounds
    print(f"\n1. Creating payoff vs rounds plot...")
    partners = grouped['partner'].unique()
    n_partners = len(partners)
    
    fig, axes = plt.subplots(2, (n_partners + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, partner in enumerate(partners):
        ax = axes[idx]
        data = grouped[grouped['partner'] == partner]
        
        for agent in ['Trust-Based', 'Bayesian']:
            agent_data = data[data['agent'] == agent]
            ax.errorbar(
                agent_data['num_rounds'],
                agent_data['total_payoff_mean'],
                yerr=agent_data['total_payoff_std'],
                label=agent,
                marker='o',
                linewidth=2,
                capsize=5,
                alpha=0.8
            )
        
        ax.set_xlabel('Number of Rounds', fontweight='bold')
        ax.set_ylabel('Total Payoff', fontweight='bold')
        ax.set_title(f'{partner}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    # Hide extra subplots
    for idx in range(n_partners, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Total Payoff vs Rounds: {partner_category.replace("_", " ").title()} Partners',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{partner_category}_payoff_vs_rounds.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Cooperation Success (Mutual Cooperation Rate)
    print(f"2. Creating cooperation success plot...")
    fig, axes = plt.subplots(2, (n_partners + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, partner in enumerate(partners):
        ax = axes[idx]
        data = grouped[grouped['partner'] == partner]
        
        for agent in ['Trust-Based', 'Bayesian']:
            agent_data = data[data['agent'] == agent]
            ax.errorbar(
                agent_data['num_rounds'],
                agent_data['mutual_cooperation_rate_mean'] * 100,
                yerr=agent_data['mutual_cooperation_rate_std'] * 100,
                label=agent,
                marker='s',
                linewidth=2,
                capsize=5,
                alpha=0.8
            )
        
        ax.set_xlabel('Number of Rounds', fontweight='bold')
        ax.set_ylabel('Mutual Cooperation Rate (%)', fontweight='bold')
        ax.set_title(f'{partner}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    for idx in range(n_partners, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Cooperation Success vs Rounds: {partner_category.replace("_", " ").title()} Partners',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{partner_category}_cooperation_vs_rounds.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # 3. Collaboration Failure (Betrayal Rate)
    print(f"3. Creating collaboration failure plot...")
    fig, axes = plt.subplots(2, (n_partners + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, partner in enumerate(partners):
        ax = axes[idx]
        data = grouped[grouped['partner'] == partner]
        
        for agent in ['Trust-Based', 'Bayesian']:
            agent_data = data[data['agent'] == agent]
            ax.errorbar(
                agent_data['num_rounds'],
                agent_data['betrayal_rate_mean'] * 100,
                yerr=agent_data['betrayal_rate_std'] * 100,
                label=agent,
                marker='^',
                linewidth=2,
                capsize=5,
                alpha=0.8
            )
        
        ax.set_xlabel('Number of Rounds', fontweight='bold')
        ax.set_ylabel('Betrayal Rate (%)', fontweight='bold')
        ax.set_title(f'{partner}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    for idx in range(n_partners, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Collaboration Failure vs Rounds: {partner_category.replace("_", " ").title()} Partners',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{partner_category}_betrayal_vs_rounds.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # 4. Adaptation Speed (First Quarter Performance)
    print(f"4. Creating adaptation speed plot...")
    fig, axes = plt.subplots(2, (n_partners + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, partner in enumerate(partners):
        ax = axes[idx]
        data = grouped[grouped['partner'] == partner]
        
        for agent in ['Trust-Based', 'Bayesian']:
            agent_data = data[data['agent'] == agent]
            ax.errorbar(
                agent_data['num_rounds'],
                agent_data['first_quarter_payoff_mean'],
                yerr=agent_data['first_quarter_payoff_std'],
                label=agent,
                marker='D',
                linewidth=2,
                capsize=5,
                alpha=0.8
            )
        
        ax.set_xlabel('Number of Rounds', fontweight='bold')
        ax.set_ylabel('First Quarter Mean Payoff', fontweight='bold')
        ax.set_title(f'{partner}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    for idx in range(n_partners, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Adaptation Speed (First 25%): {partner_category.replace("_", " ").title()} Partners',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{partner_category}_adaptation_speed.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # 5. Summary comparison table
    print(f"5. Creating summary comparison table...")
    summary = results_df.groupby(['agent', 'partner', 'num_rounds']).agg({
        'total_payoff': 'mean',
        'cooperation_rate': 'mean',
        'mutual_cooperation_rate': 'mean',
        'betrayal_rate': 'mean',
    }).reset_index()
    
    # Pivot to compare agents side-by-side
    comparison = summary.pivot_table(
        index=['partner', 'num_rounds'],
        columns='agent',
        values=['total_payoff', 'mutual_cooperation_rate', 'betrayal_rate']
    ).reset_index()
    
    # Calculate advantage
    comparison['payoff_advantage'] = (
        comparison[('total_payoff', 'Trust-Based')] - 
        comparison[('total_payoff', 'Bayesian')]
    )
    comparison['cooperation_advantage'] = (
        comparison[('mutual_cooperation_rate', 'Trust-Based')] - 
        comparison[('mutual_cooperation_rate', 'Bayesian')]
    ) * 100
    comparison['betrayal_reduction'] = (
        comparison[('betrayal_rate', 'Bayesian')] - 
        comparison[('betrayal_rate', 'Trust-Based')]
    ) * 100
    
    # Save to CSV
    comparison.to_csv(output_dir / f'{partner_category}_comparison_table.csv', index=False)
    print(f"   Saved comparison table to: {output_dir / f'{partner_category}_comparison_table.csv'}")
    
    return grouped, comparison


def create_summary_plots(all_results: pd.DataFrame, output_dir: Path):
    """Create overall summary visualizations across all partner types."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Creating Summary Visualizations")
    print(f"{'='*80}")
    
    # 1. Payoff Advantage Heatmap by Partner Type and Rounds
    print("1. Creating payoff advantage heatmap...")
    pivot_data = all_results.groupby(['agent', 'partner', 'num_rounds'])['total_payoff'].mean().reset_index()
    pivot_wide = pivot_data.pivot_table(
        index='partner',
        columns='num_rounds',
        values='total_payoff',
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else 0
    )
    
    # Calculate advantage (Trust - Bayesian)
    trust_data = all_results[all_results['agent'] == 'Trust-Based'].groupby(['partner', 'num_rounds'])['total_payoff'].mean().reset_index()
    bayes_data = all_results[all_results['agent'] == 'Bayesian'].groupby(['partner', 'num_rounds'])['total_payoff'].mean().reset_index()
    
    advantage = trust_data.merge(bayes_data, on=['partner', 'num_rounds'], suffixes=('_trust', '_bayes'))
    advantage['advantage'] = advantage['total_payoff_trust'] - advantage['total_payoff_bayes']
    advantage_pivot = advantage.pivot(index='partner', columns='num_rounds', values='advantage')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(advantage_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Payoff Advantage (Trust - Bayesian)'}, ax=ax)
    ax.set_xlabel('Number of Rounds', fontweight='bold', fontsize=12)
    ax.set_ylabel('Partner Type', fontweight='bold', fontsize=12)
    ax.set_title('Trust-Based Agent Payoff Advantage Across Rounds', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_advantage_heatmap.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Win rate by rounds
    print("2. Creating win rate plot...")
    wins = []
    for num_rounds in ROUNDS_LIST:
        round_data = all_results[all_results['num_rounds'] == num_rounds]
        grouped = round_data.groupby(['agent', 'partner', 'run_id'])['total_payoff'].sum().reset_index()
        
        for partner in grouped['partner'].unique():
            partner_data = grouped[grouped['partner'] == partner]
            trust_payoffs = partner_data[partner_data['agent'] == 'Trust-Based']['total_payoff'].values
            bayes_payoffs = partner_data[partner_data['agent'] == 'Bayesian']['total_payoff'].values
            
            wins_count = (trust_payoffs > bayes_payoffs).sum()
            win_rate = wins_count / len(trust_payoffs) * 100
            
            wins.append({
                'num_rounds': num_rounds,
                'partner': partner,
                'win_rate': win_rate
            })
    
    wins_df = pd.DataFrame(wins)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    for partner in wins_df['partner'].unique():
        partner_wins = wins_df[wins_df['partner'] == partner]
        ax.plot(partner_wins['num_rounds'], partner_wins['win_rate'], 
                marker='o', linewidth=2, label=partner, alpha=0.8)
    
    ax.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Equal (50%)')
    ax.set_xlabel('Number of Rounds', fontweight='bold', fontsize=12)
    ax.set_ylabel('Trust-Based Win Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Trust-Based Agent Win Rate vs Bayesian Across Rounds', fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_win_rates.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\nAll summary visualizations saved to: {output_dir}/")


def main():
    """Run complete rounds analysis."""
    print("\n" + "="*80)
    print("ROUNDS ANALYSIS: EFFECT OF SIMULATION LENGTH")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Rounds tested: {ROUNDS_LIST}")
    print(f"  - Runs per configuration: {NUM_RUNS}")
    print(f"  - Belief-driven partners: {len(get_belief_driven_partners(100))}")
    print(f"  - Reactive partners: {len(get_reactive_partners(100))}")
    print(f"  - Fixed partners: {len(get_fixed_partners(100))}")
    
    # Create output directory
    output_dir = RESULTS_DIR / "rounds_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = FIGURES_DIR / "rounds_analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses for each partner category
    all_results = []
    
    # 1. Belief-driven partners (non-stationary)
    print("\n" + "="*80)
    print("BELIEF-DRIVEN PARTNERS (Non-Stationary)")
    print("="*80)
    
    belief_results_trust = run_rounds_comparison(
        'Trust-Based', lambda: FocalAgent(), get_belief_driven_partners,
        ROUNDS_LIST, NUM_RUNS, BASE_SEED
    )
    belief_results_bayes = run_rounds_comparison(
        'Bayesian', lambda: BayesianFocalAgent(), get_belief_driven_partners,
        ROUNDS_LIST, NUM_RUNS, BASE_SEED + 1000000
    )
    belief_results = pd.concat([belief_results_trust, belief_results_bayes], ignore_index=True)
    belief_results['partner_category'] = 'Belief-Driven'
    
    belief_results.to_csv(output_dir / 'belief_driven_results.csv', index=False)
    analyze_and_visualize(belief_results, 'belief_driven', figures_dir / 'belief_driven')
    all_results.append(belief_results)
    
    # 2. Reactive partners
    print("\n" + "="*80)
    print("REACTIVE PARTNERS")
    print("="*80)
    
    reactive_results_trust = run_rounds_comparison(
        'Trust-Based', lambda: FocalAgent(), get_reactive_partners,
        ROUNDS_LIST, NUM_RUNS, BASE_SEED + 2000000
    )
    reactive_results_bayes = run_rounds_comparison(
        'Bayesian', lambda: BayesianFocalAgent(), get_reactive_partners,
        ROUNDS_LIST, NUM_RUNS, BASE_SEED + 3000000
    )
    reactive_results = pd.concat([reactive_results_trust, reactive_results_bayes], ignore_index=True)
    reactive_results['partner_category'] = 'Reactive'
    
    reactive_results.to_csv(output_dir / 'reactive_results.csv', index=False)
    analyze_and_visualize(reactive_results, 'reactive', figures_dir / 'reactive')
    all_results.append(reactive_results)
    
    # 3. Fixed partners (baseline)
    print("\n" + "="*80)
    print("FIXED PARTNERS (Baseline)")
    print("="*80)
    
    fixed_results_trust = run_rounds_comparison(
        'Trust-Based', lambda: FocalAgent(), get_fixed_partners,
        ROUNDS_LIST, NUM_RUNS, BASE_SEED + 4000000
    )
    fixed_results_bayes = run_rounds_comparison(
        'Bayesian', lambda: BayesianFocalAgent(), get_fixed_partners,
        ROUNDS_LIST, NUM_RUNS, BASE_SEED + 5000000
    )
    fixed_results = pd.concat([fixed_results_trust, fixed_results_bayes], ignore_index=True)
    fixed_results['partner_category'] = 'Fixed'
    
    fixed_results.to_csv(output_dir / 'fixed_results.csv', index=False)
    analyze_and_visualize(fixed_results, 'fixed', figures_dir / 'fixed')
    all_results.append(fixed_results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_dir / 'all_rounds_analysis.csv', index=False)
    
    # Create summary visualizations
    create_summary_plots(combined_results, figures_dir)
    
    print("\n" + "="*80)
    print("ROUNDS ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Figures saved to: {figures_dir}/")
    print("\nKey outputs:")
    print(f"  - Combined results: {output_dir}/all_rounds_analysis.csv")
    print(f"  - Belief-driven analysis: {figures_dir}/belief_driven/")
    print(f"  - Reactive analysis: {figures_dir}/reactive/")
    print(f"  - Fixed analysis: {figures_dir}/fixed/")
    print(f"  - Summary heatmap: {figures_dir}/summary_advantage_heatmap.png")
    print(f"  - Win rates: {figures_dir}/summary_win_rates.png")


if __name__ == "__main__":
    main()
