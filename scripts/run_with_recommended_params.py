#!/usr/bin/env python3
"""
Run Monte Carlo simulations using recommended parameter values for the Trust-based (Focal) agent.

Reads recommendations from results/recommended_params.json
and runs paired simulations against all partners to compare with the Bayesian agent.
Results and figures are written to the standard results/ and results/figures/ locations.
"""
from __future__ import annotations

import json
from pathlib import Path

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.analysis import MonteCarloManager
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
from trust_bayesian_agent_comparison.config import RESULTS_DIR, FIGURES_DIR, NUM_MONTE_CARLO_RUNS, NUM_ROUNDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REC_FILE = PROJECT_ROOT / "results" / "recommended_params.json"


def load_recommendations() -> dict:
    if not REC_FILE.exists():
        raise FileNotFoundError(f"Recommendation file not found: {REC_FILE}. Run scripts/recommend_params.py first.")
    with open(REC_FILE, encoding="utf-8") as f:
        recs = json.load(f)
    return recs


def make_focal_factory(recs: dict):
    # Extract the recommended numeric values with defensive defaults
    def get(p: str, default: float):
        try:
            return float(recs[p]["recommended"])
        except (KeyError, ValueError, TypeError):
            return default

    # Map to FocalAgent constructor parameters
    eta = get("eta", 0.3)
    memory_discount = get("memory_discount", 0.9)
    trust_discount = get("trust_discount", 0.8)
    trust_smoothing = get("trust_smoothing", 0.2)
    loss_aversion = get("loss_aversion", 2.0)
    lambda_surprise = get("lambda_surprise", 0.5)
    inv_temp = get("inverse_temperature", 2.0)
    initial_x = get("initial_x", 0.33)
    t_init = get("t_init", 0.0)

    def factory() -> FocalAgent:
        return FocalAgent(
            u_i=initial_x,
            t_init=t_init,
            eta=eta,
            memory_discount=memory_discount,
            trust_discount=trust_discount,
            trust_smoothing=trust_smoothing,
            loss_aversion=loss_aversion,
            lambda_surprise=lambda_surprise,
            inv_temp=inv_temp,
        )

    return factory


def partners() -> dict[str, callable]:
    return {
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


def main() -> None:
    print("=" * 80)
    print("RUNNING STUDY WITH RECOMMENDED PARAMETERS")
    print("=" * 80)

    recs = load_recommendations()
    focal_factory = make_focal_factory(recs)
    bayesian_factory = lambda: BayesianFocalAgent()

    mc_manager = MonteCarloManager()
    all_results = {}

    for partner_name, partner_factory in partners().items():
        print(f"\n{'='*80}")
        print(f"Partner: {partner_name}")
        print(f"{'='*80}")
        df_focal, df_bayesian = mc_manager.run_monte_carlo(
            agent1_factory=focal_factory,
            agent2_factory=bayesian_factory,
            partner_factory=partner_factory,
            partner_name=partner_name,
            overwrite=True,
        )
        all_results[partner_name] = {"focal": df_focal, "bayesian": df_bayesian}

    # Basic summary and visualizations
    # --- Local summary and visualization functions (adapted from run_full_study.py) ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    def compute_summary_statistics_local(all_results_dict):
        print("\n" + "="*80)
        print("COMPUTING SUMMARY STATISTICS")
        print("="*80)

        summary_data = []
        for partner_name, data in all_results_dict.items():
            df_focal = data['focal']
            df_bayesian = data['bayesian']

            focal_by_run = df_focal.groupby('run_id').agg({
                'agent_payoff': 'sum',
                'agent_action': 'mean',
            }).reset_index()
            bayesian_by_run = df_bayesian.groupby('run_id').agg({
                'agent_payoff': 'sum',
                'agent_action': 'mean',
            }).reset_index()

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

        summary_df_local = pd.DataFrame(summary_data)
        summary_file = RESULTS_DIR / 'summary_statistics.csv'
        summary_df_local.to_csv(summary_file, index=False)
        print(f"\nSummary statistics saved to: {summary_file}")
        return summary_df_local

    def create_comparison_table_local(summary_df):
        print("\n" + "="*80)
        print("AGENT PERFORMANCE COMPARISON")
        print("="*80)

        display_df = summary_df.copy()
        display_df['Trust-Based Payoff'] = display_df['Focal_Mean_Payoff'].apply(lambda x: f"{x:.1f}")
        display_df['Bayesian Payoff'] = display_df['Bayesian_Mean_Payoff'].apply(lambda x: f"{x:.1f}")
        display_df['Trust-Based Coop%'] = (display_df['Focal_Cooperation_Rate'] * 100).apply(lambda x: f"{x:.1f}%")
        display_df['Bayesian Coop%'] = (display_df['Bayesian_Cooperation_Rate'] * 100).apply(lambda x: f"{x:.1f}%")
        display_df['Δ Payoff'] = display_df['Payoff_Difference'].apply(lambda x: f"{x:+.1f}")

        table = display_df[['Partner', 'Trust-Based Payoff', 'Bayesian Payoff', 'Δ Payoff',
                            'Trust-Based Coop%', 'Bayesian Coop%']]
        print("\n" + table.to_string(index=False))

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

    def create_visualizations_local(all_results_dict, summary_df):
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100

        # 1. Payoff comparison
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

        # 3. Payoff advantage
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

    summary_df = compute_summary_statistics_local(all_results)
    create_comparison_table_local(summary_df)
    create_visualizations_local(all_results, summary_df)

    print("\n" + "=" * 80)
    print("STUDY WITH RECOMMENDED PARAMETERS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Monte Carlo runs: {NUM_MONTE_CARLO_RUNS}, Rounds per simulation: {NUM_ROUNDS}")


if __name__ == "__main__":
    main()
