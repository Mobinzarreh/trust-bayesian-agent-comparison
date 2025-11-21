"""
Comprehensive Monte Carlo visualization and analysis module.

Generates all visualizations and comparison tables from Monte Carlo results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import FIGURES_DIR, RESULTS_DIR, DECISION_THRESHOLD


class MonteCarloVisualizer:
    """
    Visualizer for Monte Carlo simulation results.
    
    Creates comprehensive visualizations and comparison tables for:
    - Trust-based agent analysis
    - Bayesian agent analysis
    - Comparative performance metrics
    """
    
    def __init__(self, results_dict: Dict, figures_dir: Path = None):
        """
        Initialize visualizer.
        
        Args:
            results_dict: Dictionary mapping partner names to their Monte Carlo results
                         {partner_name: {'focal': df, 'bayesian': df}}
            figures_dir: Directory to save figures (default: FIGURES_DIR)
        """
        self.results = results_dict
        self.figures_dir = figures_dir or FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
    
    def generate_all_visualizations(self):
        """Generate all visualizations and tables."""
        print("\n" + "="*80)
        print("GENERATING MONTE CARLO VISUALIZATIONS AND ANALYSIS")
        print("="*80)
        
        # Trust-based agent analysis
        print("\n1. Trust-based agent visualizations...")
        self.plot_trust_evolution()
        self.plot_signal_evolution()
        self.plot_focal_expected_p_evolution()
        self.plot_focal_final_distributions()
        self.plot_focal_final_decisions()
        
        # Bayesian agent analysis
        print("\n2. Bayesian agent visualizations...")
        self.plot_bayesian_expected_p_evolution()
        self.plot_bayesian_final_distributions()
        self.plot_bayesian_final_decisions()
        
        # Comparative analysis
        print("\n3. Comparative analysis tables...")
        self.create_mutual_cooperation_table()
        self.create_failed_collaboration_table()
        self.create_total_payoff_table()
        
        print(f"\n✓ All visualizations saved to: {self.figures_dir}/")
    
    def plot_trust_evolution(self):
        """Plot trust evolution for focal agent across all partners."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = sns.color_palette("husl", len(self.results))
        
        for idx, (partner_name, data) in enumerate(self.results.items()):
            df = data['focal']
            
            # Compute mean and std trust across runs
            trust_by_round = df.groupby('round').agg({
                'trust_level': ['mean', 'std']
            })['trust_level']
            
            rounds = trust_by_round.index
            mean_trust = trust_by_round['mean']
            std_trust = trust_by_round['std']
            
            # Plot mean with confidence band
            ax.plot(rounds, mean_trust, label=partner_name, 
                   linewidth=2, color=colors[idx], alpha=0.8)
            ax.fill_between(rounds, mean_trust - std_trust, mean_trust + std_trust,
                           alpha=0.15, color=colors[idx])
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trust Level (t)', fontsize=12, fontweight='bold')
        ax.set_title('Trust-Based Agent: Trust Evolution Across Partners', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'focal_trust_evolution.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_signal_evolution(self):
        """Plot signal evolution for focal agent across all partners."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = sns.color_palette("husl", len(self.results))
        
        for idx, (partner_name, data) in enumerate(self.results.items()):
            df = data['focal']
            
            # Compute mean and std signal across runs
            signal_by_round = df.groupby('round').agg({
                'signal': ['mean', 'std']
            })['signal']
            
            rounds = signal_by_round.index
            mean_signal = signal_by_round['mean']
            std_signal = signal_by_round['std']
            
            # Plot mean with confidence band
            ax.plot(rounds, mean_signal, label=partner_name,
                   linewidth=2, color=colors[idx], alpha=0.8)
            ax.fill_between(rounds, mean_signal - std_signal, mean_signal + std_signal,
                           alpha=0.15, color=colors[idx])
        
        # Add threshold line
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                  label='Decision Threshold (0.5)', alpha=0.7)
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal (x)', fontsize=12, fontweight='bold')
        ax.set_title('Trust-Based Agent: Signal Evolution Across Partners',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'focal_signal_evolution.png',
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_focal_expected_p_evolution(self):
        """Plot expected cooperation probability evolution for focal agent."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = sns.color_palette("husl", len(self.results))
        threshold = DECISION_THRESHOLD
        
        for idx, (partner_name, data) in enumerate(self.results.items()):
            df = data['focal']
            
            # Compute mean expected_p across runs
            expected_p_by_round = df.groupby('round').agg({
                'agent_belief': ['mean', 'std']
            })['agent_belief']
            
            rounds = expected_p_by_round.index
            mean_p = expected_p_by_round['mean']
            std_p = expected_p_by_round['std']
            
            ax.plot(rounds, mean_p, label=partner_name,
                   linewidth=2, color=colors[idx], alpha=0.8)
            ax.fill_between(rounds, mean_p - std_p, mean_p + std_p,
                           alpha=0.15, color=colors[idx])
        
        # Add decision threshold
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                  label=f'Decision Threshold ({threshold:.3f})', alpha=0.7)
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Cooperation Probability E[p]', fontsize=12, fontweight='bold')
        ax.set_title('Trust-Based Agent: Evolution of Expected Cooperation Probability',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'focal_expected_p_evolution.png',
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_focal_final_distributions(self):
        """Plot final round belief distributions for focal agent."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = sns.color_palette("husl", len(self.results))
        threshold = DECISION_THRESHOLD
        
        for idx, (partner_name, data) in enumerate(self.results.items()):
            df = data['focal']
            
            # Get final round data
            max_round = df['round'].max()
            final_df = df[df['round'] == max_round]
            
            # Plot KDE of final beliefs
            final_beliefs = final_df['agent_belief'].values
            x_vals = np.linspace(0, 1, 200)
            
            # Compute KDE
            if len(final_beliefs) > 1:
                kde = stats.gaussian_kde(final_beliefs)
                density = kde(x_vals)
                ax.plot(x_vals, density, label=f"{partner_name} (mean={np.mean(final_beliefs):.3f})",
                       linewidth=2, color=colors[idx], alpha=0.8)
        
        # Add decision threshold
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                  label=f'Decision Threshold ({threshold:.3f})', alpha=0.7)
        
        ax.set_xlabel('Expected Cooperation Probability E[p]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Trust-Based Agent: Final Round Belief Distributions',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'focal_final_distributions.png',
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_focal_final_decisions(self):
        """Print focal agent's final round decision trends."""
        print("\n" + "="*80)
        print("FOCAL AGENT: Decision-Making Trends in the Last Round")
        print("="*80)
        
        decisions_data = []
        
        for partner_name, data in self.results.items():
            df = data['focal']
            
            # Get final round
            max_round = df['round'].max()
            final_df = df[df['round'] == max_round]
            
            mean_belief = final_df['agent_belief'].mean()
            coop_rate = final_df['agent_action'].mean()
            
            decisions_data.append({
                'Partner': partner_name,
                'Mean E[p]': f"{mean_belief:.3f}",
                'Cooperation Rate': f"{coop_rate:.1%}",
                'Choice': 'Cooperate' if mean_belief > DECISION_THRESHOLD else 'Defect'
            })
        
        decisions_df = pd.DataFrame(decisions_data)
        print(decisions_df.to_string(index=False))
        print("="*80)
        
        # Save to file
        decisions_df.to_csv(RESULTS_DIR / 'focal_final_decisions.csv', index=False)
    
    def plot_bayesian_expected_p_evolution(self):
        """Plot expected cooperation probability evolution for Bayesian agent."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = sns.color_palette("husl", len(self.results))
        threshold = DECISION_THRESHOLD
        
        for idx, (partner_name, data) in enumerate(self.results.items()):
            df = data['bayesian']
            
            # Compute mean expected_p across runs
            expected_p_by_round = df.groupby('round').agg({
                'agent_belief': ['mean', 'std']
            })['agent_belief']
            
            rounds = expected_p_by_round.index
            mean_p = expected_p_by_round['mean']
            std_p = expected_p_by_round['std']
            
            ax.plot(rounds, mean_p, label=partner_name,
                   linewidth=2, color=colors[idx], alpha=0.8)
            ax.fill_between(rounds, mean_p - std_p, mean_p + std_p,
                           alpha=0.15, color=colors[idx])
        
        # Add decision threshold
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                  label=f'Decision Threshold ({threshold:.3f})', alpha=0.7)
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Cooperation Probability E[p]', fontsize=12, fontweight='bold')
        ax.set_title('Bayesian Agent: Evolution of Expected Cooperation Probability',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bayesian_expected_p_evolution.png',
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_bayesian_final_distributions(self):
        """Plot final round Beta distributions for Bayesian agent."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = sns.color_palette("husl", len(self.results))
        threshold = DECISION_THRESHOLD
        p_vals = np.linspace(0, 1, 200)
        
        for idx, (partner_name, data) in enumerate(self.results.items()):
            df = data['bayesian']
            
            # Get final round alpha and beta
            max_round = df['round'].max()
            final_df = df[df['round'] == max_round]
            
            # Average alpha and beta across runs
            mean_alpha = final_df['alpha'].mean() if 'alpha' in final_df.columns else final_df['history_alpha'].apply(lambda x: x[-1] if isinstance(x, list) else x).mean()
            mean_beta = final_df['beta'].mean() if 'beta' in final_df.columns else final_df['history_beta'].apply(lambda x: x[-1] if isinstance(x, list) else x).mean()
            
            posterior_mean = mean_alpha / (mean_alpha + mean_beta)
            
            # Plot Beta PDF
            pdf_vals = stats.beta.pdf(p_vals, mean_alpha, mean_beta)
            ax.plot(p_vals, pdf_vals,
                   label=f"{partner_name} (α={mean_alpha:.1f}, β={mean_beta:.1f})",
                   linewidth=2, color=colors[idx], alpha=0.8)
            
            # Mark posterior mean
            ax.axvline(x=posterior_mean, linestyle=':', linewidth=1,
                      alpha=0.5, color=colors[idx])
        
        # Add decision threshold
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                  label=f'Decision Threshold ({threshold:.3f})', alpha=0.7)
        
        ax.set_xlabel('Expected Cooperation Probability p', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density f(p)', fontsize=12, fontweight='bold')
        ax.set_title('Bayesian Agent: Final Round Beta Distributions',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bayesian_final_distributions.png',
                   bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_bayesian_final_decisions(self):
        """Print Bayesian agent's final round decision trends."""
        print("\n" + "="*80)
        print("BAYESIAN AGENT: Decision-Making Trends in the Last Round")
        print("="*80)
        
        decisions_data = []
        
        for partner_name, data in self.results.items():
            df = data['bayesian']
            
            # Get final round
            max_round = df['round'].max()
            final_df = df[df['round'] == max_round]
            
            mean_belief = final_df['agent_belief'].mean()
            coop_rate = final_df['agent_action'].mean()
            
            decisions_data.append({
                'Partner': partner_name,
                'Mean E[p]': f"{mean_belief:.3f}",
                'Cooperation Rate': f"{coop_rate:.1%}",
                'Choice': 'Cooperate' if mean_belief > DECISION_THRESHOLD else 'Defect'
            })
        
        decisions_df = pd.DataFrame(decisions_data)
        print(decisions_df.to_string(index=False))
        print("="*80)
        
        # Save to file
        decisions_df.to_csv(RESULTS_DIR / 'bayesian_final_decisions.csv', index=False)
    
    def create_mutual_cooperation_table(self):
        """Create mutual cooperation rate comparison table."""
        print("\n" + "="*130)
        print("MONTE CARLO: SUCCESSFUL COLLABORATION RATE COMPARISON")
        num_runs = len(self.results[list(self.results.keys())[0]]['focal']['run_id'].unique())
        print(f"Based on {num_runs} runs with paired seeds per partner")
        print("KPI: Mutual Cooperation Rate (both agent and partner choose to collaborate)")
        print("="*130)
        
        comparison_data = []
        
        for partner_name, data in self.results.items():
            df_focal = data['focal']
            df_bayesian = data['bayesian']
            
            # Compute mutual cooperation rate per run
            focal_mutual = df_focal.groupby('run_id').apply(
                lambda x: ((x['agent_action'] == 1) & (x['partner_action'] == 1)).mean(),
                include_groups=False
            )
            bayesian_mutual = df_bayesian.groupby('run_id').apply(
                lambda x: ((x['agent_action'] == 1) & (x['partner_action'] == 1)).mean(),
                include_groups=False
            )
            
            # Statistics
            focal_mean = focal_mutual.mean()
            focal_std = focal_mutual.std()
            bayesian_mean = bayesian_mutual.mean()
            bayesian_std = bayesian_mutual.std()
            difference = focal_mean - bayesian_mean
            
            # Statistical test
            t_stat, p_value = stats.ttest_rel(focal_mutual, bayesian_mutual)
            
            # Percent trust better (higher mutual cooperation is better)
            pct_trust_better = (focal_mutual > bayesian_mutual).mean()
            
            # Significance
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            comparison_data.append({
                'Partner Strategy': partner_name,
                'Trust-Based Mean': f"{focal_mean:.1%}",
                'Trust-Based Std': f"{focal_std:.1%}",
                'Bayesian Mean': f"{bayesian_mean:.1%}",
                'Bayesian Std': f"{bayesian_std:.1%}",
                'Difference (Trust - Bayes)': f"{difference:+.1%}",
                'Sig': sig,
                '% Trust Better': f"{pct_trust_better:.0%}",
                'Diff_Raw': difference
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df_sorted = comparison_df.sort_values('Diff_Raw', ascending=False)
        
        display_df = comparison_df_sorted.drop('Diff_Raw', axis=1)
        print(display_df.to_string(index=False))
        print("="*130)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        trust_wins = (comparison_df['Diff_Raw'] > 0).sum()
        bayes_wins = (comparison_df['Diff_Raw'] < 0).sum()
        ties = (comparison_df['Diff_Raw'] == 0).sum()
        
        print(f"Trust-Based Wins:     {trust_wins} partners ({trust_wins/len(comparison_df):.0%})")
        print(f"Bayesian Wins:        {bayes_wins} partners ({bayes_wins/len(comparison_df):.0%})")
        print(f"Ties:                 {ties} partners ({ties/len(comparison_df):.0%})")
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("\nINTERPRETATION:")
        print("  - Positive difference means Trust-Based achieves MORE mutual cooperation (better)")
        print("  - Negative difference means Bayesian achieves MORE mutual cooperation (better)")
        print("  - This metric represents COLLABORATIVE SUCCESS RATE")
        
        # Save
        display_df.to_csv(RESULTS_DIR / 'mutual_cooperation_comparison.csv', index=False)
    
    def create_failed_collaboration_table(self):
        """Create failed collaboration (betrayal) rate comparison table."""
        print("\n" + "="*130)
        print("MONTE CARLO: UNSUCCESSFUL COLLABORATION (BETRAYAL) RATE COMPARISON")
        num_runs = len(self.results[list(self.results.keys())[0]]['focal']['run_id'].unique())
        print(f"Based on {num_runs} runs with paired seeds per partner")
        print("KPI: Betrayal Rate (agent cooperates but partner defects) - LOWER IS BETTER")
        print("="*130)
        
        comparison_data = []
        
        for partner_name, data in self.results.items():
            df_focal = data['focal']
            df_bayesian = data['bayesian']
            
            # Compute betrayal rate per run (agent cooperates, partner defects)
            focal_betrayal = df_focal.groupby('run_id').apply(
                lambda x: ((x['agent_action'] == 1) & (x['partner_action'] == 0)).mean(),
                include_groups=False
            )
            bayesian_betrayal = df_bayesian.groupby('run_id').apply(
                lambda x: ((x['agent_action'] == 1) & (x['partner_action'] == 0)).mean(),
                include_groups=False
            )
            
            # Statistics
            focal_mean = focal_betrayal.mean()
            focal_std = focal_betrayal.std()
            bayesian_mean = bayesian_betrayal.mean()
            bayesian_std = bayesian_betrayal.std()
            difference = focal_mean - bayesian_mean
            
            # Statistical test
            t_stat, p_value = stats.ttest_rel(focal_betrayal, bayesian_betrayal)
            
            # Percent trust better (lower betrayal is better, so trust better when focal < bayesian)
            pct_trust_better = (focal_betrayal < bayesian_betrayal).mean()
            
            # Significance
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            comparison_data.append({
                'Partner Strategy': partner_name,
                'Trust-Based Mean': f"{focal_mean:.1%}",
                'Trust-Based Std': f"{focal_std:.1%}",
                'Bayesian Mean': f"{bayesian_mean:.1%}",
                'Bayesian Std': f"{bayesian_std:.1%}",
                'Difference (Trust - Bayes)': f"{difference:+.1%}",
                'Sig': sig,
                '% Trust Better': f"{pct_trust_better:.0%}",
                'Diff_Raw': difference
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df_sorted = comparison_df.sort_values('Diff_Raw', ascending=True)  # Lower is better
        
        display_df = comparison_df_sorted.drop('Diff_Raw', axis=1)
        print(display_df.to_string(index=False))
        print("="*130)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        trust_wins = (comparison_df['Diff_Raw'] < 0).sum()  # Negative diff is better (lower betrayal)
        bayes_wins = (comparison_df['Diff_Raw'] > 0).sum()
        ties = (comparison_df['Diff_Raw'] == 0).sum()
        
        print(f"Trust-Based Lower Betrayal:  {trust_wins} partners ({trust_wins/len(comparison_df):.0%})")
        print(f"Bayesian Lower Betrayal:     {bayes_wins} partners ({bayes_wins/len(comparison_df):.0%})")
        print(f"Ties:                        {ties} partners ({ties/len(comparison_df):.0%})")
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("\nINTERPRETATION:")
        print("  - Negative difference means Trust-Based has LOWER betrayal rate (better - less exploited)")
        print("  - Positive difference means Bayesian has LOWER betrayal rate (better - less exploited)")
        print("  - This metric represents VULNERABILITY TO EXPLOITATION")
        
        # Save
        display_df.to_csv(RESULTS_DIR / 'failed_collaboration_comparison.csv', index=False)
    
    def create_total_payoff_table(self):
        """Create total payoff comparison table."""
        print("\n" + "="*130)
        print("MONTE CARLO: TOTAL PAYOFF COMPARISON")
        num_runs = len(self.results[list(self.results.keys())[0]]['focal']['run_id'].unique())
        num_rounds = self.results[list(self.results.keys())[0]]['focal']['round'].max() + 1
        print(f"Based on {num_runs} runs with paired seeds per partner")
        print(f"KPI: Sum of Payoffs over {num_rounds} rounds - HIGHER IS BETTER")
        print("="*130)
        
        comparison_data = []
        
        for partner_name, data in self.results.items():
            df_focal = data['focal']
            df_bayesian = data['bayesian']
            
            # Compute total payoff per run
            focal_payoffs = df_focal.groupby('run_id')['agent_payoff'].sum()
            bayesian_payoffs = df_bayesian.groupby('run_id')['agent_payoff'].sum()
            
            # Statistics
            focal_mean = focal_payoffs.mean()
            focal_std = focal_payoffs.std()
            bayesian_mean = bayesian_payoffs.mean()
            bayesian_std = bayesian_payoffs.std()
            difference = focal_mean - bayesian_mean
            
            # Statistical test
            t_stat, p_value = stats.ttest_rel(focal_payoffs, bayesian_payoffs)
            
            # Significance
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            
            # Percent trust better (higher payoff is better)
            pct_trust_better = (focal_payoffs > bayesian_payoffs).mean()
            
            comparison_data.append({
                'Partner Strategy': partner_name,
                'Trust-Based Mean': f"{focal_mean:.1f}",
                'Trust-Based Std': f"{focal_std:.1f}",
                'Bayesian Mean': f"{bayesian_mean:.1f}",
                'Bayesian Std': f"{bayesian_std:.1f}",
                'Difference (Trust - Bayes)': f"{difference:+.1f}",
                'Sig': sig,
                '% Trust Better': f"{pct_trust_better:.0%}",
                'Diff_Raw': difference
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df_sorted = comparison_df.sort_values('Diff_Raw', ascending=False)
        
        display_df = comparison_df_sorted.drop('Diff_Raw', axis=1)
        print(display_df.to_string(index=False))
        print("="*130)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        trust_wins = (comparison_df['Diff_Raw'] > 0).sum()
        bayes_wins = (comparison_df['Diff_Raw'] < 0).sum()
        ties = (comparison_df['Diff_Raw'] == 0).sum()
        
        print(f"Trust-Based Higher Payoff:  {trust_wins} partners ({trust_wins/len(comparison_df):.0%})")
        print(f"Bayesian Higher Payoff:     {bayes_wins} partners ({bayes_wins/len(comparison_df):.0%})")
        print(f"Ties:                       {ties} partners ({ties/len(comparison_df):.0%})")
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("\nINTERPRETATION:")
        print("  - Positive difference means Trust-Based achieves HIGHER total payoff (better performance)")
        print("  - Negative difference means Bayesian achieves HIGHER total payoff (better performance)")
        print("  - This metric represents OVERALL GAME-THEORETIC PERFORMANCE (final score)")
        
        # Save
        display_df.to_csv(RESULTS_DIR / 'total_payoff_comparison.csv', index=False)
