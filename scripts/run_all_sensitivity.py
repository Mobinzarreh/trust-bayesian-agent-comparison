#!/usr/bin/env python3
"""
Run comprehensive sensitivity analysis for ALL agent parameters.
Each parameter is varied individually while others are held at default values.

Parameters analyzed:
1. ETA (learning rate for signal)
2. MEMORY_DISCOUNT (recency weighting for signal)
3. TRUST_DISCOUNT (recency weighting for trust)
4. TRUST_SMOOTHING (trust update smoothing)
5. LOSS_AVERSION (betrayal penalty multiplier)
6. LAMBDA_SURPRISE (surprise penalty multiplier)
7. INVERSE_TEMPERATURE (exploration-exploitation trade-off)

Usage:
    python scripts/run_all_sensitivity.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trust_bayesian_agent_comparison.analysis.sensitivity import SensitivityAnalysisManager
from trust_bayesian_agent_comparison.config import (
    ETA, MEMORY_DISCOUNT, TRUST_DISCOUNT, TRUST_SMOOTHING,
    LOSS_AVERSION, LAMBDA_SURPRISE, INVERSE_TEMPERATURE, SENSITIVITY_SEEDS
)
from trust_bayesian_agent_comparison.partners import (
    SingleCyclePartner,
    GradualDeteriorationPartner,
    ExpectationViolationPartner,
    StrategicCheaterPartner,
    TitForTatCooperatePartner,
)

# Define partners to analyze
PARTNERS = {
    'SingleCycle': (lambda: SingleCyclePartner(num_rounds=100, cooperate_fraction=0.3), 'down'),
    'GradualDeterioration': (lambda: GradualDeteriorationPartner(deterioration_rate=0.8, num_rounds=100), 'down'),
    'ExpectationViolation': (lambda: ExpectationViolationPartner(), None),
    'StrategicCheater': (lambda: StrategicCheaterPartner(t_threshold=5.0), 'down'),
    'TitForTat': (lambda: TitForTatCooperatePartner(), 'up'),
}

# Parameter configurations: (name, grid, default)
PARAMETER_CONFIGS = {
    'eta': {
        'grid': np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        'default': ETA,
        'description': 'Learning rate for signal update'
    },
    'memory_discount': {
        'grid': np.array([0.5, 0.7, 0.8, 0.9, 0.95]),
        'default': MEMORY_DISCOUNT,
        'description': 'Recency weighting for signal'
    },
    'trust_discount': {
        'grid': np.array([0.5, 0.7, 0.8, 0.9, 0.95]),
        'default': TRUST_DISCOUNT,
        'description': 'Recency weighting for trust'
    },
    'trust_smoothing': {
        'grid': np.array([0.1, 0.2, 0.3, 0.5, 0.7]),
        'default': TRUST_SMOOTHING,
        'description': 'Trust update smoothing factor'
    },
    'loss_aversion': {
        'grid': np.array([1.0, 1.5, 2.0, 3.0, 5.0]),
        'default': LOSS_AVERSION,
        'description': 'Betrayal penalty multiplier'
    },
    'lambda_surprise': {
        'grid': np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        'default': LAMBDA_SURPRISE,
        'description': 'Surprise penalty multiplier'
    },
    'inverse_temperature': {
        'grid': np.array([0.5, 2.0, 5.0, 10.0]),
        'default': INVERSE_TEMPERATURE,
        'description': 'Exploration-exploitation trade-off'
    },
}


def run_single_parameter_sweep(param_name: str, results_base_dir: Path):
    """Run sensitivity analysis varying only one parameter."""
    
    config = PARAMETER_CONFIGS[param_name]
    param_grid = config['grid']
    
    print(f"\n{'='*80}")
    print(f"PARAMETER: {param_name}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    print(f"Default value: {config['default']}")
    print(f"Grid values: {param_grid}")
    print(f"Total simulations per partner: {len(param_grid) * len(SENSITIVITY_SEEDS)}")
    
    # Create parameter-specific results directory
    results_dir = results_base_dir / f"sensitivity_{param_name}"
    manager = SensitivityAnalysisManager(results_base_dir=results_dir)
    
    # Build sweep kwargs: fix all params at default, vary only target param
    sweep_kwargs = {
        'eta_grid': np.array([ETA]),
        'memory_discount_grid': np.array([MEMORY_DISCOUNT]),
        'trust_discount_grid': np.array([TRUST_DISCOUNT]),
        'trust_smoothing_grid': np.array([TRUST_SMOOTHING]),
        'loss_aversion_grid': np.array([LOSS_AVERSION]),
        'lambda_surprise_grid': np.array([LAMBDA_SURPRISE]),
        'inverse_temperature_grid': np.array([INVERSE_TEMPERATURE]),
        'seeds': SENSITIVITY_SEEDS,
    }
    
    # Override the target parameter with its grid
    sweep_kwargs[f'{param_name}_grid'] = param_grid
    
    # Run analysis for each partner
    all_results = {}
    for partner_name, (factory, threshold_dir) in PARTNERS.items():
        print(f"\n  Processing: {partner_name}...")
        
        result = manager.run_analysis(
            partner_name=partner_name,
            partner_factory=factory,
            threshold_direction=threshold_dir,
            overwrite=True,
            **sweep_kwargs
        )
        
        all_results[partner_name] = result
        print(f"    ✓ {len(result)} simulations complete")
    
    print(f"\n  Results saved to: {results_dir}")
    return all_results


def create_summary_table(results_base_dir: Path):
    """Create a summary table of optimal parameters for each partner."""
    
    summary_rows = []
    
    for param_name in PARAMETER_CONFIGS.keys():
        results_dir = results_base_dir / f"sensitivity_{param_name}"
        
        for partner_name in PARTNERS.keys():
            csv_path = results_dir / f"{partner_name}_results.csv"
            if not csv_path.exists():
                continue
                
            df = pd.read_csv(csv_path)
            
            # Find optimal value for each metric
            means = df.groupby(param_name)[['mutual_coop_rate', 'betrayal_rate', 'total_payoff']].mean()
            
            best_mutual_coop_val = means['mutual_coop_rate'].idxmax()
            best_betrayal_val = means['betrayal_rate'].idxmin()
            best_payoff_val = means['total_payoff'].idxmax()
            
            summary_rows.append({
                'parameter': param_name,
                'partner': partner_name,
                'best_for_mutual_coop': best_mutual_coop_val,
                'mutual_coop_rate': means.loc[best_mutual_coop_val, 'mutual_coop_rate'],
                'best_for_low_betrayal': best_betrayal_val,
                'betrayal_rate': means.loc[best_betrayal_val, 'betrayal_rate'],
                'best_for_payoff': best_payoff_val,
                'total_payoff': means.loc[best_payoff_val, 'total_payoff'],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_base_dir / "sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    return summary_df


def main():
    print("="*80)
    print("COMPREHENSIVE SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(PARTNERS)} partners:")
    for name in PARTNERS.keys():
        print(f"  - {name}")
    
    print(f"\nParameters to sweep ({len(PARAMETER_CONFIGS)}):")
    for param, config in PARAMETER_CONFIGS.items():
        print(f"  - {param}: {config['grid']} (default={config['default']})")
    
    print(f"\nSeeds: {SENSITIVITY_SEEDS}")
    print(f"\nMetrics collected:")
    print("  1. Mutual Cooperation Rate")
    print("  2. Betrayal Rate")
    print("  3. Total Payoff")
    
    results_base_dir = project_root / "results"
    
    # Run sweep for each parameter
    for param_name in PARAMETER_CONFIGS.keys():
        run_single_parameter_sweep(param_name, results_base_dir)
    
    # Create summary table
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLE")
    print("="*80)
    summary = create_summary_table(results_base_dir)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nResults directories created:")
    for param_name in PARAMETER_CONFIGS.keys():
        print(f"  - results/sensitivity_{param_name}/")
    print(f"  - results/sensitivity_summary.csv")
    
    print("\nNext steps:")
    print("  1. Open notebooks/All_Parameters_Sensitivity_Analysis.ipynb")
    print("  2. Visualize results for each parameter")
    print("  3. Compare optimal configurations across partners")


if __name__ == "__main__":
    main()
