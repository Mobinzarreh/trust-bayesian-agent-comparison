#!/usr/bin/env python3
"""
Run sensitivity analysis for INVERSE_TEMPERATURE parameter only.
Focuses on the three key metrics: mutual cooperation, betrayal rate, and total payoff.

Usage:
    python scripts/run_inverse_temp_sensitivity.py
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trust_bayesian_agent_comparison.analysis.sensitivity import SensitivityAnalysisManager
from trust_bayesian_agent_comparison.config import (
    ETA, MEMORY_DISCOUNT, TRUST_DISCOUNT, TRUST_SMOOTHING,
    LOSS_AVERSION, LAMBDA_SURPRISE, SENSITIVITY_SEEDS
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

# INVERSE_TEMPERATURE grid
INVERSE_TEMP_GRID = np.array([0.5, 2.0, 5.0, 10.0])

# Fix all other parameters at their default values (single value grids)
FIXED_PARAMS = {
    'eta_grid': np.array([ETA]),
    'memory_discount_grid': np.array([MEMORY_DISCOUNT]),
    'trust_discount_grid': np.array([TRUST_DISCOUNT]),
    'trust_smoothing_grid': np.array([TRUST_SMOOTHING]),
    'loss_aversion_grid': np.array([LOSS_AVERSION]),
    'lambda_surprise_grid': np.array([LAMBDA_SURPRISE]),
    'inverse_temperature_grid': INVERSE_TEMP_GRID,
    'seeds': SENSITIVITY_SEEDS,
}

def main():
    print("="*80)
    print("INVERSE TEMPERATURE SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(PARTNERS)} partners:")
    for name in PARTNERS.keys():
        print(f"  - {name}")
    
    print(f"\nInverse Temperature values: {INVERSE_TEMP_GRID}")
    print(f"Seeds: {SENSITIVITY_SEEDS}")
    print(f"Total simulations per partner: {len(INVERSE_TEMP_GRID) * len(SENSITIVITY_SEEDS)}")
    
    print("\nMetrics collected:")
    print("  1. Mutual Cooperation Rate (both cooperate)")
    print("  2. Betrayal Rate (agent cooperates, partner defects)")
    print("  3. Total Payoff (sum over all rounds)")
    
    print("\n" + "="*80)
    
    # Setup manager with custom results directory
    results_dir = project_root / "results" / "sensitivity_inverse_temp"
    manager = SensitivityAnalysisManager(results_base_dir=results_dir)
    
    # Run analysis for each partner
    for partner_name, (factory, threshold_dir) in PARTNERS.items():
        print(f"\n{'='*80}")
        print(f"Processing: {partner_name}")
        print(f"{'='*80}")
        
        result = manager.run_analysis(
            partner_name=partner_name,
            partner_factory=factory,
            threshold_direction=threshold_dir,
            overwrite=True,
            **FIXED_PARAMS
        )
        
        # Show preview of results
        print(f"\n✓ Completed {len(result)} simulations")
        print(f"\nResults preview:")
        print(result[['inverse_temperature', 'seed', 'mutual_coop_rate', 
                      'betrayal_rate', 'total_payoff']].head(8).to_string(index=False))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {results_dir}")
    print("\nNext steps:")
    print("  1. Open the analysis notebook to visualize results")
    print("  2. Inspect plots to identify optimal inverse_temperature")
    print("  3. Compare performance across partners")
    print("\nFiles created:")
    for partner_name in PARTNERS.keys():
        csv_file = results_dir / f"{partner_name}_results.csv"
        print(f"  - {csv_file.name}")

if __name__ == "__main__":
    main()
