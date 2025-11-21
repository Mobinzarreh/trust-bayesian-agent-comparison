#!/usr/bin/env python3
"""
Command-line interface for running sensitivity analysis.

Usage:
    python scripts/run_sensitivity.py --partners TitForTatCoop AlwaysDefect --overwrite
    python scripts/run_sensitivity.py --all --seeds 42 43 44 45 46
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trust_bayesian_agent_comparison.analysis.sensitivity import SensitivityAnalysisManager
from trust_bayesian_agent_comparison.config import SENSITIVITY_SEEDS
import numpy as np


# Partner registry
def get_partner_configs():
    """Define all available partners for sensitivity analysis."""
    from trust_bayesian_agent_comparison.partners import (
        AlwaysCooperatePartner,
        AlwaysDefectPartner,
        RandomPartner,
        TitForTatCooperatePartner,
        TitForTatDefectPartner,
        GrimTriggerPartner,
        PavlovPartner,
        SuspiciousTitForTatPartner,
        StrategicCheaterPartner,
        AdaptiveStrategicPartner,
        BayesianDeceptivePartner,
        ExploitativePartner,
    )
    
    return {
        # Fixed strategies
        "AlwaysCoop": (lambda: AlwaysCooperatePartner(), "up"),
        "AlwaysDefect": (lambda: AlwaysDefectPartner(), "down"),
        "Random": (lambda: RandomPartner(p=0.5), "neutral"),
        
        # Reactive strategies
        "TitForTatCoop": (lambda: TitForTatCooperatePartner(), "up"),
        "TitForTatDefect": (lambda: TitForTatDefectPartner(), "down"),
        "GrimTrigger": (lambda: GrimTriggerPartner(), "up"),
        "Pavlov": (lambda: PavlovPartner(), "neutral"),
        "SuspiciousTFT": (lambda: SuspiciousTitForTatPartner(), "down"),
        
        # Adaptive/Deceptive strategies
        "StrategicCheat": (lambda: StrategicCheaterPartner(coop_length=5), "down"),
        "AdaptiveStrategic": (lambda: AdaptiveStrategicPartner(exploration_rate=0.1), "down"),
        "BayesianDeceptive": (lambda: BayesianDeceptivePartner(exploit_threshold=7), "down"),
        "Exploitative": (lambda: ExploitativePartner(), "down"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for trust-based agents"
    )
    
    parser.add_argument(
        '--partners',
        nargs='+',
        help='Partner names to analyze (space-separated)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run analysis for all partners'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )
    
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=list(SENSITIVITY_SEEDS),
        help='Random seeds for robustness (default: 42 43 44)'
    )
    
    parser.add_argument(
        '--eta-points',
        type=int,
        default=6,
        help='Number of points in eta grid (default: 6)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (default: -1 = all cores)'
    )
    
    args = parser.parse_args()
    
    # Get partner configurations
    all_partners = get_partner_configs()
    
    # Determine which partners to run
    if args.all:
        partner_names = list(all_partners.keys())
    elif args.partners:
        partner_names = args.partners
        # Validate
        invalid = [p for p in partner_names if p not in all_partners]
        if invalid:
            print(f"Error: Unknown partners: {invalid}")
            print(f"Available: {list(all_partners.keys())}")
            sys.exit(1)
    else:
        print("Error: Specify --partners or --all")
        parser.print_help()
        sys.exit(1)
    
    # Setup manager
    manager = SensitivityAnalysisManager()
    
    # Custom grids if specified
    sweep_kwargs = {
        'seeds': tuple(args.seeds),
        'n_jobs': args.n_jobs,
    }
    
    if args.eta_points != 6:
        sweep_kwargs['eta_grid'] = np.linspace(0.0, 1.0, args.eta_points)
    
    # Run analysis
    print(f"Running sensitivity analysis for: {partner_names}")
    print(f"Seeds: {args.seeds}")
    print(f"Overwrite: {args.overwrite}")
    print("-" * 60)
    
    for partner_name in partner_names:
        factory, threshold_dir = all_partners[partner_name]
        
        print(f"\nProcessing {partner_name}...")
        result = manager.run_analysis(
            partner_name=partner_name,
            partner_factory=factory,
            threshold_direction=threshold_dir,
            overwrite=args.overwrite,
            **sweep_kwargs
        )
        
        print(f"  ✓ Completed: {len(result)} simulations")
    
    print("\n" + "=" * 60)
    print("Sensitivity analysis complete!")
    print(f"Results saved in: {manager.results_dir}")


if __name__ == "__main__":
    main()
