#!/usr/bin/env python3
"""
Verification script for refactored trust-bayesian agent comparison.

This script tests all core functionality to ensure the refactored structure works correctly.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
    from trust_bayesian_agent_comparison.partners import (
        AlwaysCooperatePartner,
        AlwaysDefectPartner,
        RandomPartner,
        TitForTatCooperatePartner,
        GrimTriggerPartner,
    )
    from trust_bayesian_agent_comparison.simulation import run_agent_simulation
    from trust_bayesian_agent_comparison.analysis import (
        SensitivityAnalysisManager,
        MonteCarloManager,
        compute_strategy_statistics,
    )
    from trust_bayesian_agent_comparison.config import (
        TRUST_MAX,
        LOSS_AVERSION,
        SENSITIVITY_SEEDS,
    )
    print("  ✓ All imports successful")
    return True


def test_focal_agent():
    """Test FocalAgent simulation."""
    print("\nTesting FocalAgent...")
    from trust_bayesian_agent_comparison.agents import FocalAgent
    from trust_bayesian_agent_comparison.partners import TitForTatCooperatePartner
    from trust_bayesian_agent_comparison.simulation import run_agent_simulation
    from trust_bayesian_agent_comparison.analysis import compute_strategy_statistics
    
    agent = FocalAgent(loss_aversion=2.0, mu=0.5)
    partner = TitForTatCooperatePartner()
    df = run_agent_simulation(agent, partner, num_rounds=30, seed=42)
    stats = compute_strategy_statistics(df)
    
    print(f"  ✓ Simulation completed: {len(df)} rounds")
    print(f"    - Agent cooperation: {stats['agent_coop_rate']:.3f}")
    print(f"    - Mutual cooperation: {stats['mutual_coop_rate']:.3f}")
    print(f"    - Final belief: {stats['final_belief']:.3f}")
    return True


def test_bayesian_agent():
    """Test BayesianFocalAgent simulation."""
    print("\nTesting BayesianFocalAgent...")
    from trust_bayesian_agent_comparison.agents import BayesianFocalAgent
    from trust_bayesian_agent_comparison.partners import AlwaysCooperatePartner
    from trust_bayesian_agent_comparison.simulation import run_agent_simulation
    from trust_bayesian_agent_comparison.analysis import compute_strategy_statistics
    
    agent = BayesianFocalAgent(alpha_0=0.5, beta_0=0.5)
    partner = AlwaysCooperatePartner()
    df = run_agent_simulation(agent, partner, num_rounds=30, seed=42)
    stats = compute_strategy_statistics(df)
    
    print(f"  ✓ Simulation completed: {len(df)} rounds")
    print(f"    - Agent cooperation: {stats['agent_coop_rate']:.3f}")
    print(f"    - Final belief: {stats['final_belief']:.3f}")
    return True


def test_all_partners():
    """Test all partner types."""
    print("\nTesting all partner types...")
    from trust_bayesian_agent_comparison.agents import FocalAgent
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
    from trust_bayesian_agent_comparison.simulation import run_agent_simulation
    
    partners = [
        ("AlwaysCooperate", AlwaysCooperatePartner()),
        ("AlwaysDefect", AlwaysDefectPartner()),
        ("Random", RandomPartner(p=0.5)),
        ("TitForTatCoop", TitForTatCooperatePartner()),
        ("TitForTatDefect", TitForTatDefectPartner()),
        ("GrimTrigger", GrimTriggerPartner()),
        ("Pavlov", PavlovPartner()),
        ("SuspiciousTFT", SuspiciousTitForTatPartner()),
        ("StrategicCheat", StrategicCheaterPartner(coop_length=5)),
        ("AdaptiveStrategic", AdaptiveStrategicPartner(exploration_rate=0.1)),
        ("BayesianDeceptive", BayesianDeceptivePartner(exploit_threshold=7)),
        ("Exploitative", ExploitativePartner()),
    ]
    
    for name, partner in partners:
        agent = FocalAgent(loss_aversion=2.0, mu=0.5)
        df = run_agent_simulation(agent, partner, num_rounds=10, seed=42)
        print(f"  ✓ {name}")
    
    print(f"  ✓ All {len(partners)} partner types working")
    return True


def test_configuration():
    """Test configuration values."""
    print("\nTesting configuration...")
    from trust_bayesian_agent_comparison.config import (
        TRUST_MAX,
        TRUST_MIN,
        LOSS_AVERSION,
        SENSITIVITY_SEEDS,
        DECISION_THRESHOLD,
        NUM_ROUNDS,
    )
    
    print(f"  ✓ TRUST_MAX = {TRUST_MAX}")
    print(f"  ✓ TRUST_MIN = {TRUST_MIN}")
    print(f"  ✓ LOSS_AVERSION = {LOSS_AVERSION}")
    print(f"  ✓ SENSITIVITY_SEEDS = {SENSITIVITY_SEEDS}")
    print(f"  ✓ DECISION_THRESHOLD = {DECISION_THRESHOLD:.4f}")
    print(f"  ✓ NUM_ROUNDS = {NUM_ROUNDS}")
    
    assert TRUST_MAX == 10.0, "TRUST_MAX should be 10.0"
    assert DECISION_THRESHOLD == 2/3, "Decision threshold should be 2/3"
    assert SENSITIVITY_SEEDS == (42, 43, 44), "Seeds should be (42, 43, 44)"
    
    print("  ✓ All configuration values correct")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("REFACTORED STRUCTURE VERIFICATION")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("FocalAgent", test_focal_agent),
        ("BayesianAgent", test_bayesian_agent),
        ("All Partners", test_all_partners),
        ("Configuration", test_configuration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 SUCCESS! All tests passed.")
        print("\nThe refactored structure is working correctly!")
        print("\nNext steps:")
        print("  1. Open notebooks/00_quick_start_refactored.ipynb")
        print("  2. Run: python scripts/run_sensitivity.py --help")
        print("  3. Check REFACTORING_GUIDE.md for details")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
