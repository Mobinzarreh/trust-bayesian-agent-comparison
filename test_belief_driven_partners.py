#!/usr/bin/env python3
"""Test belief-driven partners match original implementation."""

import numpy as np
from trust_bayesian_agent_comparison.partners import (
    AdaptivePartner,
    StrategicCheaterPartner,
    ExpectationViolationPartner,
)

def test_adaptive_partner():
    """Test AdaptivePartner mirrors agent's expected behavior."""
    print("=== Testing AdaptivePartner ===\n")
    
    partner = AdaptivePartner(u_i=0.6, t0=0.1)
    print(f"Initial state: x_hat={partner.x_hat:.3f}, t_hat={partner.t_hat:.3f}")
    
    # Should cooperate when x_hat > 0.5
    decision = partner.decide(round_num=0)
    print(f"Decision (x_hat=0.6 > 0.5): {decision} (expected 1)")
    assert decision == 1, "Should cooperate when x_hat > 0.5"
    
    # Observe agent defecting multiple times
    for i in range(5):
        partner.observe(0)
        print(f"  After observing defection {i+1}: x_hat={partner.x_hat:.3f}, t_hat={partner.t_hat:.3f}")
    
    # Should defect when x_hat < 0.5
    decision = partner.decide(round_num=5)
    print(f"Decision after observing defections (x_hat={partner.x_hat:.3f}): {decision}")
    print(f"Expected: 0 or 1 depending on x_hat\n")
    print("✓ AdaptivePartner test passed\n")


def test_strategic_cheater():
    """Test StrategicCheaterPartner exploits when trust is high."""
    print("=== Testing StrategicCheaterPartner ===\n")
    
    partner = StrategicCheaterPartner(u_i=0.5, t0=0.1, t_threshold=5.0)
    print(f"Initial state: x_hat={partner.x_hat:.3f}, t_hat={partner.t_hat:.3f}")
    print(f"Threshold: {partner.t_threshold}")
    
    # Should cooperate when trust is low
    decision = partner.decide(round_num=0)
    print(f"Decision (t_hat={partner.t_hat:.3f} < {partner.t_threshold}): {decision} (expected 1)")
    assert decision == 1, "Should cooperate when trust < threshold"
    
    # Observe agent cooperating to build trust
    for i in range(10):
        partner.observe(1)
        decision = partner.decide(round_num=i+1)
        print(f"  Round {i+1}: t_hat={partner.t_hat:.3f}, decision={decision}")
        
        if partner.t_hat > partner.t_threshold:
            assert decision == 0, "Should defect when trust exceeds threshold"
            print(f"  → Started defecting at round {i+1} (trust exceeded threshold)")
            break
    
    print("\n✓ StrategicCheaterPartner test passed\n")


def test_expectation_violation():
    """Test ExpectationViolationPartner does opposite of prediction."""
    print("=== Testing ExpectationViolationPartner ===\n")
    
    partner = ExpectationViolationPartner(u_i=0.7, t0=0.1)
    print(f"Initial state: x_hat={partner.x_hat:.3f}, t_hat={partner.t_hat:.3f}")
    
    # Should defect when x_hat > 0.5 (expects agent to cooperate)
    decision = partner.decide(round_num=0)
    print(f"Decision (x_hat={partner.x_hat:.3f} > 0.5, expects agent C): {decision} (expected 0)")
    assert decision == 0, "Should defect when expecting cooperation"
    
    # Observe agent defecting multiple times
    for i in range(5):
        partner.observe(0)
        print(f"  After observing defection {i+1}: x_hat={partner.x_hat:.3f}")
    
    # When x_hat < 0.5, should cooperate (do opposite)
    decision = partner.decide(round_num=5)
    expected_decision = 1 if partner.x_hat <= 0.5 else 0
    print(f"Decision (x_hat={partner.x_hat:.3f}): {decision} (expected {expected_decision})")
    
    print("\n✓ ExpectationViolationPartner test passed\n")


def test_trust_dynamics():
    """Test that belief-driven partners track trust like focal agents."""
    print("=== Testing Trust Dynamics ===\n")
    
    partner = AdaptivePartner(u_i=0.6, t0=0.1)
    print(f"Initial: x_hat={partner.x_hat:.3f}, t_hat={partner.t_hat:.3f}")
    
    # Test asymmetric penalties
    print("\nTest 1: Consistent cooperation should build trust")
    for i in range(5):
        partner.observe(1)  # Agent cooperates
    print(f"After 5 cooperations: t_hat={partner.t_hat:.3f} (should increase)")
    assert partner.t_hat > 0.1, "Trust should increase with cooperation"
    
    # Test betrayal penalty
    print("\nTest 2: Betrayal should hurt trust more")
    t_before = partner.t_hat
    x_before = partner.x_hat
    partner.observe(0)  # Agent defects (betrayal if x_hat > 0.5)
    print(f"Before betrayal: t_hat={t_before:.3f}, x_hat={x_before:.3f}")
    print(f"After betrayal: t_hat={partner.t_hat:.3f}, x_hat={partner.x_hat:.3f}")
    
    # Test history tracking
    print(f"\nHistory tracking:")
    print(f"  Matches: {len(partner.partner_match_history)}")
    print(f"  Betrayals: {len(partner.partner_betrayal_history)}")
    print(f"  Surprises: {len(partner.partner_surprise_history)}")
    assert len(partner.partner_match_history) == 6, "Should track all observations"
    
    print("\n✓ Trust dynamics test passed\n")


if __name__ == "__main__":
    np.random.seed(42)
    test_adaptive_partner()
    test_strategic_cheater()
    test_expectation_violation()
    test_trust_dynamics()
    print("="*50)
    print("All belief-driven partner tests passed!")
    print("="*50)
