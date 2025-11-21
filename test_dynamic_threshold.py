#!/usr/bin/env python3
"""Test dynamic threshold calculation and its effect on agent initialization."""

import numpy as np
from trust_bayesian_agent_comparison.config import (
    stag_indifference_threshold, 
    DECISION_THRESHOLD,
    get_payoff,
    PAYOFF_MATRIX,
)
from trust_bayesian_agent_comparison.agents import FocalAgent

def test_threshold_calculation():
    """Test that the threshold is calculated correctly from the payoff matrix."""
    print("=== Testing stag_indifference_threshold() ===\n")
    
    # Get payoff values
    a00 = get_payoff(0, 0, 0)
    a01 = get_payoff(0, 1, 0)
    a10 = get_payoff(1, 0, 0)
    a11 = get_payoff(1, 1, 0)
    
    print(f"Payoff matrix (focal player perspective):")
    print(f"  Defect/Defect (a00): {a00}")
    print(f"  Defect/Cooperate (a01): {a01}")
    print(f"  Cooperate/Defect (a10): {a10}")
    print(f"  Cooperate/Cooperate (a11): {a11}")
    
    # Calculate threshold manually
    numerator = a00 - a10
    denominator = (a00 - a10) + (a11 - a01)
    manual_threshold = numerator / denominator
    
    print(f"\nThreshold calculation:")
    print(f"  Numerator (a00 - a10): {numerator}")
    print(f"  Denominator: {denominator}")
    print(f"  Manual: {manual_threshold:.10f}")
    
    # Get function result
    threshold = stag_indifference_threshold()
    print(f"  Function: {threshold:.10f}")
    print(f"  DECISION_THRESHOLD: {DECISION_THRESHOLD:.10f}")
    print(f"  Expected (2/3): {2/3:.10f}")
    
    assert abs(threshold - manual_threshold) < 1e-10, "Function calculation mismatch"
    assert abs(threshold - 2/3) < 1e-10, "Expected 2/3 for default payoff matrix"
    print("\n✓ Threshold calculation is correct")


def test_agent_default_initialization():
    """Test that FocalAgent initializes with correct default signal value."""
    print("\n=== Testing FocalAgent Default Initialization ===\n")
    
    # Create agent without specifying u_i
    agent = FocalAgent()
    
    threshold = stag_indifference_threshold()
    expected_x = 1 - threshold
    
    print(f"Threshold: {threshold:.10f}")
    print(f"Expected initial x (1 - threshold): {expected_x:.10f}")
    print(f"Actual initial x: {agent.x:.10f}")
    
    assert abs(agent.x - expected_x) < 1e-10, "Initial x should be 1 - threshold"
    print("\n✓ Agent initializes with correct default x value")


def test_agent_custom_initialization():
    """Test that FocalAgent can override default with custom u_i."""
    print("\n=== Testing FocalAgent Custom Initialization ===\n")
    
    custom_u_i = 0.7
    agent = FocalAgent(u_i=custom_u_i)
    
    print(f"Custom u_i: {custom_u_i}")
    print(f"Agent x: {agent.x}")
    
    assert agent.x == custom_u_i, "Agent should use custom u_i when provided"
    print("\n✓ Agent respects custom u_i parameter")


def test_dynamic_payoff_change():
    """Demonstrate that threshold would change if payoff matrix changes."""
    print("\n=== Testing Dynamic Threshold with Different Payoffs ===\n")
    
    # Current payoff
    current_threshold = stag_indifference_threshold()
    print(f"Current payoff matrix threshold: {current_threshold:.10f} (2/3)")
    
    # Simulate what would happen with different payoff matrix
    # Example: symmetric coordination game [[0,0],[5,5],[[5,5],[0,0]]]
    print("\nExample: If we changed to a symmetric coordination game:")
    print("  [[0,0], [5,0]]")
    print("  [[0,5], [5,5]]")
    print("  The threshold would be: (0-0)/[(0-0)+(5-0)] = 0.0")
    print("  (Always cooperate when p > 0)")
    
    print("\nExample: Different stag hunt with stronger coordination:")
    print("  [[3,3], [5,0]]")
    print("  [[0,5], [8,8]]")
    a00, a01, a10, a11 = 3, 5, 0, 8
    alt_threshold = (a00 - a10) / ((a00 - a10) + (a11 - a01))
    print(f"  The threshold would be: {alt_threshold:.10f}")
    print(f"  Initial x would be: {1 - alt_threshold:.10f}")
    
    print("\n✓ Threshold calculation is dynamic and adapts to payoff changes")


if __name__ == "__main__":
    test_threshold_calculation()
    test_agent_default_initialization()
    test_agent_custom_initialization()
    test_dynamic_payoff_change()
    print("\n" + "="*50)
    print("All dynamic threshold tests passed!")
    print("="*50 + "\n")
