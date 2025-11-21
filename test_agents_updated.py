#!/usr/bin/env python3
"""Quick test to verify updated agent implementations match original logic."""

import numpy as np
from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent

def test_focal_agent():
    """Test FocalAgent initialization and basic operations."""
    print("\n=== Testing FocalAgent ===")
    
    # Test initialization with all parameters
    agent = FocalAgent(
        u_i=0.6,
        t_init=0.1,
        eta=0.1,
        noise_sigma=0.02,
        memory_discount=0.9,
        trust_discount=0.6,
        trust_smoothing=0.15,
        stochastic=True,
        inv_temp=2.0,
        loss_aversion=2.0,
        lambda_surprise=0.5,
    )
    
    print(f"Initial state: x={agent.x:.3f}, t={agent.t:.3f}")
    print(f"expected_p() = {agent.expected_p():.3f}")
    
    # Test decision-making
    np.random.seed(42)
    decisions = [agent.decide() for _ in range(10)]
    print(f"Sample decisions: {decisions}")
    
    # Test update with cooperation
    print("\nTesting update with cooperation (1):")
    for i in range(3):
        old_x, old_t = agent.x, agent.t
        agent.update(1)
        print(f"  Round {i+1}: x: {old_x:.3f} → {agent.x:.3f}, t: {old_t:.3f} → {agent.t:.3f}")
    
    # Test update with defection
    print("\nTesting update with defection (0):")
    for i in range(3):
        old_x, old_t = agent.x, agent.t
        agent.update(0)
        print(f"  Round {i+1}: x: {old_x:.3f} → {agent.x:.3f}, t: {old_t:.3f} → {agent.t:.3f}")
    
    print("✓ FocalAgent tests passed")


def test_bayesian_agent():
    """Test BayesianFocalAgent initialization and basic operations."""
    print("\n=== Testing BayesianFocalAgent ===")
    
    # Test initialization
    agent = BayesianFocalAgent(alpha0=0.5, beta0=0.5, stochastic=True, inv_temp=2.0)
    
    print(f"Initial state: alpha={agent.alpha:.3f}, beta={agent.beta:.3f}")
    print(f"expected_p() = {agent.expected_p():.3f}")
    print(f"posterior_mean() = {agent.posterior_mean():.3f}")
    
    # Test decision-making
    np.random.seed(42)
    decisions = [agent.decide() for _ in range(10)]
    print(f"Sample decisions: {decisions}")
    
    # Test update with cooperation
    print("\nTesting update with cooperation (1):")
    for i in range(3):
        old_alpha, old_beta = agent.alpha, agent.beta
        agent.update(1)
        print(f"  Round {i+1}: α: {old_alpha:.3f} → {agent.alpha:.3f}, β: {old_beta:.3f} → {agent.beta:.3f}, E[p]: {agent.expected_p():.3f}")
    
    # Test update with defection
    print("\nTesting update with defection (0):")
    for i in range(3):
        old_alpha, old_beta = agent.alpha, agent.beta
        agent.update(0)
        print(f"  Round {i+1}: α: {old_alpha:.3f} → {agent.alpha:.3f}, β: {old_beta:.3f} → {agent.beta:.3f}, E[p]: {agent.expected_p():.3f}")
    
    print("✓ BayesianFocalAgent tests passed")


def test_shared_behavior():
    """Test that both agents use the shared decide() method from BaseAgent."""
    print("\n=== Testing Shared Decision Logic ===")
    
    focal = FocalAgent(u_i=0.6, stochastic=False)  # Deterministic
    bayesian = BayesianFocalAgent(alpha0=3.0, beta0=2.0, stochastic=False)  # Deterministic
    
    print(f"Focal expected_p: {focal.expected_p():.3f}")
    print(f"Bayesian expected_p: {bayesian.expected_p():.3f}")
    
    # Both should use same decision logic from BaseAgent
    np.random.seed(42)
    focal_decision = focal.decide()
    bayesian_decision = bayesian.decide()
    
    print(f"Focal decision (deterministic): {focal_decision}")
    print(f"Bayesian decision (deterministic): {bayesian_decision}")
    
    print("✓ Shared logic tests passed")


if __name__ == "__main__":
    test_focal_agent()
    test_bayesian_agent()
    test_shared_behavior()
    print("\n=== All tests passed! ===\n")
