"""
Sensitivity analysis module for parameter sweeps.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
from typing import Callable, Optional, Tuple
import logging

from ..config import (
    SENSITIVITY_SEEDS, ETA_GRID, MEMORY_DISCOUNT_GRID, TRUST_DISCOUNT_GRID,
    TRUST_SMOOTHING_GRID, LOSS_AVERSION_GRID, LAMBDA_SURPRISE_GRID,
    NUM_ROUNDS, N_JOBS, VERBOSE, get_dated_subdir, RESULTS_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_param_simulation(
    eta: float,
    memory_discount: float,
    trust_discount: float,
    trust_smoothing: float,
    loss_aversion: float,
    lambda_surprise: float,
    seed: int,
    partner_factory: Callable,
    num_rounds: int,
    threshold_direction: Optional[str] = None
) -> dict:
    """
    Run a single parameter set simulation.
    
    This function is called by the parallel sweep - imports are done inside
    to avoid pickling issues with multiprocessing.
    """
    from ..simulation.runner import run_agent_simulation
    from ..agents import FocalAgent
    from ..analysis.metrics import (
        agent_coop_rate, time_to_threshold, final_decision
    )
    
    # Create agent and partner
    agent = FocalAgent(
        loss_aversion=loss_aversion,
        mu=lambda_surprise,
        eta_x=eta,
        eta_t=eta,
    )
    partner = partner_factory()
    
    # Run simulation
    df = run_agent_simulation(
        agent=agent,
        partner=partner,
        num_rounds=num_rounds,
        seed=seed,
    )
    
    return {
        "eta": eta,
        "memory_discount": memory_discount,
        "trust_discount": trust_discount,
        "trust_smoothing": trust_smoothing,
        "loss_aversion": loss_aversion,
        "lambda_surprise": lambda_surprise,
        "seed": seed,
        "agent_coop_rate": agent_coop_rate(df),
        "E_p_last": float(df["E_p"].iloc[-1]),
        "final_decision": final_decision(df),
        "time_to_threshold": time_to_threshold(df, direction=threshold_direction)
    }


def sweep_learning_params(
    partner_factory: Callable,
    eta_grid: np.ndarray = ETA_GRID,
    memory_discount_grid: np.ndarray = MEMORY_DISCOUNT_GRID,
    trust_discount_grid: np.ndarray = TRUST_DISCOUNT_GRID,
    trust_smoothing_grid: np.ndarray = TRUST_SMOOTHING_GRID,
    loss_aversion_grid: np.ndarray = LOSS_AVERSION_GRID,
    lambda_surprise_grid: np.ndarray = LAMBDA_SURPRISE_GRID,
    seeds: Tuple[int, ...] = SENSITIVITY_SEEDS,
    num_rounds: int = NUM_ROUNDS,
    threshold_direction: Optional[str] = None,
    n_jobs: int = N_JOBS,
    verbose: int = VERBOSE
) -> pd.DataFrame:
    """
    Parallelized parameter sweep with multiple seeds for robustness.
    
    Args:
        partner_factory: Callable that returns a partner instance
        *_grid: Parameter grids to sweep over
        seeds: Tuple of random seeds for robustness
        num_rounds: Rounds per simulation
        threshold_direction: 'up', 'down', or None for threshold crossing
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Verbosity level for joblib
        
    Returns:
        DataFrame with results for all parameter combinations
    """
    param_combinations = list(product(
        eta_grid, memory_discount_grid, trust_discount_grid, trust_smoothing_grid,
        loss_aversion_grid, lambda_surprise_grid, seeds
    ))
    
    logger.info("Starting parameter sweep: %d combinations", len(param_combinations))
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_param_simulation)(
            eta, md, td, ts, la, ls, seed,
            partner_factory, num_rounds, threshold_direction
        ) for eta, md, td, ts, la, ls, seed in param_combinations
    )
    
    logger.info("Parameter sweep complete")
    return pd.DataFrame(results)


class SensitivityAnalysisManager:
    """
    Manages sensitivity analysis experiments with automatic result storage.
    """
    def __init__(self, results_base_dir: Optional[Path] = None):
        self.results_dir = results_base_dir or (RESULTS_DIR / "sensitivity")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_analysis(
        self,
        partner_name: str,
        partner_factory: Callable,
        threshold_direction: Optional[str] = None,
        overwrite: bool = True,
        **sweep_kwargs
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis for a partner strategy.
        
        Args:
            partner_name: Name for result file
            partner_factory: Callable returning partner instance
            threshold_direction: Threshold crossing direction
            overwrite: If True, always run new analysis. If False, load existing.
            **sweep_kwargs: Additional arguments for sweep_learning_params
            
        Returns:
            DataFrame with sensitivity results
        """
        # Generate filename
        dated_dir = get_dated_subdir(self.results_dir)
        filename = dated_dir / f"{partner_name.replace(' ', '_')}.csv"
        
        # Check if we should load existing results
        if filename.exists() and not overwrite:
            logger.info("Loading existing results from %s", filename)
            return pd.read_csv(filename)
        
        # Run new analysis
        logger.info("Running sensitivity analysis for %s", partner_name)
        results = sweep_learning_params(
            partner_factory=partner_factory,
            threshold_direction=threshold_direction,
            **sweep_kwargs
        )
        
        # Save results
        results.to_csv(filename, index=False)
        logger.info("Results saved to %s", filename)
        
        return results
    
    def run_multiple(
        self,
        partner_configs: list,
        overwrite: bool = True,
        **sweep_kwargs
    ) -> dict:
        """
        Run sensitivity analysis for multiple partners.
        
        Args:
            partner_configs: List of (name, factory, threshold_direction) tuples
            overwrite: Whether to overwrite existing results
            **sweep_kwargs: Additional sweep parameters
            
        Returns:
            Dictionary mapping partner names to result DataFrames
        """
        results = {}
        for name, factory, direction in partner_configs:
            results[name] = self.run_analysis(
                partner_name=name,
                partner_factory=factory,
                threshold_direction=direction,
                overwrite=overwrite,
                **sweep_kwargs
            )
        return results
