"""Monte Carlo simulation management with automatic result storage."""

import pandas as pd
from pathlib import Path
from typing import Callable
from joblib import Parallel, delayed

from ..config import (
    NUM_MONTE_CARLO_RUNS,
    MC_BASE_SEED,
    NUM_ROUNDS,
    RESULTS_DIR,
    N_JOBS,
)
from ..simulation import run_paired_simulation


class MonteCarloManager:
    """
    Manager for Monte Carlo simulations with automatic result storage.
    
    Handles:
    - Paired simulations (same partner for both agents)
    - Dated result folders
    - Overwrite control
    - Parallel execution
    """
    
    def __init__(self, results_dir: Path = None):
        """
        Initialize Monte Carlo manager.
        
        Args:
            results_dir: Base directory for results (default: PROJECT_ROOT/results)
        """
        if results_dir is None:
            results_dir = RESULTS_DIR
        
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_monte_carlo(
        self,
        agent1_factory: Callable,
        agent2_factory: Callable,
        partner_factory: Callable,
        partner_name: str,
        num_runs: int = NUM_MONTE_CARLO_RUNS,
        base_seed: int = MC_BASE_SEED,
        num_rounds: int = NUM_ROUNDS,
        n_jobs: int = N_JOBS,
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run Monte Carlo simulation comparing two agents.
        
        Args:
            agent1_factory: Callable that creates fresh agent1 instances
            agent2_factory: Callable that creates fresh agent2 instances
            partner_factory: Callable that creates fresh partner instances
            partner_name: Name of partner for file naming
            num_runs: Number of Monte Carlo runs
            base_seed: Base random seed (run i uses seed base_seed + i)
            num_rounds: Rounds per simulation
            n_jobs: Number of parallel jobs (-1 = all cores)
            overwrite: If False and results exist, load them instead
            
        Returns:
            Tuple of (agent1_results, agent2_results) DataFrames
        """
        # Check for existing results
        file1 = self.results_dir / f"{partner_name}_agent1.csv"
        file2 = self.results_dir / f"{partner_name}_agent2.csv"
        
        if not overwrite and file1.exists() and file2.exists():
            print(f"Loading existing results for {partner_name}...")
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            return df1, df2
        
        print(f"Running Monte Carlo simulation ({num_runs} runs)...")
        
        # Run simulations in parallel
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._single_run)(
                agent1_factory,
                agent2_factory,
                partner_factory,
                base_seed + i,
                num_rounds,
                run_id=i,
            )
            for i in range(num_runs)
        )
        
        # Separate results
        all_df1 = []
        all_df2 = []
        
        for run_id, df1, df2 in results:
            df1['run_id'] = run_id
            df2['run_id'] = run_id
            all_df1.append(df1)
            all_df2.append(df2)
        
        # Combine
        combined_df1 = pd.concat(all_df1, ignore_index=True)
        combined_df2 = pd.concat(all_df2, ignore_index=True)
        
        # Save results
        combined_df1.to_csv(file1, index=False)
        combined_df2.to_csv(file2, index=False)
        
        print(f"Results saved for {partner_name}")
        
        return combined_df1, combined_df2
    
    def _single_run(
        self,
        agent1_factory: Callable,
        agent2_factory: Callable,
        partner_factory: Callable,
        seed: int,
        num_rounds: int,
        run_id: int,
    ) -> tuple[int, pd.DataFrame, pd.DataFrame]:
        """
        Execute single Monte Carlo run with paired agents.
        
        Returns:
            Tuple of (run_id, df1, df2)
        """
        agent1 = agent1_factory()
        agent2 = agent2_factory()
        
        df1, df2 = run_paired_simulation(
            agent1, agent2, partner_factory, num_rounds, seed
        )
        
        return run_id, df1, df2
