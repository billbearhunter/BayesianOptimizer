import os
import math
from simulation.taichi import MPMSimulator
from optimization.Bayesian6 import BayesianOptimizer
from config.config import (
    XML_TEMPLATE_PATH,
    DEFAULT_OUTPUT_DIR,
    MIN_N,
    MAX_N,
    MIN_ETA,
    MAX_ETA,
    MIN_SIGMA_Y,
    MAX_SIGMA_Y,
    MAX_HEIGHT,
    MIN_HEIGHT,
    MAX_WIDTH,
    MIN_WIDTH,
)


def _count_existing_evals(results_csv_path: str) -> int:
    """Count how many evaluation rows already exist in the CSV (excluding header)."""
    if not os.path.exists(results_csv_path):
        return 0
    try:
        with open(results_csv_path, "r", encoding="utf-8") as f:
            # First line is header, so subtract 1
            num_lines = sum(1 for _ in f)
        return max(0, num_lines - 1)
    except Exception:
        return 0


def run_optimization(
    total_evaluations: int,
    n_initial_points: int,
    batch_size: int,
    seed: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    svgp_threshold: int = 2000,
):
    """
    Run Bayesian optimization for material parameters.

    IMPORTANT:
        - `total_evaluations` is interpreted as the GLOBAL target total number
          of simulations you want in this output_dir.

    Behavior:
        1) If no existing CSV in output_dir:
            * This is a fresh run
            * We will generate up to `n_initial_points` LHS samples (but not exceeding target)
            * The remaining (target - initial) points are collected via BO/AL batches
        2) If an existing non-empty CSV is found:
            * This is a resume run
            * We first count how many evaluations already exist (N_existing)
            * If N_existing >= total_evaluations:
                - We do nothing and return immediately
            * Else:
                - We add exactly (total_evaluations - N_existing) new points
                - No new LHS points are used (n_initial_points is ignored)
    """
    os.makedirs(output_dir, exist_ok=True)

    results_csv_path = os.path.join(output_dir, "optimization_results.csv")
    existing_evals = _count_existing_evals(results_csv_path)
    resume = existing_evals > 0

    target_total = int(total_evaluations)
    print(f"[run_optimization] Target total evaluations: {target_total}")
    print(f"[run_optimization] Existing evaluations in '{output_dir}': {existing_evals}")

    if existing_evals >= target_total:
        print("[run_optimization] Existing evaluations already >= target total.")
        print("[run_optimization] Nothing to do, exiting.")
        return None, None

    # Remaining evaluations we still want to add (global view)
    remaining_global = target_total - existing_evals

    if resume:
        # Resume: all existing points are already in the CSV
        effective_init_points = 0
        remaining_after_init = remaining_global
        print("[run_optimization] RESUME mode detected automatically.")
        print(f"[run_optimization] Will add {remaining_after_init} new evaluations.")
    else:
        # Fresh run: we can use LHS for the first n_initial_points, but not exceed remaining_global
        effective_init_points = min(n_initial_points, remaining_global)
        remaining_after_init = remaining_global - effective_init_points
        print("[run_optimization] FRESH run (no existing data found).")
        print(f"[run_optimization] Will use {effective_init_points} initial LHS samples.")

    # Number of BO/AL batches needed to cover the remaining points after LHS
    if remaining_after_init <= 0:
        n_batches = 0
    else:
        n_batches = math.ceil(remaining_after_init / batch_size)

    print(f"[run_optimization] Batch size (q): {batch_size}")
    print(f"[run_optimization] Number of BO/AL batches to run: {n_batches}")
    print(f"[run_optimization] SVGP threshold: {svgp_threshold}")
    print(f"[run_optimization] Output directory: {output_dir}")

    simulator = MPMSimulator(XML_TEMPLATE_PATH)

    bounds_list = [
        (MIN_N,       MAX_N),
        (MIN_ETA,     MAX_ETA),
        (MIN_SIGMA_Y, MAX_SIGMA_Y),
        (MIN_WIDTH,   MAX_WIDTH),
        (MIN_HEIGHT,  MAX_HEIGHT),
    ]

    try:
        optimizer = BayesianOptimizer(
            simulator=simulator,
            bounds_list=bounds_list,
            output_dir=output_dir,
            n_initial_points=effective_init_points,
            n_batches=n_batches,
            batch_size=batch_size,
            svgp_threshold=svgp_threshold,
            resume=resume,          # detect if we should load CSV
            target_total=target_total,  # global target for batch-wise resume
            test_csv_path="validation_set.csv"
        )

        best_params, best_value = optimizer.optimize()
        return best_params, best_value

    finally:
        simulator.cleanup()
        print("Simulation resources cleaned up")
