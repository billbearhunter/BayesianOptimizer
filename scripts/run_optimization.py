import os
import json
from simulation.taichi import MPMSimulator
from optimization.Bayesian import BayesianOptimizer
from config.config import XML_TEMPLATE_PATH, DEFAULT_OUTPUT_DIR, MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,MAX_HEIGHT, MIN_HEIGHT, MAX_WIDTH, MIN_WIDTH


def run_optimization(total_evaluations, n_initial_points, batch_size, seed, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Run Bayesian optimization for material parameters.
    
    Args:
        total_evaluations: Total number of simulations to run.
        n_initial_points: Number of initial points for LHS.
        batch_size: Number of points per batch (q).
        seed: Random seed for reproducibility.
        output_dir: Output directory for results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the number of batches
    # Ensure there's at least one optimization batch
    n_batches = max(1, (total_evaluations - n_initial_points) // batch_size)
    print(f"Total Evaluations: {total_evaluations}")
    print(f"Initial Points: {n_initial_points}")
    print(f"Batch Size (q): {batch_size}")
    print(f"Number of Batches: {n_batches}")

    simulator = MPMSimulator(XML_TEMPLATE_PATH)
    
    bounds_list = [
        (MIN_N, MAX_N),
        (MIN_ETA, MAX_ETA),
        (MIN_SIGMA_Y, MAX_SIGMA_Y),
        (MIN_WIDTH, MAX_WIDTH),   
        (MIN_HEIGHT, MAX_HEIGHT),
    ]
    
    try:
        optimizer = BayesianOptimizer(
            simulator=simulator,
            bounds_list=bounds_list,
            output_dir=output_dir,
            n_initial_points=n_initial_points,
            n_batches=n_batches,
            batch_size=batch_size
        )
        
        best_params, best_value = optimizer.optimize()

        # Save results
        # results = {
        #     "width": width,
        #     "height": height,
        #     "best_params": best_params,
        #     "best_value": best_value,
        #     "seed": seed
        # }
        
        # result_file = os.path.join(output_dir, f"results_w{width}_h{height}.json")
        # with open(result_file, "w") as f:
        #     json.dump(results, f, indent=2)
        # print(f"Saved results to {result_file}")

        return best_params, best_value
        
    finally:
        simulator.cleanup()
        print("Simulation resources cleaned up")