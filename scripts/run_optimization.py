import os
import json
from simulation.taichi import MPMSimulator
from optimization.Bayesian import BayesianOptimizer
from config.config import XML_TEMPLATE_PATH, DEFAULT_OUTPUT_DIR, MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,MAX_HEIGHT, MIN_HEIGHT, MAX_WIDTH, MIN_WIDTH

def run_optimization(sampling_number, seed, output_dir=DEFAULT_OUTPUT_DIR, max_displacement=10.0):
    """
    Run Bayesian optimization for material parameters
    
    Args:
        width: Simulation domain width
        height: Simulation domain height
        sampling_number: Number of initial samples
        seed: Random seed for reproducibility
        output_dir: Output directory for results
        max_displacement: Maximum allowed displacement constraint
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulator
    simulator = MPMSimulator(XML_TEMPLATE_PATH)
    # simulator.configure_geometry(width, height)
    
    # Define parameter bounds
    bounds = [
        (MIN_N, MAX_N),
        (MIN_ETA, MAX_ETA),
        (MIN_SIGMA_Y, MAX_SIGMA_Y),
        (MIN_WIDTH, MAX_WIDTH),   
        (MIN_HEIGHT, MAX_HEIGHT),
    ]
    
    try:
        # Create optimizer
        optimizer = BayesianOptimizer(
            simulator=simulator,
            bounds=bounds,
            output_dir=output_dir,
            max_iter=sampling_number, 
        )
        
        # Run optimization
        best_params, best_value = optimizer.optimize()
        
        # # Save results
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
        # Ensure cleanup
        simulator.cleanup()
        print("Simulation resources cleaned up")