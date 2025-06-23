import os
from simulation.taichi import MPMSimulator
from optimization.Bayesian import BayesianOptimizer
from config.config import XML_TEMPLATE_PATH, DEFAULT_OUTPUT_DIR

def run_optimization(width, height, sampling_number, seed, output_dir=DEFAULT_OUTPUT_DIR):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulator
    simulator = MPMSimulator(XML_TEMPLATE_PATH)
    simulator.configure_geometry(width, height)
    
    try:
        # Create optimizer
        optimizer = BayesianOptimizer(simulator, output_dir)
        
        # Run optimization
        print(f"Starting optimization with width={width}, height={height}")
        optimizer.optimize(sampling_number, seed)
        
    finally:
        # Ensure cleanup
        simulator.cleanup()
        print("Simulation resources cleaned up")