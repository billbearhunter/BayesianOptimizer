import os
import torch
import numpy as np
import taichi as ti
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from scipy.stats import qmc
from config.config import XML_TEMPLATE_PATH, DEFAULT_OUTPUT_DIR, MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,MAX_HEIGHT, MIN_HEIGHT, MAX_WIDTH, MIN_WIDTH

# ti.init(arch=ti.cpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

def min_max_scale(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val + 1e-8)
    # return (tensor - min_val) / (max_val - min_val + 1e-8) * (max_val - min_val) + min_val

class BayesianOptimizer:
    def __init__(self, simulator, bounds, output_dir, n_initial_points, n_batches, batch_size, device=None):
        """
        Bayesian Optimizer for batch optimization.
        
        Args:
            simulator: MPM simulator instance.
            bounds: Parameter bounds.
            output_dir: Output directory.
            n_initial_points: Number of initial points for LHS.
            n_batches: Number of optimization batches to run.
            batch_size: Number of points to evaluate in each batch (q).
            device: torch.device to use.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Using device: {self.device}")

        self.simulator = simulator
        self.bounds = bounds
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # self.max_iter = max_iter

        self.n_initial_points = n_initial_points
        self.n_batches = n_batches
        self.batch_size = batch_size # This is 'q'

        # Define parameter bounds using config values
        self.bounds = [
            (MIN_N, MAX_N),        # n bounds
            (MIN_ETA, MAX_ETA),     # eta bounds
            (MIN_SIGMA_Y, MAX_SIGMA_Y),  # sigma_y bounds
            (MIN_WIDTH, MAX_WIDTH), # width bounds
            (MIN_HEIGHT, MAX_HEIGHT) # height bounds
        ]

        # Create min and max tensors for scaling
        self.param_min = torch.tensor([b[0] for b in self.bounds], dtype=torch.float64, device=self.device)
        self.param_max = torch.tensor([b[1] for b in self.bounds], dtype=torch.float64, device=self.device)
        self.param_range = self.param_max - self.param_min

        # Create scaled bounds tensor
        self.scaled_bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ], dtype=torch.float64, device=self.device)
        
        self.bounds_tensor = torch.tensor([
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        ], dtype=torch.float64)
   
        # Initialize variables
        self.X = []          # Original parameters: [n, eta, sigma_y, width, height]
        self.X_scaled = []   # Scaled parameters: same parameters in [0,1] range
        self.displacements = []  # 8 displacement values per simulation
        
        # Create results file
        self.results_file = os.path.join(output_dir, "optimization_results.csv")
        self._init_results_file()
        
    def _init_results_file(self):
        """Initialize results file with header"""
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{0}{i+1}" for i in range(8)]
        with open(self.results_file, 'w') as f:
            f.write(",".join(headers) + "\n")
            
    def _save_iteration_data(self, params, displacements):
        """Save data for one iteration"""
        n, eta, sigma_y, width, height = params
        # Ensure exactly 8 displacement values
        disp_list = list(displacements[:8]) if len(displacements) >= 8 else list(displacements) + [0.0]*(8-len(displacements))
        row = [n, eta, sigma_y, width, height] + disp_list
        with open(self.results_file, 'a') as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")

    def scale_to_unit_cube(self, tensor):
        """Scale tensor from original space to unit cube [0,1]"""
        return min_max_scale(tensor, self.param_min, self.param_max)
    
    def scale_from_unit_cube(self, tensor):
        """Scale tensor from unit cube [0,1] back to original space"""
        return tensor * self.param_range + self.param_min
        
    def collect_initial_points(self):
        """Collect Initial Points (5 points using LHS)"""
        sampler = qmc.LatinHypercube(d=5)
        sample = sampler.random(n=1)
        mins = [b[0] for b in self.bounds]
        maxs = [b[1] for b in self.bounds]
        return qmc.scale(sample, mins, maxs)
        
    def run_simulation(self, params):
        """
        Run Simulation and return 8 displacements
        
        Args:
            params: [n, eta, sigma_y]
            
        Returns:
            displacements: Array of 8 displacement values
        """
        n, eta, sigma_y, width, height = params

        self.simulator.configure_geometry(width, height)

        displacements = self.simulator.run_simulation(n, eta, sigma_y)
        
        # Handle invalid results
        if displacements is None or len(displacements) == 0 or np.isnan(displacements).any():
            return np.zeros(8)
        
        # Ensure we have at least 8 displacements
        if len(displacements) < 8:
            return np.concatenate([displacements, np.zeros(8 - len(displacements))])
        
        return np.array(displacements[:8])
    
    def fit_gp_model(self):
        """Fit GP Model"""
        # Compute average displacement as target value
        Y = [np.mean(d) for d in self.displacements]
        
        X_tensor = torch.tensor(self.X, dtype=torch.float64, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float64, device=self.device).unsqueeze(-1)
        
        gp = SingleTaskGP(X_tensor, Y_tensor)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp
    
    def optimize_acquisition_function(self, gp):
        """Optimize Acquisition Function in scaled space using qEI."""
        # Find the best observed value (incumbent)
        # Note: BoTorch maximizes, so if you want to minimize, negate your objective function values (Y)
        best_Y = gp.train_targets.max()
        
        # Use q-Expected Improvement for batch optimization
        qEI = qExpectedImprovement(gp, best_Y)
        
        # Optimize in the scaled space [0, 1]^d
        candidate, acq_value = optimize_acqf(
            acq_function=qEI, 
            bounds=self.scaled_bounds,
            q=self.batch_size, # <-- Use the batch size here
            num_restarts=10,   # Increase restarts for better exploration
            raw_samples=512,  # Increase raw samples for better exploration
            options={"batch_limit": 5, "maxiter": 200}
        )
        # candidate is now a tensor of shape (q, d)
        return candidate
    
    def return_best_result(self):
        """Return Best Result"""
        # Find parameters with highest average displacement
        avg_displacements = [np.mean(d) for d in self.displacements]
        best_idx = np.argmax(avg_displacements)
        return self.X[best_idx], self.displacements[best_idx]
        
    def optimize(self):
        """Execute the optimization workflow"""
        print("Starting optimization...")
        
        # 1. Collect initial points (5 points) in original space
        initial_points = self.collect_initial_points()
        
        # 2. Run simulations for initial points
        for params in initial_points:
            # Run simulation with original parameters
            displacements = self.run_simulation(params)
            
            # Store results
            self.X.append(params)
            self.displacements.append(displacements)
            
            # Convert to tensor and scale to unit cube
            param_tensor = torch.tensor(params, dtype=torch.float64, device=self.device)
            scaled_params = self.scale_to_unit_cube(param_tensor)
            self.X_scaled.append(scaled_params)
            
            # Save iteration data
            self._save_iteration_data(params, displacements)
            
    # Optimization loop
    def optimize(self):
        """Execute the batch optimization workflow."""
        print("Starting optimization...")
        
        # 1. Generate and evaluate initial points
        initial_points = self.collect_initial_points()
        for params in initial_points:
            displacements = self.run_simulation(params)
            self.X.append(params)
            self.displacements.append(displacements)
            
            param_tensor = torch.tensor(params, dtype=torch.float64, device=self.device)
            scaled_params = self.scale_to_unit_cube(param_tensor)
            self.X_scaled.append(scaled_params)
            self._save_iteration_data(params, displacements)

        # Optimization loop runs for n_batches
        for i in range(self.n_batches):
            print(f"Batch {i+1}/{self.n_batches}")
            
            # 3. Fit GP Model using all available data
            gp = self.fit_gp_model()
            
            # 4. Optimize Acquisition Function to get a batch of new candidates
            new_params_scaled_batch = self.optimize_acquisition_function(gp)
            
            # 5. Evaluate the batch of candidates
            # !!! This is the ideal place for parallel execution of your simulator !!!
            for new_params_scaled in new_params_scaled_batch:
                new_params_orig = self.scale_from_unit_cube(new_params_scaled)
                params_numpy = new_params_orig.cpu().numpy()
                
                displacements = self.run_simulation(params_numpy)
                
                # Store results for this point
                self.X.append(params_numpy)
                self.X_scaled.append(new_params_scaled)
                self.displacements.append(displacements)
                self._save_iteration_data(params_numpy, displacements)

                print(f"  New point: n={params_numpy[0]:.3f}, eta={params_numpy[1]:.3f}, "
                      f"sigma_y={params_numpy[2]:.3f}, width={params_numpy[3]:.3f}, "
                      f"height={params_numpy[4]:.3f}")
        
        # 6. Return best result
        best_params, best_displacements = self.return_best_result()
        print("Optimization completed!")
        return best_params, best_displacements