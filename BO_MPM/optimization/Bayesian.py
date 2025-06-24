import os
import torch
import numpy as np
import taichi as ti
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from scipy.stats import qmc

# ti.init(arch=ti.cpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

class BayesianOptimizer:
    def __init__(self, simulator, bounds, output_dir, max_iter):
        """
        Bayesian Optimizer strictly following the flowchart
        
        Args:
            simulator: MPM simulator instance
            bounds: Parameter bounds as [(min_n, max_n), (min_eta, max_eta), (min_sigma_y, max_sigma_y)]
            output_dir: Output directory for results
            max_iter: Maximum optimization iterations
        """
        self.simulator = simulator
        self.bounds = bounds
        self.bounds_tensor = torch.tensor([
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        ], dtype=torch.float32)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.max_iter = max_iter
        
        # Initialize variables
        self.X = []  # Parameters: [n, eta, sigma_y]
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
        
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
        
        gp = SingleTaskGP(X_tensor, Y_tensor)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp
    
    def optimize_acquisition_function(self, gp):
        """Optimize Acquisition Function"""
        # Get best average displacement
        best_Y = max([np.mean(d) for d in self.displacements])
        EI = ExpectedImprovement(gp, best_Y)
        
        candidate, _ = optimize_acqf(
            EI, 
            bounds=self.bounds_tensor,
            q=1, 
            num_restarts=5,
            raw_samples=128,
            options={"batch_limit": 5}  # Optional: limit batch size for each restart
        )
        return candidate[0].cpu().numpy()
    
    def return_best_result(self):
        """Return Best Result"""
        # Find parameters with highest average displacement
        avg_displacements = [np.mean(d) for d in self.displacements]
        best_idx = np.argmax(avg_displacements)
        return self.X[best_idx], self.displacements[best_idx]
        
    def optimize(self):
        """Execute the optimization workflow"""
        print("Starting optimization...")
        
        # 1. Collect initial points (5 points)
        initial_points = self.collect_initial_points()
        
        # 2. Run simulations for initial points
        for params in initial_points:
            displacements = self.run_simulation(params)
            self.X.append(params)
            self.displacements.append(displacements)
            self._save_iteration_data(params, displacements)
            
        # Optimization loop
        for i in range(self.max_iter):
            print(f"Iteration {i+1}/{self.max_iter}")
            
            # 3. Fit GP Model
            gp = self.fit_gp_model()
            
            # 4. Optimize Acquisition Function
            new_params = self.optimize_acquisition_function(gp)
            
            # 5. Run simulation for new point
            displacements = self.run_simulation(new_params)
            self.X.append(new_params)
            self.displacements.append(displacements)
            self._save_iteration_data(new_params, displacements)
            
            print(f"New point: n={new_params[0]:.3f}, eta={new_params[1]:.3f}, sigma_y={new_params[2]:.3f}, width={new_params[3]:.3f}, height={new_params[4]:.3f}")
        
        # 6. Return best result
        best_params, best_displacements = self.return_best_result()
        print("Optimization completed!")
        return best_params, best_displacements