import torch
import numpy as np
import time
import os
import warnings
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from scipy.optimize import minimize
from config.config import MIN_ETA, MAX_ETA, MIN_N, MAX_N, MIN_SIGMA_Y, MAX_SIGMA_Y
from simulation.file_ops import FileOperations

# Suppress numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The value of the smallest subnormal*")

# Setup device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

class BayesianOptimizer:
    def __init__(self, simulator, output_dir, target_displacements=None):
        """
        Initialize Bayesian Optimizer
        
        :param simulator: Simulator object
        :param output_dir: Output directory path
        :param target_displacements: List of displacement indices to optimize
        """
        self.simulator = simulator
        self.output_dir = output_dir
        self.file_ops = FileOperations()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set target displacement indices
        self.target_indices = target_displacements or [8]
        print(f"Optimizing displacements: {self.target_indices}")
        
        # Set parameter bounds
        self.bounds = torch.tensor([
            [MIN_N, MIN_ETA, MIN_SIGMA_Y],
            [MAX_N, MAX_ETA, MAX_SIGMA_Y]
        ], device=device, dtype=dtype)
        
        # Initialize data storage
        self.train_X = None
        self.train_Y = None
        self.model = None
        
    def optimize(self, sampling_number, seed=42):
        """Run Bayesian optimization process"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Starting Bayesian optimization with {sampling_number} samples")
        
        # Generate initial design points
        self._initialize_data(seed, min(5, sampling_number))
        
        # Optimization loop
        start_time = time.time()
        for i in range(len(self.train_X), sampling_number):
            print(f"\nIteration {i+1}/{sampling_number}")
            
            # Update GP model
            self._update_model()
            
            # Get next candidate using selected method
            candidate = self._get_next_candidate()
            
            # Evaluate candidate
            self._evaluate_candidate(candidate, i)
        
        # Complete optimization
        self._finalize_results(start_time)
        return self.train_X.cpu().numpy(), self.train_Y.cpu().numpy()

    def _run_simulation(self, n, eta, sigma_y):
        """Run simulation and return all displacements"""
        return self.simulator.run_simulation(n, eta, sigma_y)
    
    def _initialize_data(self, seed, num_points):
        """Initialize with Sobol sequence design"""
        self.train_X = draw_sobol_samples(
            bounds=self.bounds, 
            n=1,
            q=num_points,
            seed=seed
        ).squeeze(0).to(device)
        
        # Run simulations for initial points
        for i, params in enumerate(self.train_X):
            n, eta, sigma_y = params.cpu().numpy()
            displacements = self._run_simulation(n, eta, sigma_y)
            targets = [displacements[i-1] for i in self.target_indices]
            
            # Save data with parameters and displacements only
            self._save_iteration_data(n, eta, sigma_y, displacements)
            
            # Store targets for GP training
            if i == 0:
                self.train_Y = torch.tensor([targets], device=device, dtype=dtype)
            else:
                self.train_Y = torch.cat([self.train_Y, torch.tensor([targets], device=device, dtype=dtype)], dim=0)
    
    def _update_model(self):
        """Update GP model with current data"""
        # Normalize input parameters
        X_norm = (self.train_X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        
        # Calculate weighted objectives
        weights = torch.ones(len(self.target_indices), device=device) / len(self.target_indices)
        Y_weighted = (self.train_Y * weights).sum(dim=1)
        
        # Normalize objective values
        Y_norm = (Y_weighted - Y_weighted.mean()) / (Y_weighted.std() + 1e-8)
        
        # Create and fit GP model
        self.model = SingleTaskGP(X_norm, Y_norm.unsqueeze(-1)).to(device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
    
    def _get_next_candidate(self):
        """Get next candidate point using selected method"""
        # ===== OPTION 1: BoTorch's built-in optimizer =====
        # Calculate best observation
        weights = torch.ones(len(self.target_indices), device=device) / len(self.target_indices)
        Y_weighted = (self.train_Y * weights).sum(dim=1)
        best_value = Y_weighted.max().item()
        
        # Create EI acquisition function
        EI = ExpectedImprovement(self.model, best_f=best_value)
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds=torch.tensor([[0, 0, 0], [1, 1, 1]], device=device, dtype=dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        
        # Unnormalize candidate
        candidate = self.bounds[0] + candidates * (self.bounds[1] - self.bounds[0])
        
        # ===== OPTION 2: Custom optimization with scipy =====
        # Uncomment below and comment above to use custom method
        """
        # Define bounds for scipy
        bounds = [(MIN_N, MAX_N), (MIN_ETA, MAX_ETA), (MIN_SIGMA_Y, MAX_SIGMA_Y)]
        
        # Random starting point
        x0 = np.array([
            MIN_N + (MAX_N - MIN_N) * np.random.rand(),
            MIN_ETA + (MAX_ETA - MIN_ETA) * np.random.rand(),
            MIN_SIGMA_Y + (MAX_SIGMA_Y - MIN_SIGMA_Y) * np.random.rand()
        ])
        
        # Optimize with scipy
        result = minimize(
            self._custom_acquisition, 
            x0=x0,
            bounds=bounds, 
            method='L-BFGS-B'
        )
        candidate = torch.tensor(result.x, device=device, dtype=dtype).unsqueeze(0)
        """
        
        return candidate
    
    def _custom_acquisition(self, x):
        """Custom acquisition function for scipy optimizer"""
        x_tensor = torch.tensor(x, device=device, dtype=dtype).unsqueeze(0)
        x_norm = (x_tensor - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        
        with torch.no_grad():
            posterior = self.model.posterior(x_norm)
            mean = posterior.mean
            std = posterior.variance.sqrt()
            
            # Calculate weighted objectives
            weights = torch.ones(len(self.target_indices), device=device) / len(self.target_indices)
            Y_weighted = (self.train_Y * weights).sum(dim=1)
            best_value = Y_weighted.max()
            
            # Calculate EI
            improvement = mean - best_value
            z = improvement / (std + 1e-8)  # Avoid division by zero
            dist = torch.distributions.Normal(0, 1)
            ei = improvement * dist.cdf(z) + std * torch.exp(dist.log_prob(z))
        
        return -ei.item()
    
    def _evaluate_candidate(self, candidate, iteration):
        """Evaluate new candidate and update data"""
        params = candidate.cpu().numpy().squeeze()
        n, eta, sigma_y = params
        
        # Run simulation
        displacements = self._run_simulation(n, eta, sigma_y)
        targets = [displacements[i-1] for i in self.target_indices]
        
        # Save data with parameters and displacements only
        self._save_iteration_data(n, eta, sigma_y, displacements)
        
        # Update training data
        self.train_X = torch.cat([self.train_X, candidate], dim=0)
        self.train_Y = torch.cat([
            self.train_Y, 
            torch.tensor([targets], device=device, dtype=dtype)
        ], dim=0)
        
        # Print progress
        print(f"Parameters: n={n:.4f}, eta={eta:.4f}, σ_y={sigma_y:.4f}")
        print(f"Displacements: {dict(zip(self.target_indices, targets))}")
    
    def _finalize_results(self, start_time):
        """Finalize and print optimization results"""
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.1f} seconds")
        
        # Calculate scores and find best
        weights = torch.ones(len(self.target_indices), device=device) / len(self.target_indices)
        scores = (self.train_Y * weights).sum(dim=1)
        best_idx = torch.argmax(scores).item()
        best_params = self.train_X[best_idx].cpu().numpy()
        best_disps = self.train_Y[best_idx].cpu().numpy()
        
        # Print best results
        print(f"\nBest parameters: n={best_params[0]:.4f}, eta={best_params[1]:.4f}, σ_y={best_params[2]:.4f}")
        print(f"Displacements: {dict(zip(self.target_indices, best_disps))}")
        print(f"Score: {scores[best_idx].item():.6f}")
    
    def _save_iteration_data(self, n, eta, sigma_y, displacements):
        """Save iteration data to file (only parameters and displacements)"""
        file_path = os.path.join(self.output_dir, "optimization_results.csv")
        
        # Create headers
        headers = ["n", "eta", "sigma_y"] + [f"x_{i:02d}" for i in range(1, 9)]
        
        # Create file with headers if needed
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(",".join(headers) + "\n")
        
        # Prepare data row
        row = [n, eta, sigma_y] + list(displacements)
        
        # Append data
        with open(file_path, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")