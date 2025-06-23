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
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from scipy.optimize import minimize
from config.config import MIN_ETA, MAX_ETA, MIN_N, MAX_N, MIN_SIGMA_Y, MAX_SIGMA_Y
from simulation.file_ops import FileOperations

warnings.filterwarnings("ignore", category=UserWarning, message="The value of the smallest subnormal*")

# Setup device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

class BayesianOptimizer:
    def __init__(self, simulator, output_dir, target_displacements=None):
        """
        Initialize Bayesian Optimizer with BoTorch
        
        :param simulator: Simulator object
        :param output_dir: Output directory path
        :param target_displacements: List of displacement indices to optimize
        """
        self.simulator = simulator
        self.output_dir = output_dir
        self.file_ops = FileOperations()
        
        # Ensure output directory exists
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
        self.iteration_data = []
        
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
        train_Y_list = []
        for i, params in enumerate(self.train_X):
            n, eta, sigma_y = params.cpu().numpy()
            displacements = self._run_simulation(n, eta, sigma_y)
            targets = [displacements[i-1] for i in self.target_indices]
            self._save_iteration_data(i, n, eta, sigma_y, displacements)
            train_Y_list.append(targets)
        
        self.train_Y = torch.tensor(train_Y_list, device=device, dtype=dtype)
    
    def _update_model(self):
        """Update GP model with current data"""
        X_norm = self._normalize(self.train_X)
        Y_weighted = self._calculate_weighted_Y(self.train_Y)
        Y_norm = self._normalize_Y(Y_weighted)
        
        # Create and fit GP model
        self.model = SingleTaskGP(X_norm, Y_norm.unsqueeze(-1)).to(device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
    
    def _get_next_candidate(self):
        """Get next candidate point using selected method"""
        # --- OPTION 1: BoTorch's built-in optimizer (recommended) ---
        # Calculate best observation
        Y_weighted = self._calculate_weighted_Y(self.train_Y)
        best_value = self._normalize_Y(Y_weighted).max().item()
        
        # Create EI acquisition function
        EI = ExpectedImprovement(self.model, best_f=best_value)
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds=torch.tensor([[0]*3, [1]*3], device=device, dtype=dtype),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        
        # --- OPTION 2: Custom optimization with scipy ---
        # Uncomment below and comment above to use custom method
        """
        # Define bounds for scipy
        bounds = [(MIN_N, MAX_N), (MIN_ETA, MAX_ETA), (MIN_SIGMA_Y, MAX_SIGMA_Y)]
        
        # Random starting point
        x0 = np.array([
            np.random.uniform(MIN_N, MAX_N), 
            np.random.uniform(MIN_ETA, MAX_ETA), 
            np.random.uniform(MIN_SIGMA_Y, MAX_SIGMA_Y)
        ])
        
        # Optimize with scipy
        result = minimize(
            self._custom_acquisition, 
            x0=x0,
            bounds=bounds, 
            method='L-BFGS-B'
        )
        candidates = torch.tensor(result.x, device=device, dtype=dtype).unsqueeze(0)
        """
        
        return self._unnormalize(candidates)
    
    def _custom_acquisition(self, x):
        """Custom acquisition function for scipy optimizer"""
        x_tensor = torch.tensor(x, device=device, dtype=dtype).unsqueeze(0)
        x_norm = self._normalize(x_tensor)
        
        with torch.no_grad():
            posterior = self.model.posterior(x_norm)
            mean = posterior.mean
            std = posterior.variance.sqrt()
            
            # Calculate EI
            Y_weighted = self._calculate_weighted_Y(self.train_Y)
            best_value = Y_weighted.max()
            improvement = mean - best_value
            z = improvement / std
            
            ei = improvement * torch.distributions.Normal(0, 1).cdf(z) + \
                 std * torch.exp(torch.distributions.Normal(0, 1).log_prob(z))
        
        return -ei.item()
    
    def _evaluate_candidate(self, candidate, iteration):
        """Evaluate new candidate and update data"""
        n, eta, sigma_y = candidate.cpu().numpy().squeeze()
        
        # Run simulation
        start_time = time.time()
        displacements = self._run_simulation(n, eta, sigma_y)
        targets = [displacements[i-1] for i in self.target_indices]
        sim_time = time.time() - start_time
        
        # Save results
        self._save_iteration_data(iteration, n, eta, sigma_y, displacements, sim_time)
        
        # Update training data
        self.train_X = torch.cat([self.train_X, candidate], dim=0)
        self.train_Y = torch.cat([
            self.train_Y, 
            torch.tensor([targets], device=device, dtype=dtype)
        ], dim=0)
        
        # Print progress
        print(f"Parameters: n={n:.4f}, eta={eta:.4f}, σ_y={sigma_y:.4f}")
        print(f"Displacements: {dict(zip(self.target_indices, targets))}")
        print(f"Simulation time: {sim_time:.1f}s")
    
    def _finalize_results(self, start_time):
        """Finalize and print optimization results"""
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.1f} seconds")
        
        # Calculate scores and find best
        scores = self._calculate_weighted_Y(self.train_Y)
        best_idx = torch.argmax(scores).item()
        best_params = self.train_X[best_idx].cpu().numpy()
        best_disps = self.train_Y[best_idx].cpu().numpy()
        
        # Print best results
        print(f"\nBest parameters: n={best_params[0]:.4f}, eta={best_params[1]:.4f}, σ_y={best_params[2]:.4f}")
        print(f"Displacements: {dict(zip(self.target_indices, best_disps))}")
        print(f"Score: {scores[best_idx].item():.6f}")
        
        # Save final results
        self._save_final_results()
    
    def _normalize(self, X):
        """Normalize parameters to [0, 1] range"""
        return (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
    
    def _unnormalize(self, X):
        """Unnormalize parameters to original scale"""
        return self.bounds[0] + X * (self.bounds[1] - self.bounds[0])
    
    def _normalize_Y(self, Y):
        """Standardize objectives to zero mean, unit variance"""
        return (Y - Y.mean()) / Y.std()
    
    def _calculate_weighted_Y(self, Y):
        """Calculate weighted sum of objectives"""
        weights = torch.ones(len(self.target_indices), device=device) / len(self.target_indices)
        return (Y * weights).sum(dim=1)
    
    def _save_iteration_data(self, iter, n, eta, sigma_y, displacements, sim_time=None):
        """Save iteration data to file"""
        # Prepare data
        targets = [displacements[i-1] for i in self.target_indices]
        data = {
            'n': n,
            'eta': eta,
            'sigma_y': sigma_y,
            'displacements': displacements,
        }
        self.iteration_data.append(data)
        
        # Write to CSV
        file_path = os.path.join(self.output_dir, "optimization_results.csv")
        headers = ["n", "eta", "sigma_y"] + \
                 [f"x_{i:02d}" for i in range(1, 9)]
        
        # Create file with headers if needed
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(",".join(headers) + "\n")
        
        # Prepare data row
        row = [n, eta, sigma_y] + displacements
        
        # Append data
        with open(file_path, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")
    
    def _save_final_results(self):
        """Save final optimization results"""
        results = {
            'parameters': self.train_X.cpu().numpy(),
            'objectives': self.train_Y.cpu().numpy(),
            'target_indices': self.target_indices,
            'iteration_data': self.iteration_data
        }
        # In practice, save to file using pickle or similar
        # Example: torch.save(results, os.path.join(self.output_dir, "final_results.pt"))