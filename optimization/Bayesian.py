import os
import torch
import numpy as np
import taichi as ti
import joblib  # Imported joblib

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement  # (optional import; unused here)
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from botorch.models.transforms.outcome import Standardize  # <-- Added for Scheme B
from scipy.stats import qmc

from config.config import (
    XML_TEMPLATE_PATH, DEFAULT_OUTPUT_DIR, MIN_N, MAX_N, MIN_ETA, MAX_ETA,
    MIN_SIGMA_Y, MAX_SIGMA_Y, MAX_HEIGHT, MIN_HEIGHT, MAX_WIDTH, MIN_WIDTH
)

# ti.init(arch=ti.cpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)


class BayesianOptimizer:
    def __init__(self, simulator, bounds_list, output_dir, n_initial_points, n_batches, batch_size, device=None):
        """
        Bayesian Optimizer for batch optimization (Scheme B: keep Standardize(m=1) and transform best_f accordingly).
        The algorithm is the same as your original version: LHS -> 1D avg displacement objective -> qLogEI -> batched simulation.
        Only two functional changes:
          (1) Use outcome_transform=Standardize(m=1) in the 1D GP for stability.
          (2) Transform best_f into the same (standardized) space when constructing qLogEI.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Using device: {self.device}")

        self.simulator = simulator
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.n_initial_points = n_initial_points
        self.n_batches = n_batches
        self.batch_size = batch_size  # This is 'q'

        # Convert bounds to a (2, d) tensor for BoTorch
        self.bounds = torch.tensor(bounds_list, dtype=torch.float64, device=self.device).t()

        # Data storage for modeling (always in normalized space [0, 1])
        self.train_X = torch.empty((0, self.bounds.shape[1]), dtype=torch.float64, device=self.device)
        self.train_Y = torch.empty((0, 1), dtype=torch.float64, device=self.device)
        
        # Data storage for tracking original values and final results
        self.original_X = []
        self.displacements_list = []
        
        # Create results file
        self.results_file = os.path.join(output_dir, "optimization_results.csv")
        self._init_results_file()
        
    def _init_results_file(self):
        """Initialize results file with header"""
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_0{i+1}" for i in range(8)]
        with open(self.results_file, 'w') as f:
            f.write(",".join(headers) + "\n")
            
    def _save_iteration_data(self, params_numpy, displacements_numpy):
        """Save data for one iteration"""
        row = params_numpy.tolist() + displacements_numpy.tolist()
        with open(self.results_file, 'a') as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")

    def collect_initial_points(self):
        """Generate normalized initial points [0, 1]^d using LHS"""
        print(f"Generating {self.n_initial_points} initial points using Latin Hypercube Sampling...")
        sampler = qmc.LatinHypercube(d=self.bounds.shape[1])
        initial_X_scaled = sampler.random(n=self.n_initial_points)
        return torch.from_numpy(initial_X_scaled).to(dtype=torch.float64, device=self.device)
        
    def run_simulation(self, params_numpy):
        """Run simulation and return 8 displacements"""
        n, eta, sigma_y, width, height = params_numpy
        self.simulator.configure_geometry(width, height)
        displacements = self.simulator.run_simulation(n, eta, sigma_y)
        
        if displacements is None or len(displacements) == 0 or np.isnan(displacements).any():
            return np.zeros(8)
        
        if len(displacements) < 8:
            return np.concatenate([displacements, np.zeros(8 - len(displacements))])
        
        return np.array(displacements[:8])
    
    def fit_gp_model(self):
        """Fit 1D GP Model using normalized inputs and standardized output"""
        gp = SingleTaskGP(self.train_X, self.train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval(); gp.likelihood.eval()
        return gp
    
    def optimize_acquisition_function(self, gp):
        """Optimize qLogExpectedImprovement in the normalized space [0, 1]^d with correctly scaled best_f"""
        # Robustify best_f on the RAW scale first (handle NaN/Inf)
        best_Y_raw = torch.nan_to_num(self.train_Y, neginf=-1e30, posinf=1e30)

        # IMPORTANT: transform best_f into the SAME space as the model outputs
        if getattr(gp, "outcome_transform", None) is not None:
            best_Y_std = gp.outcome_transform(best_Y_raw)[0]  # (N,1) -> standardized space
            best_f = best_Y_std.max()
        else:
            best_f = best_Y_raw.max()
        
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        qEI = qLogExpectedImprovement(model=gp, best_f=best_f, sampler=qmc_sampler)
        
        standard_bounds = torch.tensor([[0.0] * self.bounds.shape[1], [1.0] * self.bounds.shape[1]], dtype=torch.float64, device=self.device)
        
        candidate, acq_value = optimize_acqf(
            acq_function=qEI,
            bounds=standard_bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200}
        )
        return candidate  # Returns a (q, d) tensor of normalized candidates
    
    def return_best_result(self):
        """Return the best parameter set and its corresponding 8 displacements (based on 1D objective)"""
        best_idx = self.train_Y.argmax()
        best_params = self.original_X[best_idx]
        best_displacements = self.displacements_list[best_idx]
        
        print("\n--- Best Result Found ---")
        print(f"Best Parameters (Original Scale): {best_params}")
        print(f"Best Displacements: {best_displacements}")
        print(f"Best Average Displacement: {self.train_Y.max().item()}")
        
        return best_params, best_displacements
        
    def optimize(self):
        """Execute the batch optimization workflow"""
        print("Starting optimization...")
        
        # 1. Generate normalized initial points
        initial_X_scaled = self.collect_initial_points()
        
        # 2. Evaluate initial points
        for x_scaled in initial_X_scaled:
            x_orig_numpy = unnormalize(x_scaled, bounds=self.bounds).cpu().numpy()
            
            displacements = self.run_simulation(x_orig_numpy)
            avg_displacement = np.mean(displacements)
            
            # Append data to all relevant lists
            self.train_X = torch.cat([self.train_X, x_scaled.unsqueeze(0)])
            self.train_Y = torch.cat([self.train_Y, torch.tensor([[avg_displacement]], dtype=torch.float64, device=self.device)])
            self.original_X.append(x_orig_numpy)
            self.displacements_list.append(displacements)
            
            self._save_iteration_data(x_orig_numpy, displacements)

        # Main optimization loop
        for i in range(self.n_batches):
            print(f"\n--- Batch {i+1}/{self.n_batches} ---")
            
            # 3. Fit 1D GP Model
            gp = self.fit_gp_model()
            try:
                print(f"Learned Lengthscale: {gp.covar_module.lengthscale.detach().cpu().numpy()}")
            except Exception:
                pass

            # # --- Save the trained GP model using joblib (optional, for inspection) ---
            # model_filename = os.path.join(self.output_dir, f'gp_model_batch_{i+1}.joblib')
            # try:
            #     joblib.dump(gp, model_filename)
            #     print(f"Saved GP model to {model_filename}")
            # except Exception as e:
            #     print(f"Warning: failed to save GP model with joblib: {e}")
            # -----------------------------------------------------------------------

            # 4. Optimize acquisition function to get a batch of new candidates
            new_X_scaled_batch = self.optimize_acquisition_function(gp)
            print(f"New Scaled Candidates:\n{new_X_scaled_batch.detach().cpu().numpy()}")
            
            # 5. Evaluate the batch of new candidates
            for x_scaled in new_X_scaled_batch:
                x_orig_numpy = unnormalize(x_scaled, bounds=self.bounds).cpu().numpy()
                displacements = self.run_simulation(x_orig_numpy)
                avg_displacement = np.mean(displacements)

                # Append new data to all relevant lists
                self.train_X = torch.cat([self.train_X, x_scaled.unsqueeze(0)])
                self.train_Y = torch.cat([self.train_Y, torch.tensor([[avg_displacement]], dtype=torch.float64, device=self.device)])
                self.original_X.append(x_orig_numpy)
                self.displacements_list.append(displacements)
                
                self._save_iteration_data(x_orig_numpy, displacements)
                print(f"  Evaluated point (original scale): n={x_orig_numpy[0]:.3f}, eta={x_orig_numpy[1]:.3f}, ... | Avg Disp: {avg_displacement:.4f}")
        
        # 6. Return the best result found
        best_params, best_displacements = self.return_best_result()
        print("\nOptimization completed!")
        return best_params, best_displacements
