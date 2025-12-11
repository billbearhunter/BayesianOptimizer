import os
import gc
import math
import time
from typing import List, Sequence, Optional, Dict, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import taichi as ti
import gpytorch

from torch.utils.data import DataLoader, TensorDataset

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

# GPyTorch imports
from gpytorch.models import ApproximateGP
from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    LinearKernel,
)
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

# Sampling / LHS
from scipy.stats import qmc


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
@dataclass
class GPConfig:
    """Configuration for GP training and optimization."""
    batch_size: int = 256
    num_epochs: int = 300
    lr: float = 0.01
    lr_patience: int = 20
    lr_factor: float = 0.5
    early_stop_patience: int = 40
    min_delta: float = 1e-5
    
    # SVGP settings
    svgp_threshold: int = 1500      # switch to SVGP when N > this
    max_inducing_points: int = 512  
    min_inducing_points: int = 128
    inducing_ratio: float = 0.25
    
    # Acquisition settings
    jitter_val: float = 1e-4
    num_acq_restarts: int = 15
    raw_acq_samples: int = 768
    default_quota: int = 1
    
    # Allocation settings
    # 0.1 means 10% weight on Variance, 90% weight on Validation Error (1-R2)
    allocation_alpha: float = 0.1  


# -------------------------------------------------------------------------
# Taichi initialization
# -------------------------------------------------------------------------
try:
    ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)
    print("[Bayesian6] Taichi initialized on GPU.")
except Exception as e:
    print(f"[Bayesian6] Warning: Taichi init failed or already initialized: {e}")


# -------------------------------------------------------------------------
# Utility: Farthest Point Sampling
# -------------------------------------------------------------------------
def farthest_point_sampling(
    X: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Select m indices from X using farthest point sampling."""
    n = X.shape[0]
    if m >= n:
        return torch.arange(n, device=X.device)

    # Start from a random point to avoid deterministic bias
    idx = torch.randint(low=0, high=n, size=(1,), device=X.device).item()
    idxs = [idx]
    dists = torch.cdist(X, X[idxs], p=2).squeeze(-1)

    for _ in range(1, m):
        farthest = torch.argmax(dists).item()
        idxs.append(farthest)
        new_dists = torch.cdist(X, X[[farthest]], p=2).squeeze(-1)
        dists = torch.minimum(dists, new_dists)

    return torch.tensor(idxs, device=X.device, dtype=torch.long)


# -------------------------------------------------------------------------
# Custom Acquisition: qPosteriorStandardDeviation (no decorators)
# -------------------------------------------------------------------------
class qPosteriorStandardDeviation(MCAcquisitionFunction):
    """
    MC-based batch posterior standard deviation for active learning.

    This acquisition:
    - draws MC samples from the model posterior,
    - applies a user-provided objective to select one output,
    - computes std over MC samples,
    - sums over q to get a scalar value.
    """

    def __init__(self, model, sampler: SobolQMCNormalSampler, objective=None):
        super().__init__(model=model, sampler=sampler, objective=objective)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: tensor of shape (..., q, d) in unit space.

        Returns a scalar acquisition value per batch element (...,).
        """
        posterior = self.model.posterior(X)
        # samples: [mc_samples, ..., q, m]
        samples = self.sampler(posterior)

        if self.objective is not None:
            # User objective: samples -> scalar per output dimension
            # shape: [mc_samples, ..., q]
            obj_vals = self.objective(samples, X=X)
        else:
            # Fallback: use L2 norm over outputs
            if samples.shape[-1] == 1:
                obj_vals = samples.squeeze(-1)
            else:
                obj_vals = samples.norm(dim=-1)

        # Std over MC dimension
        std = obj_vals.std(dim=0)      # [..., q]
        return std.sum(dim=-1)         # [...]


# -------------------------------------------------------------------------
# Custom SingleOutputSVGP
# -------------------------------------------------------------------------
class SingleOutputSVGP(ApproximateGP, GPyTorchModel):
    """
    Single-output SVGP with integrated input transformation (log-standardized).

    Input pipeline:
    unit cube -> physical bounds -> log -> standardize (mean/std over log space).
    """
    _num_outputs = 1
    
    def __init__(
        self,
        inducing_points_log_std: torch.Tensor,
        bounds_list: Sequence[Sequence[float]],
        x_log_mean: torch.Tensor,
        x_log_std: torch.Tensor,
        likelihood: GaussianLikelihood,
    ):
        GPyTorchModel.__init__(self)

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points_log_std.size(-2)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points_log_std,
            variational_distribution,
            learn_inducing_locations=True,
        )

        ApproximateGP.__init__(self, variational_strategy)
        
        self.likelihood = likelihood
        
        self.register_buffer("bounds_tensor", torch.tensor(bounds_list, dtype=torch.float64).T)
        self.register_buffer("x_log_mean", x_log_mean)
        self.register_buffer("x_log_std", x_log_std)
        
        d = inducing_points_log_std.shape[-1]
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            LinearKernel(ard_num_dims=d) + MaternKernel(nu=2.5, ard_num_dims=d)
        )
    
    def _transform_inputs(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Map inputs from [0,1]^d to standardized log space."""
        lower = self.bounds_tensor[0].to(x_unit)
        upper = self.bounds_tensor[1].to(x_unit)
        x_phys = x_unit * (upper - lower) + lower
        
        # Hardcoded clamping for stability in Log-Space
        # Warning: This assumes parameters are strictly positive (common in MPM: n, eta, sigma_y)
        x_phys_clamped = torch.clamp(x_phys, min=1e-6)
        x_log = torch.log(x_phys_clamped)
        
        x_std = (x_log - self.x_log_mean) / self.x_log_std
        return x_std
    
    def __call__(self, inputs, **kwargs):
        if torch.is_tensor(inputs):
            inputs_transformed = self._transform_inputs(inputs)
            return super().__call__(inputs_transformed, **kwargs)
        return super().__call__(inputs, **kwargs)
    
    def forward(self, x_std: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x_std)
        covar_x = self.covar_module(x_std)
        return MultivariateNormal(mean_x, covar_x)


# -------------------------------------------------------------------------
# Main BayesianOptimizer
# -------------------------------------------------------------------------
class BayesianOptimizer:
    def __init__(
        self,
        simulator,
        bounds_list: Sequence[Sequence[float]],
        output_dir: str,
        n_initial_points: int,
        n_batches: int,
        batch_size: int,
        num_outputs: int = 8,  # [Fix] Added parameter to avoid hardcoding
        svgp_threshold: int = 2500,  
        resume: bool = False,
        target_total: Optional[int] = None,
        device: Optional[torch.device] = None,
        gp_config: Optional[GPConfig] = None,
        **kwargs 
    ):
        """
        simulator: an MPMSimulator-like object providing:
          - configure_geometry(width, height)
          - run_simulation(n, eta, sigma_y) -> array-like (num_outputs,)
        bounds_list: [[min_n, max_n], [min_eta, max_eta], ...] in physical space.
        """
        if device is None:
            self.gp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.gp_device = device
            
        self.config = gp_config or GPConfig()
        self.config.svgp_threshold = svgp_threshold
        
        self.dtype = torch.float64
        print(f"[BayesianOptimizer] Device: {self.gp_device} | SVGP Threshold: {self.config.svgp_threshold}")
        print(f"[BayesianOptimizer] Strategy: Dynamic Validation (Alpha={self.config.allocation_alpha})")
        
        if kwargs:
            print(f"[BayesianOptimizer] Warning: Ignoring unused arguments: {list(kwargs.keys())}")

        self.simulator = simulator
        self.physical_bounds = list(bounds_list)
        self.dim = len(self.physical_bounds)
        
        # [Fix] Check bounds for negative values warning
        bounds_arr = np.array(self.physical_bounds)
        if np.any(bounds_arr[:, 0] <= 0):
            print("\n[WARNING] Some lower bounds are <= 0. "
                  "The current Log-transform logic will clamp inputs to 1e-6. "
                  "Ensure your physical model (n, eta, sigma) is strictly positive.\n")
        
        self.n_initial_points = int(n_initial_points)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        
        # [Fix] Use the passed argument
        self.num_outputs = int(num_outputs)
        
        self.resume = resume
        self.target_total = target_total
        
        self.train_X = torch.empty((0, self.dim), dtype=self.dtype, device=self.gp_device)
        self.train_Y_raw = torch.empty((0, self.num_outputs), dtype=self.dtype, device=self.gp_device)
        
        # Output normalization in log-space
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None
        self.y_log_shift: Optional[torch.Tensor] = None

        # Input log stats (for SVGP)
        self.x_log_mean: Optional[torch.Tensor] = None
        self.x_log_std: Optional[torch.Tensor] = None
        
        self.gp_mode: str = "exact"
        self.gp: Optional[torch.nn.Module] = None
        self.models: List[SingleOutputSVGP] = []
        self.likelihoods: List[GaussianLikelihood] = []
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        # NOTE: keep your original CSV naming here if you rely on it elsewhere
        self.results_csv_path = os.path.join(output_dir, "optimization_results.csv")
        self.model_save_path = os.path.join(output_dir, "trained_botorch_model.pt")
        self.failure_log_path = os.path.join(output_dir, "simulation_failures.log")
        
        self._init_or_load_results_file()

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------
    def _init_or_load_results_file(self) -> None:
        if not os.path.exists(self.results_csv_path):
            self._create_new_csv()
            return
        
        if self.resume:
            print(f"[BayesianOptimizer] Resuming from {self.results_csv_path}")
            self._load_existing_data()
        else:
            print(f"[BayesianOptimizer] Overwriting {self.results_csv_path}")
            self._create_new_csv()
    
    def _create_new_csv(self):
        headers = ["n", "eta", "sigma_y", "width", "height"] + [
            f"x_{i:02d}" for i in range(1, self.num_outputs + 1)
        ]
        with open(self.results_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
    
    def _load_existing_data(self) -> None:
        try:
            df = pd.read_csv(self.results_csv_path)
        except Exception as e:
            print(f"[Error] Failed to read CSV: {e}")
            self._create_new_csv()
            return
        
        if df.empty:
            return
        
        x_cols = ["n", "eta", "sigma_y", "width", "height"]
        y_cols = [f"x_{i:02d}" for i in range(1, self.num_outputs + 1)]
        
        if not all(c in df.columns for c in x_cols + y_cols):
            print("[Error] CSV columns mismatch")
            self._create_new_csv()
            return
        
        X_phys = df[x_cols].to_numpy(dtype=np.float64)
        Y_raw = df[y_cols].to_numpy(dtype=np.float64)
        
        bounds = np.array(self.physical_bounds, dtype=np.float64)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        X_unit = (X_phys - lower) / (upper - lower)
        
        self.train_X = torch.from_numpy(X_unit).to(self.gp_device, dtype=self.dtype)
        self.train_Y_raw = torch.from_numpy(Y_raw).to(self.gp_device, dtype=self.dtype)
        
        print(f"[BayesianOptimizer] Loaded {self.train_X.shape[0]} samples.")
    
    def _save_iteration_data(self, x_phys: np.ndarray, disp: np.ndarray) -> None:
        with open(self.results_csv_path, "a", encoding="utf-8") as f:
            row = np.concatenate([x_phys, disp])
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")
            
    def _log_simulation_failure(self, params: np.ndarray, error_msg: str):
        with open(self.failure_log_path, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {params} -> {error_msg}\n")

    # ------------------------------------------------------------------
    # Simulator (kept exactly like your original design)
    # ------------------------------------------------------------------
    def _scaled_to_original(self, x_scaled: torch.Tensor) -> np.ndarray:
        """
        Convert a point in unit space [0,1]^d to physical parameter space
        using the provided bounds.
        """
        values = []
        for i, (lower, upper) in enumerate(self.physical_bounds):
            values.append(lower + (upper - lower) * x_scaled[i].item())
        return np.array(values, dtype=np.float64)

    def run_simulation(self, params) -> Optional[np.ndarray]:
        """
        Call the external MPMSimulator with your original interface:

        params: [n, eta, sigma_y, width, height]
        - configure_geometry(width, height)
        - run_simulation(n, eta, sigma_y)

        Returns:
            disp: np.ndarray of shape (num_outputs,) or None on failure.
        """
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            params_np = np.asarray(params, dtype=np.float64).reshape(-1)
        
        n, eta, sigma_y, width, height = params_np[:5]
        
        try:
            self.simulator.configure_geometry(float(width), float(height))
            displacements = self.simulator.run_simulation(
                float(n), float(eta), float(sigma_y)
            )
        except Exception as e:
            self._log_simulation_failure(params_np, f"Exception: {e}")
            return None
        
        if displacements is None:
            self._log_simulation_failure(params_np, "Simulator returned None")
            return None
        
        disp = np.asarray(displacements, dtype=np.float64).reshape(-1)
        
        # Pad or truncate to num_outputs
        if disp.shape[0] < self.num_outputs:
            padded = np.zeros(self.num_outputs, dtype=np.float64)
            padded[: disp.shape[0]] = disp
            disp = padded
        else:
            disp = disp[: self.num_outputs]
        
        if np.any(np.isnan(disp)) or np.any(np.isinf(disp)):
            self._log_simulation_failure(params_np, "Result contains NaN/Inf")
            return None
            
        return disp

    # ------------------------------------------------------------------
    # Stats Helpers
    # ------------------------------------------------------------------
    def _compute_safe_epsilon(self, tensor: torch.Tensor) -> float:
        """Compute a small epsilon relative to tensor scale to keep values positive."""
        if tensor.numel() == 0:
            return 1e-6
        abs_max = tensor.abs().max().item()
        return max(1e-12, abs_max * 1e-6)
    
    def _update_output_stats(self) -> None:
        """
        Update log-transform statistics for outputs:
        - global scalar shift (shared across outputs),
        - per-output mean and std in log space.
        """
        if self.train_Y_raw.shape[0] < 2:
            return

        Y = self.train_Y_raw
        y_min = float(Y.min().item())
        
        if y_min <= 0.0:
            shift_val = -y_min + self._compute_safe_epsilon(Y)
        else:
            shift_val = self._compute_safe_epsilon(Y)
            
        self.y_log_shift = torch.tensor(shift_val, dtype=self.dtype, device=self.gp_device)
        Y_log = torch.log(Y + self.y_log_shift)
        
        self.y_mean = Y_log.mean(dim=0, keepdim=True)
        
        std = Y_log.std(dim=0, keepdim=True)
        constant_mask = std < 1e-12
        if torch.any(constant_mask):
            std = torch.where(constant_mask, torch.ones_like(std) * 1e-12, std)
        self.y_std = std
    
    def _compute_log_input_stats(self) -> None:
        """
        Compute mean and std of log-physical inputs for SVGP inducing points.
        """
        X_unit = self.train_X
        bounds = torch.tensor(self.physical_bounds, dtype=self.dtype, device=self.gp_device).T
        lower, upper = bounds[0], bounds[1]
        
        X_phys = X_unit * (upper - lower) + lower
        X_phys_clamped = torch.clamp(X_phys, min=1e-6)
        X_log = torch.log(X_phys_clamped)
        
        self.x_log_mean = X_log.mean(dim=0, keepdim=True)
        self.x_log_std = X_log.std(dim=0, keepdim=True).clamp_min(1e-12)

    # ------------------------------------------------------------------
    # GP Fitting (Exact and SVGP)
    # ------------------------------------------------------------------
    def _fit_exact_gp(self) -> SingleTaskGP:
        """
        Fit an exact multi-output GP (SingleTaskGP) in log-standardized space.
        """
        X_unit = self.train_X
        Y_raw = self.train_Y_raw
        
        self._update_output_stats()
        Y_log = torch.log(Y_raw + self.y_log_shift)
        if torch.isnan(Y_log).any():
            print("[Warning] NaN in Y_log. Clamping.")
            Y_log = torch.nan_to_num(Y_log, nan=0.0)

        Y_log_std = (Y_log - self.y_mean) / self.y_std
        if torch.isnan(Y_log_std).any():
            print("[Warning] NaN in Y_log_std. Clamping.")
            Y_log_std = torch.nan_to_num(Y_log_std, nan=0.0)
        
        d = X_unit.shape[-1]
        covar_module = ScaleKernel(
            LinearKernel(ard_num_dims=d) + MaternKernel(nu=2.5, ard_num_dims=d)
        )
        
        gp = SingleTaskGP(
            train_X=X_unit,
            train_Y=Y_log_std,
            covar_module=covar_module,
        ).to(self.gp_device, self.dtype)
        
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        gp.train()
        
        try:
            with gpytorch.settings.cholesky_jitter(self.config.jitter_val):
                fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"[GP Warning] Fitting failed. Retrying with higher jitter. {e}")
            with gpytorch.settings.cholesky_jitter(1e-2):
                fit_gpytorch_mll(mll)

        gp.eval()
        return gp

    def _fit_svgp(self) -> ModelListGP:
        """
        Fit a ModelListGP of SVGPs, one per output dimension.
        Each SVGP uses log-standardized inputs and outputs.
        """
        X_unit = self.train_X
        Y_raw = self.train_Y_raw
        
        self._update_output_stats()
        Y_log = torch.log(Y_raw + self.y_log_shift)
        
        if torch.isnan(Y_log).any():
            Y_log = torch.nan_to_num(Y_log, nan=0.0)
            
        Y_log_std = (Y_log - self.y_mean) / self.y_std
        if torch.isnan(Y_log_std).any():
            Y_log_std = torch.nan_to_num(Y_log_std, nan=0.0)
        
        self._compute_log_input_stats()
        
        bounds = torch.tensor(self.physical_bounds, dtype=self.dtype, device=self.gp_device).T
        lower, upper = bounds[0], bounds[1]
        # Inducing point selection in log-standardized input space
        X_phys = X_unit * (upper - lower) + lower
        X_log = torch.log(torch.clamp(X_phys, min=1e-6))
        X_log_std = (X_log - self.x_log_mean) / self.x_log_std
        
        num_inducing = min(
            self.config.max_inducing_points, 
            max(
                self.config.min_inducing_points, 
                int(self.config.inducing_ratio * X_unit.shape[0])
            )
        )
        
        idx = farthest_point_sampling(X_log_std, num_inducing)
        inducing_points_log_std = X_log_std[idx].clone()
        
        models = []
        likelihoods = []
        
        for i in range(self.num_outputs):
            y_i = Y_log_std[:, i]
            lik_i = GaussianLikelihood().to(self.gp_device, self.dtype)
            
            model_i = SingleOutputSVGP(
                inducing_points_log_std=inducing_points_log_std,
                bounds_list=self.physical_bounds,
                x_log_mean=self.x_log_mean,
                x_log_std=self.x_log_std,
                likelihood=lik_i,
            ).to(self.gp_device, self.dtype)
            
            self._train_single_svgp(model_i, lik_i, X_unit, y_i, self.config.num_epochs)
            models.append(model_i)
            likelihoods.append(lik_i)
            
        self.models = models
        self.likelihoods = likelihoods
        
        gp = ModelListGP(*models).to(self.gp_device, self.dtype)
        gp.eval()
        return gp

    def _train_single_svgp(self, model, likelihood, X_unit, y_std, num_epochs):
        """
        Train a single-output SVGP with mini-batches and early stopping.
        """
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=self.config.lr_patience, 
            factor=self.config.lr_factor, 
        )
        
        mll = VariationalELBO(likelihood, model, num_data=X_unit.shape[0])
        dataset = TensorDataset(X_unit, y_std)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        clip_value = 5.0 
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            total_count = 0
            
            for Xb, yb in loader:
                optimizer.zero_grad()
                out = model(Xb)
                loss = -mll(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                optimizer.step()
                
                bs = Xb.shape[0]
                epoch_loss += loss.item() * bs
                total_count += bs
            if total_count == 0:
                break
                
            avg_loss = epoch_loss / total_count
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss - self.config.min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stop_patience:
                break
                
        model.eval()
        likelihood.eval()

    def fit_gp_model(self) -> torch.nn.Module:
        """
        Fit a GP model (Exact or SVGP) depending on data size.
        """
        if self.gp is not None:
            del self.gp
            self.models = []
            self.likelihoods = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        N = self.train_X.shape[0]
        if N < 2:
            raise RuntimeError("Need at least two observations.")
            
        if N <= self.config.svgp_threshold:
            self.gp_mode = "exact"
            print(f"[GP] Fitting Exact GP (N={N})")
            self.gp = self._fit_exact_gp()
        else:
            self.gp_mode = "svgp"
            print(f"[GP] Fitting SVGP (N={N})")
            self.gp = self._fit_svgp()
            
        self.save_gp_model()
        return self.gp

    # ------------------------------------------------------------------
    # Dynamic Validation Helper (with proper stats restore)
    # ------------------------------------------------------------------
    def _compute_validation_metrics(self, train_ratio=0.8):
        """
        Temporarily splits dataset to train a proxy model and compute
        validation metrics.
        
        [Fix] Explicitly checks proxy_gp type to handle shape mismatches
        between ExactGP and ModelListGP independently of self.gp_mode.
        """
        print(f"[Validation] Splitting data {train_ratio*100:.0f}/{100-train_ratio*100:.0f} for error estimation...")
        
        N = self.train_X.shape[0]
        indices = torch.randperm(N)
        split = int(N * train_ratio)
        
        train_idx = indices[:split]
        val_idx = indices[split:]
        
        # Save full data
        original_X = self.train_X
        original_Y = self.train_Y_raw

        # Backup stats
        orig_y_mean = self.y_mean.clone() if self.y_mean is not None else None
        orig_y_std = self.y_std.clone() if self.y_std is not None else None
        orig_y_log_shift = self.y_log_shift.clone() if self.y_log_shift is not None else None
        orig_x_log_mean = self.x_log_mean.clone() if self.x_log_mean is not None else None
        orig_x_log_std = self.x_log_std.clone() if self.x_log_std is not None else None
        
        X_train_sub = original_X[train_idx]
        Y_train_sub = original_Y[train_idx]
        X_val_sub = original_X[val_idx]
        Y_val_sub = original_Y[val_idx]
        
        try:
            # Train proxy model on subset
            self.train_X = X_train_sub
            self.train_Y_raw = Y_train_sub
            
            # Decide proxy type based on subset size
            if self.train_X.shape[0] <= self.config.svgp_threshold:
                proxy_gp = self._fit_exact_gp()
            else:
                proxy_gp = self._fit_svgp()
            
            proxy_gp.eval()
            with torch.no_grad():
                posterior = proxy_gp.posterior(X_val_sub)
                mean_std = posterior.mean
                var_std = posterior.variance
                
                # [Fix] Handle shapes based on specific proxy model type
                if isinstance(proxy_gp, SingleTaskGP):
                    # Exact GP output: [batch_shape, output_dim, n_points] or [output_dim, n_points]
                    if mean_std.ndim == 3:
                        mean_std = mean_std.squeeze(0)
                    if var_std.ndim == 3:
                        var_std = var_std.squeeze(0)
                
                elif isinstance(proxy_gp, ModelListGP):
                    # ModelListGP output often: [output_dim, n_points]
                    # But sometimes transposed depending on posterior call
                    if mean_std.shape[0] == self.num_outputs and mean_std.shape[-1] == X_val_sub.shape[0]:
                        mean_std = mean_std.T
                        var_std = var_std.T

                # Reconstruct predictions in original scale
                y_mean = self.y_mean.to(self.gp_device)
                y_std_val = self.y_std.to(self.gp_device)
                shift = self.y_log_shift.to(self.gp_device)
                
                log_mean = mean_std * y_std_val + y_mean
                # Log-Normal Mean recovery
                log_var = var_std * (y_std_val ** 2)
                Y_pred_val = torch.exp(log_mean + 0.5 * log_var) - shift
                
                ssr = (Y_val_sub - Y_pred_val).pow(2).sum(dim=0)
                sst = (Y_val_sub - Y_val_sub.mean(dim=0)).pow(2).sum(dim=0)
                sst = torch.where(sst < 1e-9, torch.ones_like(sst), sst)
                
                val_errors = (ssr / sst).cpu().numpy()
                
                # [Fix] Robust variance averaging
                if var_std.ndim == 2:
                    mean_var = var_std.mean(dim=0) # [n_outputs] if shape is [n_points, n_outputs]
                elif var_std.ndim == 3:
                     # e.g. [1, n_points, n_outputs] -> mean over batch and points
                    mean_var = var_std.mean(dim=(0, 1)) 
                else:
                    # Fallback
                    mean_var = var_std.mean(dim=0)
                    
                val_uncertainties = mean_var.cpu().numpy()

        finally:
            # Restore full data
            self.train_X = original_X
            self.train_Y_raw = original_Y

            # Restore original statistics
            self.y_mean = orig_y_mean
            self.y_std = orig_y_std
            self.y_log_shift = orig_y_log_shift
            self.x_log_mean = orig_x_log_mean
            self.x_log_std = orig_x_log_std

            if "proxy_gp" in locals():
                del proxy_gp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("[Validation] Done.")
        
        return val_errors, val_uncertainties

    # ------------------------------------------------------------------
    # Evaluation (Training Data)
    # ------------------------------------------------------------------
    def evaluate_model(self, dataset_name: str = "Training Data") -> pd.DataFrame:
        """
        Evaluate current GP on training data, computing R2/MSE/MAE/MaxErr per output.
        """
        if self.gp is None or self.train_X.shape[0] < 5:
            return pd.DataFrame()
            
        # Quick marginal coverage metric for exploration monitoring
        if self.train_X.shape[0] > 10:
            try:
                X_np = self.train_X.cpu().numpy()
                n_bins = 10
                dim_coverages = []
                for d in range(self.dim):
                    hist, _ = np.histogram(X_np[:, d], bins=n_bins, range=(0, 1))
                    coverage_d = np.count_nonzero(hist) / n_bins
                    dim_coverages.append(coverage_d)
                avg_coverage = np.mean(dim_coverages)
                print(f"[Exploration] Marginal Coverage (avg): {avg_coverage:.2%}")
            except Exception as e:
                print(f"[Exploration Warning] Metrics failed: {e}")

        X = self.train_X.to(self.gp_device, self.dtype)
        Y_true = self.train_Y_raw.to(self.gp_device, self.dtype)
        
        self.gp.eval()
        with torch.no_grad():
            posterior = self.gp.posterior(X)
            mean_std = posterior.mean
            var_std = posterior.variance
            
            if self.gp_mode == "exact":
                if mean_std.ndim == 3:
                    mean_std = mean_std.squeeze(0)
                if var_std.ndim == 3:
                    var_std = var_std.squeeze(0)
            else:
                if mean_std.shape[0] == self.num_outputs and mean_std.shape[-1] == X.shape[0]:
                    mean_std = mean_std.T
                    var_std = var_std.T
            
            y_mean = self.y_mean.to(self.gp_device)
            y_std_val = self.y_std.to(self.gp_device)
            
            log_mean = mean_std * y_std_val + y_mean
            log_var = var_std * (y_std_val ** 2)
            
            shift = self.y_log_shift.to(self.gp_device)
            Y_pred = torch.exp(log_mean + 0.5 * log_var) - shift
            
        y_true_np = Y_true.cpu().numpy()
        y_pred_np = Y_pred.cpu().numpy()
        
        metrics = []
        for j in range(self.num_outputs):
            yt = y_true_np[:, j]
            yp = y_pred_np[:, j]
            err = yp - yt
            
            mse = float(np.mean(err**2))
            mae = float(np.mean(np.abs(err)))
            maxerr = float(np.max(np.abs(err)))
            var = float(np.var(yt))
            
            if var == 0.0:
                r2 = 1.0 if mse < 1e-9 else 0.0
            else:
                r2 = 1.0 - (float(np.sum(err**2)) / float(np.sum((yt - yt.mean())**2)))
                
            metrics.append({
                "Output": f"x_{j+1:02d}",
                "R2": r2, "MSE": mse, "MAE": mae, "MaxErr": maxerr
            })
            
        df = pd.DataFrame(metrics)
        print(f"\n--- Model Performance on {dataset_name} (N={self.train_X.shape[0]}) ---")
        print(df.to_string(index=False, float_format="%.4f"))
        print("-" * 50)
        return df

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    def _avoid_repeated_points(
        self, 
        X_cand: torch.Tensor, 
        tol: float = 1e-3, 
        max_attempts: int = 10,
        batch_check: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Avoid sampling points too close to existing data or same-batch points
        by adding small perturbations.
        """
        if self.train_X.shape[0] == 0 and (batch_check is None or batch_check.shape[0] == 0):
            return X_cand
            
        X_new = X_cand.clone()
        for i in range(X_new.shape[0]):
            xi = X_new[i]
            
            check_tensors = []
            if self.train_X.shape[0] > 0:
                check_tensors.append(self.train_X)
            if batch_check is not None and batch_check.shape[0] > 0:
                check_tensors.append(batch_check)
            
            if not check_tensors:
                continue
                
            points_to_check = torch.cat(check_tensors, dim=0)
            dists = torch.norm(points_to_check - xi, dim=-1)
            
            if torch.any(dists < tol):
                for attempt in range(max_attempts):
                    noise_scale = min(0.1, tol * (2 ** attempt))
                    noise = (torch.rand_like(xi) - 0.5) * 2.0 * noise_scale
                    new_x = (xi + noise).clamp(0.0, 1.0)
                    
                    d_check = torch.norm(points_to_check - new_x, dim=-1)
                    if torch.all(d_check >= tol * 0.5):
                        X_new[i] = new_x
                        break
        return X_new

    def _allocate_quotas(self, scores: np.ndarray, total_q: int) -> np.ndarray:
        """
        Allocate integer quotas per output based on scores.
        Ensures at least default_quota per output.
        """
        min_per_output = max(1, self.config.default_quota)
        base = np.ones_like(scores, dtype=int) * min_per_output
        remaining = total_q - len(scores) * min_per_output
        
        if remaining < 0:
            top_indices = np.argsort(-scores)[:total_q]
            quotas = np.zeros_like(scores, dtype=int)
            quotas[top_indices] = 1
            return quotas
        
        if remaining > 0:
            weights = scores / (scores.sum() + 1e-10)
            additional = np.floor(weights * remaining).astype(int)
            quotas = base + additional
            
            remainder = total_q - quotas.sum()
            if remainder > 0:
                sorted_idx = np.argsort(-scores)[:remainder]
                quotas[sorted_idx] += 1
                
        return quotas

    def save_gp_model(self, path: Optional[str] = None) -> None:
        """
        Save GP model and normalization statistics to disk.
        """
        if path is None:
            path = self.model_save_path
        if self.gp is None:
            return
        
        state = {
            "mode": self.gp_mode,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "y_log_shift": self.y_log_shift,
            "bounds_list": self.physical_bounds,
            "num_outputs": self.num_outputs,
        }
        if self.gp_mode == "exact":
            state["state_dict"] = self.gp.state_dict()
        else:
            state["models"] = [m.state_dict() for m in self.models]
            state["likelihoods"] = [lik.state_dict() for lik in self.likelihoods]
            state["x_log_mean"] = self.x_log_mean
            state["x_log_std"] = self.x_log_std
            
        torch.save(state, path)
        print(f"[GP] Checkpoint saved to {path}")

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------
    def optimize(self):
        print("[BayesianOptimizer] Starting optimization loop...")
        
        # Initial LHS if not resuming or CSV empty
        if not self.resume or self.train_X.shape[0] == 0:
            if self.n_initial_points > 0:
                print(f"[Init] Collecting {self.n_initial_points} LHS points.")
                sampler = qmc.LatinHypercube(d=self.dim)
                X_unit_np = sampler.random(n=self.n_initial_points)
                X_unit = torch.from_numpy(X_unit_np).to(self.gp_device, self.dtype)
                
                for x_u in X_unit:
                    x_phys = self._scaled_to_original(x_u)
                    print(f"  [Init] Params: {x_phys}") 
                    disp = self.run_simulation(x_phys)
                    if disp is not None:
                        self.train_X = torch.cat([self.train_X, x_u.view(1, -1)], dim=0)
                        self.train_Y_raw = torch.cat(
                            [self.train_Y_raw,
                             torch.tensor(disp, device=self.gp_device, dtype=self.dtype).view(1, -1)], 
                            dim=0
                        )
                        self._save_iteration_data(x_phys, disp)
                        print(f"  -> Avg Disp: {np.mean(disp):.6f}")
        
        for batch_idx in range(self.n_batches):
            if self.target_total and self.train_X.shape[0] >= self.target_total:
                print("[Stop] Target total evaluations reached.")
                break
                
            print(f"\n--- Batch {batch_idx + 1}/{self.n_batches} ---")
            
            if batch_idx > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # --- Step 1: Dynamic Validation ---
            if self.train_X.shape[0] > 20: 
                val_errors, val_uncertainties = self._compute_validation_metrics(train_ratio=0.8)
                print("[Metric] Using Validation Set Error (1-R2) for allocation.")
            else:
                # For very few points, just fit once and use uniform metrics
                _ = self.fit_gp_model() 
                val_errors = np.ones(self.num_outputs) 
                val_uncertainties = np.ones(self.num_outputs)
            
            # --- Step 2: Quota allocation across outputs ---
            u_norm = val_uncertainties / (val_uncertainties.sum() + 1e-10)
            e_norm = val_errors / (val_errors.sum() + 1e-10)
            
            alpha = self.config.allocation_alpha 
            scores = alpha * u_norm + (1 - alpha) * e_norm
            
            print("[Acquisition] Allocation Stats (Validation Metrics):")
            print(f"  {'Output':<6} | {'Val Uncert':<12} | {'Val 1-R2':<12} | {'Score':<12}")
            for i in range(self.num_outputs):
                print(f"  x_{i+1:02d}   | {val_uncertainties[i]:.4e}   | {val_errors[i]:.4e}   | {scores[i]:.4f}")
            
            q_total = min(
                self.batch_size, 
                self.target_total - self.train_X.shape[0] if self.target_total else self.batch_size
            )
            quotas = self._allocate_quotas(scores, q_total)
            
            print(f"[Acquisition] Final Quotas (Total={q_total}):")
            for i, q_i in enumerate(quotas):
                if q_i > 0:
                    print(f"  - x_{i+1:02d}: quota={q_i}")
                
            # --- Step 3: Retrain GP on full dataset for acquisition ---
            print("[GP] Retraining on FULL dataset for acquisition...")
            gp = self.fit_gp_model()
            
            # Diagnostics (optional)
            self.evaluate_model()

            bounds = torch.stack([
                torch.zeros(self.dim, device=self.gp_device, dtype=self.dtype),
                torch.ones(self.dim, device=self.gp_device, dtype=self.dtype)
            ])
            
            X_new_list = []
            current_batch_points = torch.empty((0, self.dim), dtype=self.dtype, device=self.gp_device)

            # [Fix] Updated sampler API: sample_shape instead of num_samples
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

            for out_idx in range(self.num_outputs):
                q_i = int(quotas[out_idx])
                if q_i <= 0:
                    continue
                
                # Objective selects a single output dimension
                def obj_fn(samples, X=None, idx=out_idx):
                    return samples[..., idx]
                
                acq = qPosteriorStandardDeviation(
                    model=gp,
                    sampler=sampler,
                    objective=obj_fn,
                )
                
                if current_batch_points.shape[0] > 0:
                    acq.set_X_pending(current_batch_points)
                
                cand, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds,
                    q=q_i,
                    num_restarts=self.config.num_acq_restarts,
                    raw_samples=self.config.raw_acq_samples,
                    options={"batch_limit": 5, "maxiter": 100, "nonnegative": True},
                )
                
                cand = self._avoid_repeated_points(cand, batch_check=current_batch_points)
                
                X_new_list.append(cand)
                current_batch_points = torch.cat([current_batch_points, cand], dim=0)
            
            if not X_new_list:
                print("[Stop] No candidates found.")
                break
                
            X_batch = torch.cat(X_new_list, dim=0)
            print(f"[Batch] Evaluating {X_batch.shape[0]} new candidates...")
            
            success_count = 0
            for x_u in X_batch:
                x_phys = self._scaled_to_original(x_u)
                
                print(f"  [Proposal] Params: {x_phys}") 

                disp = self.run_simulation(x_phys)
                
                if disp is not None:
                    self.train_X = torch.cat([self.train_X, x_u.view(1, -1)], dim=0)
                    y_t = torch.tensor(disp, device=self.gp_device, dtype=self.dtype)
                    self.train_Y_raw = torch.cat([self.train_Y_raw, y_t.view(1, -1)], dim=0)
                    self._save_iteration_data(x_phys, disp)
                    success_count += 1
                    print(f"  -> Success {success_count}: avg disp={np.mean(disp):.6f}")
                    
        # Final best point (by minimal mean absolute displacement)
        if self.train_Y_raw.shape[0] > 0:
            scores = self.train_Y_raw.abs().mean(dim=1).cpu().numpy()
            best_idx = np.argmin(scores)
            best_x_u = self.train_X[best_idx]
            best_x = self._scaled_to_original(best_x_u)
            print("\n" + "="*60)
            print(f"[Optimization Finished] Best Result: {best_x}")
            print(f"  Disp: {self.train_Y_raw[best_idx].cpu().numpy()}")
            print("="*60)
            return best_x, self.train_Y_raw[best_idx].cpu().numpy()
        return None, None