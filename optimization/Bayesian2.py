import os
import numpy as np
import torch
import taichi as ti
from typing import List, Tuple

from scipy.stats import qmc

# BoTorch / GPyTorch
from botorch.models.multitask import KroneckerMultiTaskGP  # MTGP for block design (n,d) -> (n,m)
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import LinearMCObjective
# --- FIX 1: IMPORT THE RECOMMENDED FUNCTION ---
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

# Keep your original Taichi init exactly the same
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)


# ------------------------------ Helpers ------------------------------------
def _unit_bounds(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [0,1]^d bounds as a (2, d) tensor for BoTorch optimize_acqf."""
    lb = torch.zeros(d, dtype=dtype, device=device)
    ub = torch.ones(d, dtype=dtype, device=device)
    return torch.stack([lb, ub], dim=0)


class BayesianOptimizer:
    """LHS → 8-output MTGP (KroneckerMultiTaskGP) → Linear(mean of 8) objective → joint qLogEI.

    This version models the 8 correlated outputs using KroneckerMultiTaskGP (block design),
    which natively takes train_X: (n, d) and train_Y: (n, 8) and captures task correlations.
    
    FIXED: Swapped qNEI for qLogEI per BoTorch recommendation.
    FIXED: Corrected SobolQMCNormalSampler argument from 'num_samples' to 'sample_shape'.
    """

    def __init__(
        self,
        simulator,
        bounds_list: List[Tuple[float, float]],
        output_dir: str,
        n_initial_points: int,
        n_batches: int,
        batch_size: int,
        device=None,
    ):
        # Device / dtype setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float64
        print(f"Using device: {self.device}")

        # External simulator (Taichi) and output
        self.simulator = simulator
        self.output_dir = output_dir or "results_bayes_routeA"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_file = os.path.join(self.output_dir, "optimization_results.csv")

        # Search / evaluation settings
        self.n_initial_points = int(n_initial_points)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)  # q

        # Number of tasks (multi-output size)
        self.n_tasks = 8

        # Bounds and unit cube convenience
        self.bounds = torch.tensor(bounds_list, dtype=self.dtype, device=self.device).t()  # (2, d)
        self.d = self.bounds.shape[1]
        self.unit_bounds = _unit_bounds(self.d, self.device, self.dtype)

        # Training containers: X is normalized in [0,1]^d; Y has shape (n, 8)
        self.train_X = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
        self.train_Y = torch.empty((0, self.n_tasks), dtype=self.dtype, device=self.device)

        self._init_results_file()

    # ------------------------------ I/O helpers ------------------------------
    def _init_results_file(self):
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{i+1:02d}" for i in range(self.n_tasks)]
        with open(self.results_file, "w") as f:
            f.write(",".join(headers) + "\n")

    def _save_iteration_data(self, params_numpy: np.ndarray, displacements_numpy: np.ndarray):
        """Append one row of (params, 8 displacements) to CSV."""
        row = params_numpy.tolist() + displacements_numpy.tolist()
        with open(self.results_file, "a") as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")

    # ------------------------------ Simulation ------------------------------
    def run_simulation(self, params_numpy: np.ndarray) -> np.ndarray:
        """Call your Taichi simulator; returns 8 displacements. (UNCHANGED)"""
        n, eta, sigma_y, width, height = params_numpy
        self.simulator.configure_geometry(width, height)
        displacements = self.simulator.run_simulation(n, eta, sigma_y)

        if displacements is None:
            return np.zeros(self.n_tasks, dtype=float)
        y = np.array(displacements, dtype=float).reshape(-1)
        if not np.all(np.isfinite(y)) or y.size < self.n_tasks:
            out = np.zeros(self.n_tasks, dtype=float)
            out[: min(self.n_tasks, y.size)] = y[: min(self.n_tasks, y.size)]
            return out
        return y[: self.n_tasks]

    # ------------------------------ Modeling ---------------------------------
    def _fit_gp_8out(self) -> KroneckerMultiTaskGP:
        """
        Fit a KroneckerMultiTaskGP to the 8-output data (block design).
        (UNCHANGED)
        """
        model = KroneckerMultiTaskGP(
            train_X=self.train_X,                             # (n, d) on unit cube
            train_Y=self.train_Y,                             # (n, 8)
            outcome_transform=Standardize(m=self.n_tasks),    # standardize 8 outputs
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    # ------------------------------ Acquisition: joint qEI ------------------
    def _select_q_joint_ei(self, model, q: int) -> torch.Tensor:
        
        weights = torch.ones(self.n_tasks, dtype=self.dtype, device=self.device) / self.n_tasks
        objective = LinearMCObjective(weights=weights)


        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        if getattr(model, "outcome_transform", None) is not None:
            Y_std = model.outcome_transform(self.train_Y)[0]      
            y_obj_std = Y_std @ weights                           
            best_f = y_obj_std.max()
        else:
            y_obj = (self.train_Y @ weights)
            best_f = y_obj.max()

        acqf = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        )
        cands, _ = optimize_acqf(
            acqf,
            bounds=self.unit_bounds,
            q=q,
            num_restarts=8,             
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )
        
        return cands.detach()

    # ------------------------------ Main loop --------------------------------
    def optimize(self):
        print("Starting optimization: LHS → KroneckerMTGP(8) → joint qLogNEI (linear mean objective)")

        # 1) Initial LHS design (on unit cube)
        lhs = qmc.LatinHypercube(d=self.d)
        X0_unit_np = lhs.random(self.n_initial_points)  # ndarray on [0,1]
        X0_unit = torch.tensor(X0_unit_np, dtype=self.dtype, device=self.device)

        # Evaluate initial points (in original bounds)
        Y0 = []
        for i in range(self.n_initial_points):
            x_unit = X0_unit[i]
            x_orig = unnormalize(x_unit.unsqueeze(0), bounds=self.bounds).squeeze(0).cpu().numpy()
            y_vals = self.run_simulation(x_orig)
            Y0.append(y_vals)
            self._save_iteration_data(x_orig, y_vals)

        Y0_t = torch.tensor(np.array(Y0), dtype=self.dtype, device=self.device)
        self.train_X = torch.cat([self.train_X, X0_unit], dim=0)
        self.train_Y = torch.cat([self.train_Y, Y0_t], dim=0)

        # 2) Sequential batches
        for b in range(self.n_batches):
            print(f"\n=== Batch {b+1}/{self.n_batches} ===")

            model = self._fit_gp_8out()

            # Select a q-batch on unit cube
            X_batch_unit = self._select_q_joint_nei(model, q=self.batch_size)

            # Map back to original bounds for simulation
            X_batch_orig = unnormalize(X_batch_unit, bounds=self.bounds)
            X_batch_orig_numpy = X_batch_orig.detach().cpu().numpy()

            # Evaluate batch
            Y_batch = []
            for j in range(self.batch_size):
                y_np = self.run_simulation(X_batch_orig_numpy[j])
                Y_batch.append(y_np)
                self._save_iteration_data(X_batch_orig_numpy[j], y_np)

            Y_batch_t = torch.tensor(np.array(Y_batch), dtype=self.dtype, device=self.device)
            self.train_X = torch.cat([self.train_X, X_batch_unit], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y_batch_t], dim=0)

            # Track current best by average over 8 outputs
            avg_now = self.train_Y.mean(dim=1)
            best_idx = int(torch.argmax(avg_now).item())
            best_params = unnormalize(
                self.train_X[best_idx : best_idx + 1, :], bounds=self.bounds
            ).squeeze(0).cpu().numpy()
            print(
                f"Current best avg: {float(avg_now[best_idx]):.6f} at "
                f"n={best_params[0]:.6f}, eta={best_params[1]:.6f}, sigma_y={best_params[2]:.6f}, "
                f"width={best_params[3]:.6f}, height={best_params[4]:.6f}"
            )

        # Final best
        avg_disp = self.train_Y.mean(dim=1)
        best_idx = int(torch.argmax(avg_disp).item())
        best_params_orig = unnormalize(
            self.train_X[best_idx : best_idx + 1, :], bounds=self.bounds
        ).squeeze(0).cpu().numpy()
        best_displacements = self.train_Y[best_idx].detach().cpu().numpy()

        print("\n--- Best Result (by average displacement over 8 outputs) ---")
        print(
            f"Params: n={best_params_orig[0]:.6f}, eta={best_params_orig[1]:.6f}, "
            f"sigma_y={best_params_orig[2]:.6f}, width={best_params_orig[3]:.6f}, height={best_params[4]:.6f}"
        )
        print(f"Average displacement: {float(avg_disp[best_idx]):.6f}")
        print("Optimization completed.")

        return best_params_orig, best_displacements