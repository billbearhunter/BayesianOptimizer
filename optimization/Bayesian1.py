import os
import numpy as np
import torch
import taichi as ti
from typing import List, Tuple

from scipy.stats import qmc

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement


# (Do not modify this block to respect your requirement)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)


def _unit_bounds(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [0,1]^d bounds as a (2, d) tensor for BoTorch optimize_acqf."""
    lb = torch.zeros(d, dtype=dtype, device=device)
    ub = torch.ones(d, dtype=dtype, device=device)
    return torch.stack([lb, ub], dim=0)


class BayesianOptimizer:
    """LHS → 8-output GP → Linear(mean of 8) objective → joint qNEI.

    Interface is kept the same as your original Bayesian1.py (public methods,
    argument order, taichi calls). The Taichi section remains unmodified.
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
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float64
        print(f"Using device: {self.device}")

        # External simulator (Taichi) passed in; we do not modify its code
        self.simulator = simulator

        # Paths
        self.output_dir = output_dir or "results_bayes_routeA"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_file = os.path.join(self.output_dir, "optimization_results.csv")

        # BO settings
        self.n_initial_points = int(n_initial_points)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)  # q

        # Bounds (original scale) and unit cube bounds
        self.bounds = torch.tensor(bounds_list, dtype=self.dtype, device=self.device).t()  # (2, d)
        self.d = self.bounds.shape[1]
        self.unit_bounds = _unit_bounds(self.d, self.device, self.dtype)

        # Training data containers
        self.train_X = torch.empty((0, self.d), dtype=self.dtype, device=self.device)  # normalized
        self.train_Y = torch.empty((0, 8), dtype=self.dtype, device=self.device)      # 8 outputs

        self._init_results_file()

    # ------------------------------ I/O helpers ------------------------------
    def _init_results_file(self):
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{i+1:02d}" for i in range(8)]
        with open(self.results_file, "w") as f:
            f.write(",".join(headers) + "\n")  # newline per header row

    def _save_iteration_data(self, params_numpy: np.ndarray, displacements_numpy: np.ndarray):
        row = params_numpy.tolist() + displacements_numpy.tolist()
        with open(self.results_file, "a") as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")  # newline per record

    # ------------------------------ Simulation ------------------------------
    def run_simulation(self, params_numpy: np.ndarray) -> np.ndarray:
        """Call your Taichi simulator; returns 8 displacements.

        We do NOT change any Taichi logic; we simply call your existing API.
        """
        n, eta, sigma_y, width, height = params_numpy
        self.simulator.configure_geometry(width, height)
        displacements = self.simulator.run_simulation(n, eta, sigma_y)

        if displacements is None:
            return np.zeros(8, dtype=float)
        y = np.array(displacements, dtype=float).reshape(-1)
        if not np.all(np.isfinite(y)) or y.size < 8:
            out = np.zeros(8, dtype=float)
            out[: min(8, y.size)] = y[: min(8, y.size)]
            return out
        return y[:8]

    # ------------------------------ Modeling ---------------------------------
    def _fit_gp_8out(self) -> SingleTaskGP:
        model = SingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            outcome_transform=Standardize(m=8),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    # ------------------------------ Acquisition: joint qNEI ------------------
    def _select_q_joint_nei(self, model: SingleTaskGP, q: int) -> torch.Tensor:
        """Jointly pick q points in [0,1]^d using qNEI with linear mean objective."""
        weights = torch.ones(8, dtype=self.dtype, device=self.device) / 8.0
        objective = LinearMCObjective(weights=weights)

        acqf = qNoisyExpectedImprovement(
            model=model,
            X_baseline=self.train_X,
            objective=objective,
            prune_baseline=True,
            num_fantasies=32,
            # cache_root=True,  <--- BUGGY LINE REMOVED
        )
        cands, _ = optimize_acqf(
            acqf,
            bounds=self.unit_bounds,
            q=q,
            num_restarts=12,      # adjust for speed/robustness
            raw_samples=256,      # adjust for speed/robustness
            options={"batch_limit": 5, "maxiter": 200},
        )
        return cands.detach()  # (q, d)

    # ------------------------------ Main loop --------------------------------
    def optimize(self):
        print("Starting optimization: LHS → GP(8) → joint qNEI (linear mean objective)")

        # 1) Initial LHS design (normalized)
        lhs = qmc.LatinHypercube(d=self.d)
        X0_unit = torch.tensor(lhs.random(self.n_initial_points), dtype=self.dtype, device=self.device)

        # Evaluate initial points via Taichi
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

        # 2) Iterations
        for b in range(self.n_batches):
            print(f"\n=== Batch {b+1}/{self.n_batches} ===")

            model = self._fit_gp_8out()

            # joint qNEI to pick q points at once
            X_batch_unit = self._select_q_joint_nei(model, q=self.batch_size)
            X_batch_orig = unnormalize(X_batch_unit, bounds=self.bounds)

            # Evaluate batch via Taichi
            Y_batch = []
            X_batch_orig_numpy = X_batch_orig.detach().cpu().numpy()
            for j in range(self.batch_size):
                y_np = self.run_simulation(X_batch_orig_numpy[j])
                Y_batch.append(y_np)
                self._save_iteration_data(X_batch_orig_numpy[j], y_np)

            Y_batch_t = torch.tensor(np.array(Y_batch), dtype=self.dtype, device=self.device)
            self.train_X = torch.cat([self.train_X, X_batch_unit], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y_batch_t], dim=0)

            # Progress: report best by average over 8 outputs
            avg_now = self.train_Y.mean(dim=1)
            best_idx = int(torch.argmax(avg_now).item())
            best_params = unnormalize(self.train_X[best_idx:best_idx+1, :], bounds=self.bounds).squeeze(0).cpu().numpy()
            print(
                f"Current best avg: {float(avg_now[best_idx]):.6f} at "
                f"n={best_params[0]:.6f}, eta={best_params[1]:.6f}, sigma_y={best_params[2]:.6f}, "
                f"width={best_params[3]:.6f}, height={best_params[4]:.6f}"
            )

        # Final best
        avg_disp = self.train_Y.mean(dim=1)
        best_idx = int(torch.argmax(avg_disp).item())
        best_params_orig = unnormalize(self.train_X[best_idx:best_idx+1, :], bounds=self.bounds).squeeze(0).cpu().numpy()
        best_displacements = self.train_Y[best_idx].detach().cpu().numpy()

        print("\n--- Best Result (by average displacement over 8 outputs) ---")
        print(
            f"Params: n={best_params_orig[0]:.6f}, eta={best_params_orig[1]:.6f}, "
            f"sigma_y={best_params_orig[2]:.6f}, width={best_params_orig[3]:.6f}, height={best_params_orig[4]:.6f}"
        )
        print(f"Average displacement: {float(avg_disp[best_idx]):.6f}")
        print("Optimization completed.")

        return best_params_orig, best_displacements