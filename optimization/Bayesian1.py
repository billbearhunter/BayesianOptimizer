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
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement


from config.config import (
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
    MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT,
    DEFAULT_OUTPUT_DIR,
)

# ti.init(arch=ti.cpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)


def _unit_bounds(d: int, device, dtype):
    """Return [0,1]^d bounds tensor for optimize_acqf."""
    lb = torch.zeros(d, dtype=dtype, device=device)
    ub = torch.ones(d, dtype=dtype, device=device)
    return torch.stack([lb, ub], dim=0)


class SumPosteriorVariance(AcquisitionFunction):
    """acq(X) = sum over candidates and outputs of the posterior variance.

    For a multi-output GP with outputs in the last dimension, we sum along the
    output dimension and the q-batch dimension. Any leading fantasy batch dims
    are averaged out, returning a 1D tensor of shape (b,) for b candidate batches.
    """
    def __init__(self, model):
        super().__init__(model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(X)
        var = posterior.variance  # (..., q, 8)
        # Sum over outputs and q, average over fantasy batch dims if any.
        while var.dim() > 2:
            var = var.mean(dim=0)
        return var.sum(dim=-1)


class BayesianOptimizer:
    """LHS → 8-output GP → Linear objective (mean of 8) → sequential qNEI.

    - Inputs normalized to [0,1]^d.
    - Initial design via Latin Hypercube Sampling (LHS).
    - Model: SingleTaskGP with Standardize(m=8) (independent outputs).
    - Acquisition: qNoisyExpectedImprovement with LinearMCObjective (mean over 8 outputs).
    - q-batch: Sequential greedy with fantasization (X_pending) to avoid clustering.
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
        self.dtype = torch.float64  # keep double for stability
        print(f"Using device: {self.device}")

        self.simulator = simulator
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self.n_batches = n_batches
        self.batch_size = batch_size  # q

        # Original-scale bounds (2, d) then infer d and min initial points
        self.bounds = torch.tensor(bounds_list, dtype=self.dtype, device=self.device).t()
        self.d = self.bounds.shape[1]
        self.n_initial_points = max(n_initial_points, max(2, self.d))  # >= max(2,d)

        # Unit cube bounds for acquisition
        self.unit_bounds = _unit_bounds(self.d, self.device, self.dtype)

        # Training datasets: normalized X in [0,1]^d, raw Y in R^8
        self.train_X = torch.empty((0, self.d), dtype=self.dtype, device=self.device)   # N x d
        self.train_Y = torch.empty((0, 8), dtype=self.dtype, device=self.device)        # N x 8

        # CSV logging
        self.results_file = os.path.join(self.output_dir, "optimization_results.csv")
        self._init_results_file()

    # ------------------------------ I/O helpers ------------------------------
    def _init_results_file(self):
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{i+1:02d}" for i in range(8)]
        with open(self.results_file, "w") as f:
            f.write(",".join(headers) + "")

    def _save_iteration_data(self, params_numpy: np.ndarray, displacements_numpy: np.ndarray):
        row = params_numpy.tolist() + displacements_numpy.tolist()
        with open(self.results_file, "a") as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "")

    # ------------------------------ Simulation ------------------------------
    def run_simulation(self, params_numpy: np.ndarray) -> np.ndarray:
        """Run Taichi simulator and return 8 displacements (kept as original)."""
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

    # ------------------------------ Acquisition (Route A) --------------------
    def _select_q_by_maxvar(self, model: SingleTaskGP, q: int) -> torch.Tensor:
        """
        Sequential-greedy selection of q points in [0,1]^d using qNEI
        with a linear objective equal to the mean of 8 outputs.
        """
        unit_bounds = _unit_bounds(self.d, self.device, self.dtype)

        weights = torch.ones(8, dtype=self.dtype, device=self.device) / 8.0
        objective = LinearMCObjective(weights=weights)

        X_pending = None
        selected = []
        for _ in range(q):
            acqf = qNoisyExpectedImprovement(
                model=model,
                X_baseline=self.train_X,
                objective=objective,
                X_pending=X_pending,
                prune_baseline=True,
                num_fantasies=32,
                cache_root=True,
            )
            cand, _ = optimize_acqf(
                acqf,
                bounds=unit_bounds,
                q=1,
                num_restarts=10,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 200},
            )
            cand = cand.detach()
            selected.append(cand)
            X_pending = cand if X_pending is None else torch.cat([X_pending, cand], dim=0)

        return torch.cat(selected, dim=0)  # (q, d)

    # ------------------------------ Workflow ---------------------------------
    def optimize(self):
        """
        Main loop: LHS → evaluate → fit GP → repeat(sequential qNEI → evaluate → refit).
        """
        print("Starting optimization: LHS → GP(8) → Sequential qNEI (linear mean objective) → Sequential Evaluation")

        # 1) Initial LHS design (normalized) and evaluation
        d = self.d
        lhs = qmc.LatinHypercube(d=d)
        X0_unit = torch.tensor(lhs.random(self.n_initial_points), dtype=self.dtype, device=self.device)

        # Evaluate initial points
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

        # 2) Iterations: fit GP, select q by qNEI, evaluate, augment data
        for b in range(self.n_batches):
            print(f"=== Batch {b+1}/{self.n_batches} ===")
            model = self._fit_gp_8out()

            # Select a batch of q points in unit space
            X_batch = self._select_q_by_maxvar(model, q=self.batch_size)  # (q, d)

            # Evaluate batch sequentially (simulator is often stateful / expensive)
            X_batch_orig = unnormalize(X_batch, bounds=self.bounds)
            Y_batch = []
            for j in range(self.batch_size):
                x_orig_np = X_batch_orig[j].cpu().numpy()
                y_np = self.run_simulation(x_orig_np)
                Y_batch.append(y_np)
                self._save_iteration_data(x_orig_np, y_np)

            # Append
            Y_batch_t = torch.tensor(np.array(Y_batch), dtype=self.dtype, device=self.device)
            self.train_X = torch.cat([self.train_X, X_batch], dim=0)
            self.train_Y = torch.cat([self.train_Y, Y_batch_t], dim=0)

            # Progress report by the average over 8 outputs
            avg_now = self.train_Y.mean(dim=1)
            best_idx = int(torch.argmax(avg_now).item())
            best_params = unnormalize(self.train_X[best_idx:best_idx+1, :], bounds=self.bounds).squeeze(0).cpu().numpy()
            print(
                f"Current best avg: {float(avg_now[best_idx]):.6f} at "
                f"n={best_params[0]:.6f}, eta={best_params[1]:.6f}, sigma_y={best_params[2]:.6f}, "
                f"width={best_params[3]:.6f}, height={best_params[4]:.6f}"
            )

        # Final best (by mean over 8 outputs)
        avg_disp = self.train_Y.mean(dim=1)
        best_idx = int(torch.argmax(avg_disp).item())
        best_params_orig = unnormalize(self.train_X[best_idx:best_idx+1, :], bounds=self.bounds).squeeze(0).cpu().numpy()
        best_displacements = self.train_Y[best_idx].detach().cpu().numpy()

        print("--- Best Result (by average displacement over 8 outputs) ---")
        print(
            f"Params: n={best_params_orig[0]:.6f}, eta={best_params_orig[1]:.6f}, "
            f"sigma_y={best_params_orig[2]:.6f}, width={best_params_orig[3]:.6f}, height={best_params_orig[4]:.6f}"
        )
        print(f"Average displacement: {float(avg_disp[best_idx]):.6f}")
        print("Optimization completed.")

        return best_params_orig, best_displacements
