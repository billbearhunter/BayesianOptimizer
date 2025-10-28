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
from botorch.sampling.normal import SobolQMCNormalSampler

# Project config (parameter ranges, paths). Keep import path consistent with run_optimization.py
from config.config import (
    MIN_N, MAX_N, MIN_ETA, MAX_ETA, MIN_SIGMA_Y, MAX_SIGMA_Y,
    MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT,
    DEFAULT_OUTPUT_DIR,
)

# Initialize Taichi (GPU if available)
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)


def _unit_bounds(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return [0,1]^d bounds as a (2, d) tensor for BoTorch optimize_acqf."""
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
        # X shapes: (q, d) or (b, q, d). Convert to (b, q, d).
        if X.dim() == 2:
            X = X.unsqueeze(0)  # -> (1, q, d)
        posterior = self.model.posterior(X)
        # posterior.variance shape: (*fantasy_batch, b, q, m) or (b, q, m)
        var = posterior.variance
        # Sum over outputs (m) then over q
        summed = var.sum(dim=-1).sum(dim=-1)  # -> (*fantasy_batch, b)
        # If there are fantasy batch dims, average them away to get (b,)
        while summed.dim() > 1:
            summed = summed.mean(dim=0)
        return summed  # (b,)


class BayesianOptimizer:
    """LHS → 8-output GP → MaxVar active learning with sequential q-batch fantasization.

    - Inputs are normalized to [0,1]^d for modeling / selection.
    - Initial design via Latin Hypercube Sampling (LHS).
    - Model: SingleTaskGP with Standardize(m=8) outcome transform (independent outputs).
    - Acquisition: Sum of posterior variances (higher = more uncertain).
    - q-batch: greedy one-by-one with `model.fantasize(...)` after each pick.
    - No Sobol fallback is used (robustness ensured by >= 2 initial points + scalar acq).
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
        self.n_initial_points = max(n_initial_points, max(2, self.d))  # >= max(2, d)

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
            f.write(",".join(headers) + "\n")

    def _save_iteration_data(self, params_numpy: np.ndarray, displacements_numpy: np.ndarray):
        row = params_numpy.tolist() + displacements_numpy.tolist()
        with open(self.results_file, "a") as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")

    # ------------------------------ Design & sim -----------------------------
    def collect_initial_points(self) -> torch.Tensor:
        """Generate n_initial_points LHS samples in [0,1]^d."""
        n = self.n_initial_points
        print(f"Generating {n} initial points via Latin Hypercube in [0,1]^{self.d} ...")
        sampler = qmc.LatinHypercube(d=self.d)
        X_scaled = sampler.random(n=n)  # (n, d) in [0,1]
        return torch.tensor(X_scaled, dtype=self.dtype, device=self.device)

    def run_simulation(self, params_numpy: np.ndarray) -> np.ndarray:
        """Call external simulator and return 8D displacement vector (numpy).

        Any invalid/NaN response is replaced by zeros for robustness.
        """
        n, eta, sigma_y, width, height = params_numpy
        # Width/height define geometry for the simulator
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
        """Fit an 8-output GP (independent outputs, shared X) with Standardize(m=8)."""
        gp = SingleTaskGP(self.train_X, self.train_Y, outcome_transform=Standardize(m=8))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    # ------------------------------ Acquisition ------------------------------
    def _select_q_by_maxvar(self, model: SingleTaskGP, q: int) -> torch.Tensor:
        """Greedy selection of q points in [0,1]^d using MaxVar with fantasization."""
        unit_bounds = _unit_bounds(self.d, self.device, self.dtype)
        chosen = []
        fantasy_model = model
        fantasy_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        for j in range(q):
            acq = SumPosteriorVariance(fantasy_model)
            candidate, acq_val = optimize_acqf(
                acq_function=acq,
                bounds=unit_bounds,
                q=1,                         # pick one at a time
                num_restarts=10,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )
            cand = candidate.view(-1, self.d).detach()  # (1, d)
            chosen.append(cand)

            # Safe logging of acquisition values (avoid .item() on tensors with dims)
            try:
                v = acq_val
                msg = float(v.mean()) if hasattr(v, "mean") else float(v)
                print(f"  Picked #{j+1} with acquisition ≈ {msg:.6f}")
            except Exception:
                print(f"  Picked #{j+1}")

            # Kriging Believer: condition on posterior mean at the picked point
            with torch.no_grad():
                post = fantasy_model.posterior(cand)      # mean shape (1, m)
                y_belief = post.mean.detach()              # (1, m)
                fantasy_model = fantasy_model.condition_on_observations(
                    X=cand, Y=y_belief
                )

        return torch.cat(chosen, dim=0)  # (q, d)

    # ------------------------------ Workflow ---------------------------------
    def optimize(self):
        """Main loop: LHS → evaluate → fit GP → repeat(q-batch MaxVar → evaluate → refit)."""
        print("Starting optimization: LHS → GP(8) → MaxVar (sequential q-batch)")

        # 1) Initial LHS design (normalized) and evaluation
        X0 = self.collect_initial_points()  # (n0, d)
        print(f"Evaluating {self.n_initial_points} initial points ...")
        for x_scaled in X0:
            x_scaled = x_scaled.view(1, -1)
            x_orig = unnormalize(x_scaled, bounds=self.bounds).squeeze(0).cpu().numpy()
            y = self.run_simulation(x_orig)  # (8,)
            # append
            self.train_X = torch.cat([self.train_X, x_scaled], dim=0)
            self.train_Y = torch.cat([self.train_Y, torch.tensor(y, dtype=self.dtype, device=self.device).view(1, -1)], dim=0)
            self._save_iteration_data(x_orig, y)

        # 2) Iterations: fit GP, select q by MaxVar, evaluate, augment data
        for b in range(self.n_batches):
            print(f"\n--- Batch {b+1}/{self.n_batches} ---")
            # Fit GP on all data so far
            model = self._fit_gp_8out()

            # Greedy-build a q-batch in [0,1]^d using fantasization
            q = self.batch_size
            X_batch = self._select_q_by_maxvar(model, q=q)  # (q, d)

            # Evaluate the selected batch
            for j in range(X_batch.shape[0]):
                x_scaled = X_batch[j:j+1, :]  # (1, d)
                x_orig = unnormalize(x_scaled, bounds=self.bounds).squeeze(0).cpu().numpy()
                y = self.run_simulation(x_orig)
                # append
                self.train_X = torch.cat([self.train_X, x_scaled], dim=0)
                self.train_Y = torch.cat([self.train_Y, torch.tensor(y, dtype=self.dtype, device=self.device).view(1, -1)], dim=0)
                self._save_iteration_data(x_orig, y)
                print(f"  Evaluated {j+1}/{q}: n={x_orig[0]:.6f}, eta={x_orig[1]:.6f}, sigma_y={x_orig[2]:.6f}, "
                      f"width={x_orig[3]:.6f}, height={x_orig[4]:.6f}")

        # 3) Choose a "best" point for compatibility (max average of 8 outputs)
        avg_disp = self.train_Y.mean(dim=1)  # (N,)
        best_idx = int(torch.argmax(avg_disp).item())
        best_params_orig = unnormalize(self.train_X[best_idx:best_idx+1, :], bounds=self.bounds).squeeze(0).cpu().numpy()
        best_displacements = self.train_Y[best_idx].detach().cpu().numpy()

        print("\n--- Best Result (by average displacement over 8 outputs) ---")
        print(f"Params: n={best_params_orig[0]:.6f}, eta={best_params_orig[1]:.6f}, sigma_y={best_params_orig[2]:.6f}, "
              f"width={best_params_orig[3]:.6f}, height={best_params_orig[4]:.6f}")
        print(f"Average displacement: {float(avg_disp[best_idx]):.6f}")
        print("Optimization completed.")

        return best_params_orig, best_displacements
