"""
Bayesian_MultiTaskGP.py

A minimal, production-ready wrapper around a Kronecker multi-task GP (8 outputs)
that can be trained, saved, loaded, and used to predict 8-dimensional outputs
from an input vector. Comments are in English as requested.

Requirements (typical versions, adjust as needed):
- torch >= 2.0
- gpytorch >= 1.11
- botorch >= 0.9

This file provides:
- class MultiTaskBayesModel: fit / predict / save / load
- optional propose_next(q) using qLogExpectedImprovement (objective = mean across 8 outputs)
- an example __main__ section with synthetic data to demonstrate usage

Notes:
- The model assumes block design for multi-task GP: for each input, all 8 outputs
  are observed (shape (N, 8)).
- Inputs are normalized to [0, 1]^d using user-specified bounds.
- Outputs are standardized internally via BoTorch's Standardize(m=8).
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import gpytorch
from botorch.models import KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.mlls import ExactMarginalLogLikelihood
from botorch.transforms.outcome import Standardize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf


@dataclass
class ModelMeta:
    """Metadata saved with the model for safe reloading."""
    n_tasks: int
    input_dim: int
    dtype: str  # "float32" or "float64"
    bounds_lb: list  # list of floats, length d
    bounds_ub: list  # list of floats, length d

    @staticmethod
    def from_tensors(lb: torch.Tensor, ub: torch.Tensor, n_tasks: int, dtype: torch.dtype) -> "ModelMeta":
        return ModelMeta(
            n_tasks=n_tasks,
            input_dim=lb.numel(),
            dtype="float64" if dtype == torch.float64 else "float32",
            bounds_lb=lb.detach().cpu().tolist(),
            bounds_ub=ub.detach().cpu().tolist(),
        )

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        lb = torch.tensor(self.bounds_lb, device=device)
        ub = torch.tensor(self.bounds_ub, device=device)
        dtype = torch.float64 if self.dtype == "float64" else torch.float32
        return lb, ub, dtype


class MultiTaskBayesModel:
    """A wrapper for an 8-output Kronecker multi-task GP with save/load/predict.

    Usage:
    >>> model = MultiTaskBayesModel(bounds=(lb, ub), n_tasks=8, device="cuda")
    >>> model.fit(train_X_raw, train_Y)
    >>> mean, var = model.predict(X_raw)  # shapes: (N, 8)
    >>> model.save("mtgp_8out.pt")
    >>> reloaded = MultiTaskBayesModel.load("mtgp_8out.pt", device="cpu")
    >>> mean2, var2 = reloaded.predict(X_raw)
    """

    def __init__(
        self,
        bounds: Tuple[torch.Tensor, torch.Tensor],
        n_tasks: int = 8,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
    ) -> None:
        assert n_tasks > 1, "n_tasks must be >= 2 for a non-trivial multi-task GP"
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        lb, ub = bounds
        lb = lb.to(self.device, self.dtype)
        ub = ub.to(self.device, self.dtype)
        assert lb.ndim == 1 and ub.ndim == 1 and lb.numel() == ub.numel(), "bounds must be (2,d) with matching shapes"
        assert torch.all(ub > lb), "Each upper bound must be greater than the corresponding lower bound"

        self.lb = lb
        self.ub = ub
        self.input_dim = lb.numel()
        self.n_tasks = n_tasks

        self.model: Optional[KroneckerMultiTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None

        # Cached training data in unit space for reproducible reloads
        self._train_X_unit: Optional[torch.Tensor] = None  # shape: (N, d)
        self._train_Y: Optional[torch.Tensor] = None       # shape: (N, n_tasks)

    # --------------------------
    # Utilities
    # --------------------------
    def _to_unit(self, X_raw: torch.Tensor) -> torch.Tensor:
        """Map raw inputs from [lb, ub] to [0,1]^d (broadcast-safe)."""
        return (X_raw - self.lb) / (self.ub - self.lb)

    def _check_and_cast_X(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 2 and X.shape[1] == self.input_dim, f"Expected (N, {self.input_dim}) input"
        return X.to(self.device, self.dtype)

    def _check_and_cast_Y(self, Y: torch.Tensor) -> torch.Tensor:
        assert Y.ndim == 2 and Y.shape[1] == self.n_tasks, f"Expected (N, {self.n_tasks}) targets"
        return Y.to(self.device, self.dtype)

    # --------------------------
    # Training
    # --------------------------
    def fit(self, train_X_raw: torch.Tensor, train_Y: torch.Tensor, *, training_iters: int = 200) -> None:
        """Fit the Kronecker multi-task GP.

        Args:
            train_X_raw: (N, d) inputs in ORIGINAL scale (within [lb, ub]).
            train_Y:     (N, n_tasks) targets. All tasks must be observed per input (block design).
            training_iters: Max iterations for hyperparameter optimization (Adam+LBFGS under the hood).
        """
        X_raw = self._check_and_cast_X(train_X_raw)
        Y = self._check_and_cast_Y(train_Y)

        # Normalize inputs to unit hypercube
        X_unit = self._to_unit(X_raw)
        self._train_X_unit = X_unit.detach().clone()
        self._train_Y = Y.detach().clone()

        # Build model with standardized outcomes (per-task mean/var learned from data)
        outcome_tf = Standardize(m=self.n_tasks)
        self.model = KroneckerMultiTaskGP(X_unit, Y, rank=1, outcome_transform=outcome_tf).to(self.device, self.dtype)
        self.model.train()
        self.model.likelihood.train()

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Optimize hyperparameters
        fit_gpytorch_mll(self.mll, options={"maxiter": training_iters})

    # --------------------------
    # Prediction
    # --------------------------
    @torch.no_grad()
    def predict(self, X_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the 8-D output for raw-scale inputs.

        Args:
            X_raw: (N, d) inputs in ORIGINAL scale.
        Returns:
            mean: (N, n_tasks), predictive means per task
            var:  (N, n_tasks), predictive marginal variances per task
        """
        assert self.model is not None, "Call fit() or load() before predict()"
        X_raw = self._check_and_cast_X(X_raw)
        X_unit = self._to_unit(X_raw)

        self.model.eval(); self.model.likelihood.eval()
        posterior = self.model.posterior(X_unit)
        mean = posterior.mean.view(-1, self.n_tasks)
        var = posterior.variance.view(-1, self.n_tasks)
        return mean, var

    # --------------------------
    # Propose next candidates (optional)
    # --------------------------
    @torch.no_grad()
    def propose_next(self, q: int = 1, *, samples: int = 256) -> torch.Tensor:
        """Propose q new candidates by maximizing qLogExpectedImprovement.

        Objective = mean across the 8 outputs. This is a common scalarization; adjust
        weights for other trade-offs.

        Returns:
            X_next_raw: (q, d) candidates in ORIGINAL scale.
        """
        assert self.model is not None and self._train_X_unit is not None and self._train_Y is not None, "Model must be fit before proposing"

        self.model.eval(); self.model.likelihood.eval()

        # Linear objective that averages the 8 tasks
        weights = torch.ones(self.n_tasks, device=self.device, dtype=self.dtype) / self.n_tasks
        objective = LinearMCObjective(weights=weights)

        # Compute best_f in standardized space if outcome_transform exists
        if getattr(self.model, "outcome_transform", None) is not None:
            Y_std, _ = self.model.outcome_transform(self._train_Y)
            best_f = (Y_std @ weights).max()
        else:
            best_f = (self._train_Y @ weights).max()

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([samples]))
        acqf = qLogExpectedImprovement(model=self.model, best_f=best_f, sampler=sampler, objective=objective)

        # Bounds in unit space
        unit_bounds = torch.stack([torch.zeros(self.input_dim, device=self.device, dtype=self.dtype),
                                   torch.ones(self.input_dim, device=self.device, dtype=self.dtype)])

        candidates, _ = optimize_acqf(
            acqf,
            bounds=unit_bounds,
            q=q,
            num_restarts=8,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )
        X_next_unit = candidates.detach()
        # Map back to raw/original scale
        X_next_raw = self.lb + X_next_unit * (self.ub - self.lb)
        return X_next_raw

    # --------------------------
    # Persistence
    # --------------------------
    def save(self, path: str) -> None:
        """Save model, training data (unit-space), and metadata for exact reloads."""
        assert self.model is not None and self._train_X_unit is not None and self._train_Y is not None, "Nothing to save (fit the model first)"
        meta = ModelMeta.from_tensors(self.lb, self.ub, self.n_tasks, self.dtype)
        payload = {
            "meta": meta.__dict__,
            "train_X_unit": self._train_X_unit.detach().cpu(),
            "train_Y": self._train_Y.detach().cpu(),
            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.model.likelihood.state_dict(),
        }
        torch.save(payload, path)

        # Also drop a small JSON sidecar for quick inspection (optional)
        sidecar = os.path.splitext(path)[0] + ".json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(payload["meta"], f, indent=2)

    @staticmethod
    def load(path: str, device: Optional[torch.device | str] = None) -> "MultiTaskBayesModel":
        """Reload a previously saved model for immediate prediction/proposal."""
        map_loc = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=map_loc)

        meta = ModelMeta(**payload["meta"])
        lb, ub, dtype = meta.to_tensors(map_loc)

        model = MultiTaskBayesModel(bounds=(lb, ub), n_tasks=meta.n_tasks, dtype=dtype, device=map_loc)

        # Rebuild the model with cached training data in unit space
        train_X_unit = payload["train_X_unit"].to(map_loc, dtype)
        train_Y = payload["train_Y"].to(map_loc, dtype)
        model._train_X_unit = train_X_unit.clone()
        model._train_Y = train_Y.clone()

        outcome_tf = Standardize(m=meta.n_tasks)
        model.model = KroneckerMultiTaskGP(train_X_unit, train_Y, rank=1, outcome_transform=outcome_tf).to(map_loc, dtype)
        model.mll = ExactMarginalLogLikelihood(model.model.likelihood, model.model)

        # Load learned parameters (kernel hyperparams, task covariances, outcome transform stats, etc.)
        model.model.load_state_dict(payload["model_state_dict"])  # includes transform buffers
        model.model.likelihood.load_state_dict(payload["likelihood_state_dict"])  # for completeness

        model.model.eval(); model.model.likelihood.eval()
        return model