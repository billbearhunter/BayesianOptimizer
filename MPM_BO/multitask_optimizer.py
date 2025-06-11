import numpy as np
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from base_optimizer import BaseBayesianOptimizer

class MultiTaskBayesianOptimizer(BaseBayesianOptimizer):
    def _fit_model(self):
        """
        Fits a ModelListGP composed of separate SingleTaskGP models for each task.

        Returns:
            ModelListGP: Combined multi-output GP model.
        """
        X_t = torch.tensor(self.X, dtype=torch.double)
        models = []
        for i in range(self.n_tasks):
            Y_t = torch.tensor(self.Y[:, i:i+1], dtype=torch.double)
            model = SingleTaskGP(
                X_t, Y_t,
                input_transform=Normalize(
                    d=self.dims,
                    bounds=torch.tensor(self.bounds.T, dtype=torch.double)
                ),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            models.append(model)
        return ModelListGP(*models)

    def _suggest_next(self, model):
        """
        Suggests the next query point using scalarized acquisition functions.

        Args:
            model (ModelListGP): Combined GP model with multiple outputs.

        Returns:
            np.ndarray: Next candidate point.
        """
        weights = torch.tensor(self.task_weights, dtype=torch.double)
        posterior_tf = ScalarizedPosteriorTransform(weights=weights)

        best_scalar = float(np.max(np.vstack(self.best_values).dot(self.task_weights)))

        if self.acq == "EI":
            acq_func = LogExpectedImprovement(
                model,
                best_f=best_scalar,
                posterior_transform=posterior_tf,
                maximize=True
            )
        elif self.acq == "UCB":
            acq_func = UpperConfidenceBound(
                model,
                beta=self.beta,
                posterior_transform=posterior_tf
            )
        else:
            raise ValueError(f"Unsupported acquisition: {self.acq}")

        bounds_tensor = torch.tensor(self.bounds, dtype=torch.double).T
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_tensor,
            q=1,
            num_restarts=10,
            raw_samples=50
        )

        return candidate.detach().cpu().numpy().ravel()
