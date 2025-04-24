import numpy as np
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from .base_optimizer import BaseBayesianOptimizer

class IndependentBayesianOptimizer(BaseBayesianOptimizer):
    def _fit_model(self):
        X_t = torch.tensor(self.X, dtype=torch.double)
        models = []
        for i in range(self.n_tasks):
            Y_i = torch.tensor(self.Y[:, i].reshape(-1, 1), dtype=torch.double)
            model_i = SingleTaskGP(X_t, Y_i,
                                   input_transform=self.input_transform,
                                   outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
            fit_gpytorch_mll(mll)
            models.append(model_i)
        return models[0] if len(models) == 1 else ModelListGP(*models)

    def _suggest_next(self, model):
        weights = torch.tensor(self.task_weights, dtype=torch.double)
        posterior_tf = ScalarizedPosteriorTransform(weights=weights)
        best_scalar = float(np.max(np.vstack(self.best_values).dot(self.task_weights)))

        if self.acq == "EI":
            acq_func = LogExpectedImprovement(model, best_f=best_scalar,
                                          posterior_transform=posterior_tf, maximize=True)
        elif self.acq == "UCB":
            acq_func = UpperConfidenceBound(model, beta=self.beta, posterior_transform=posterior_tf)
        else:
            raise ValueError(f"Unknown acquisition type: {self.acq}")

        bounds_tensor = torch.tensor(self.bounds.T, dtype=torch.double)
        candidate, _ = optimize_acqf(acq_function=acq_func, bounds=bounds_tensor,
                                     q=1, num_restarts=10, raw_samples=50)
        return candidate.detach().cpu().numpy().ravel()
