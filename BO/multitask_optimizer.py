import numpy as np
import torch
from botorch.models import MultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from .base_optimizer import BaseBayesianOptimizer

class MultiTaskBayesianOptimizer(BaseBayesianOptimizer):
    def _fit_model(self):
        X_t = torch.tensor(self.X, dtype=torch.double)
        Y_t = torch.tensor(self.Y, dtype=torch.double)
        n = X_t.shape[0]
        task_idx = torch.arange(self.n_tasks, dtype=torch.long).repeat_interleave(n).unsqueeze(1)
        X_mt = torch.cat([X_t.repeat(self.n_tasks, 1), task_idx], dim=1)
        Y_mt = Y_t.T.contiguous().view(-1, 1)

        model = MultiTaskGP(X_mt, Y_mt, task_feature=self.dims,
                            input_transform=Normalize(d=self.dims + 1),
                            outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _suggest_next(self, model):
        weights = torch.tensor(self.task_weights, dtype=torch.double)
        posterior_tf = ScalarizedPosteriorTransform(weights=weights)
        best_scalar = float(np.max(np.vstack(self.best_values).dot(self.task_weights)))

        if self.acq == "EI":
            base_acq = LogExpectedImprovement(model, best_f=best_scalar,
                                              posterior_transform=posterior_tf, maximize=True)
        elif self.acq == "UCB":
            base_acq = UpperConfidenceBound(model, beta=self.beta, posterior_transform=posterior_tf)
        else:
            raise ValueError(f"Unknown acquisition type: {self.acq}")

        bounds_tensor = torch.tensor(self.bounds, dtype=torch.double).T
        candidate, _ = optimize_acqf(
            acq_function=base_acq,
            bounds=bounds_tensor,
            q=1,
            num_restarts=10,
            raw_samples=50
        )

        return candidate.detach().cpu().numpy().ravel()
