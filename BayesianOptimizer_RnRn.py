import time
import numpy as np
from typing import Callable, Sequence, List, Dict, Any
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.double)


def target_function(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must be a 1-D array of length 3")
    x1, x2, x3 = x
    y1 = (x1**2) * np.sin(5*x2) + 0.5*x3*np.exp(-0.1*(x1**2 + x2**2))
    y2 = np.where(x1 > 0, x1 + 0.3*x2 - 0.5*x3, 2*x1 - x2 + 0.7*np.abs(x3))
    y3 = (x1**2)/4 + (x2**2)/9 - (x3**2)/16 + 0.2*x1*x2*x3
    y4 = erf(0.5*x1) + np.cos(3*x2 + 0.5*x3)
    y5 = np.where(x3 > 1, np.exp(0.3*(x1 + x2)), np.log1p(np.abs(x1 - x2)))
    r = np.sqrt(x1**2 + x2**2 + x3**2)
    y6 = 2*np.exp(-0.2*r**2) + 0.5*np.exp(-0.05*(r-5)**2)
    y7 = (3.9 * x1 * (1 - x1/5) * np.sin(x2) + 0.1 * np.mod(x3, 2.5))
    threshold = np.sin(x1 + x2) + np.cos(x2 + x3)
    y8 = np.where(threshold > 0.5, 1.5*x1 - 0.8*x2 + 0.3*x3**2, -0.5*x1 + 1.2*np.sqrt(np.abs(x2)) + 0.4*x3)
    return np.array([y1, y2, y3, y4, y5, y6, y7, y8])


class BayesianOptimizer:
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        dimensions: int,
        bounds: Sequence[Sequence[float]],
        n_tasks: int = 1,
        acquisition_type: str = "EI",
        exploration_rate: float = 0.1,
        random_state: int | None = None,
        task_weights: Sequence[float] | None = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.dims = dimensions
        self.bounds = np.asarray(bounds, dtype=float)
        self.acq = acquisition_type.upper()
        self.beta = float(exploration_rate)
        self.rng = check_random_state(random_state)
        self.n_tasks = int(n_tasks)
        self.w = (
            np.array(task_weights, dtype=float) / np.sum(task_weights)
            if task_weights is not None else np.full(self.n_tasks, 1.0 / self.n_tasks)
        )
        self.X: np.ndarray = np.empty((0, self.dims))
        self.Y: np.ndarray = np.empty((0, self.n_tasks))
        self.best_values: List[np.ndarray] = []
        self.iteration = 0
        self._initialize_samples()

    def _initialize_samples(self) -> None:
        while len(self.X) < 5:
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = self._safe_eval(x)
            if np.isfinite(y).all():
                self.X = np.vstack([self.X, x]) if self.X.size else x.reshape(1, -1)
                self.Y = np.vstack([self.Y, y]) if self.Y.size else y
        self.best_values.append(np.max(self.Y, axis=0))

    def _safe_eval(self, x: np.ndarray) -> np.ndarray:
        try:
            y = self.objective_fn(x)
            y_arr = np.asarray(y, dtype=float).reshape(1, -1)
            if y_arr.shape[1] != self.n_tasks:
                raise ValueError(f"Expected {self.n_tasks} outputs, got {y_arr.shape[1]}")
            return y_arr
        except Exception as e:
            print(f"Evaluation failed at {x}: {e}")
            return np.full((1, self.n_tasks), np.nan)

    def _fit_gp(self):
        X_t = _to_tensor(self.X)
        Y_t = _to_tensor(self.Y)
        if self.n_tasks == 1:
            model = SingleTaskGP(
                X_t, Y_t,
                input_transform=Normalize(self.dims),
                outcome_transform=Standardize(m=1),
            )
        else:
            n = X_t.shape[0]
            task_idx = torch.arange(self.n_tasks).repeat_interleave(n)
            X_mt = X_t.repeat(self.n_tasks, 1).clone()
            X_mt = torch.cat([X_mt, task_idx.unsqueeze(1)], dim=1)
            Y_mt = Y_t.t().reshape(-1, 1)
            model = MultiTaskGP(
                X_mt, Y_mt,
                task_feature=X_mt.shape[1] - 1,
                input_transform=Normalize(self.dims + 1),
                outcome_transform=Standardize(m=1),
            )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model

    def _acquisition(self, X_cand: np.ndarray, model) -> np.ndarray:
        X_t = _to_tensor(X_cand)
        posterior = model.posterior(X_t)
        mean = posterior.mean.detach().cpu().numpy().reshape(len(X_cand), -1)
        var = posterior.variance.clamp_min(1e-9).detach().cpu().numpy().reshape(len(X_cand), -1)
        std = np.sqrt(var)
        mean_s = (mean * self.w).sum(1)
        std_s = np.sqrt(((std * self.w) ** 2).sum(1))
        if self.acq == "EI":
            best = self.best_values[-1].dot(self.w) if self.best_values else -np.inf
            imp = mean_s - best - self.beta
            z = imp / std_s
            return imp * norm.cdf(z) + std_s * norm.pdf(z)
        elif self.acq == "UCB":
            return mean_s + self.beta * std_s
        else:
            raise ValueError(f"Unknown acquisition type: {self.acq}")

    def _generate_candidates(self, model) -> np.ndarray:
        rand = Parallel(n_jobs=-1)(
            delayed(lambda: self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1]))() for _ in range(10 * self.dims)
        )
        try:
            res = minimize(
                fun=lambda x: -self._acquisition(x.reshape(1, -1), model)[0],
                x0=self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1]),
                bounds=self.bounds,
                method="L-BFGS-B",
            )
            if res.success:
                optimized = res.x
            else:
                optimized = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        except Exception as e:
            print(f"Optimization failed: {e}")
            optimized = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return np.vstack([rand, optimized])

    def optimize(self, n_iter: int = 20) -> Dict[str, Any]:
        start_time = time.time()
        for _ in range(n_iter):
            self.iteration += 1
            try:
                model = self._fit_gp()
                candidates = self._generate_candidates(model)
                scores = self._acquisition(candidates, model)
                x_next = candidates[np.argmax(scores)]
                y_next = self._safe_eval(x_next)
                if np.isfinite(y_next).all():
                    self.X = np.vstack([self.X, x_next])
                    self.Y = np.vstack([self.Y, y_next])
                    self.best_values.append(np.maximum(self.best_values[-1], y_next.squeeze()))
                print(f"Iter {self.iteration}/{n_iter}: | Best: {self.best_values[-1]} | New: {y_next} | X: {x_next}")
            except Exception as e:
                print(f"Iteration {self.iteration} failed: {e}")
                break

        self.total_time = time.time() - start_time

        return self.history

    @property
    def history(self) -> Dict[str, Any]:
        return {
            "X": self.X,
            "Y": self.Y,
            "best_values": np.vstack(self.best_values),
            "bounds": self.bounds,
            "dims": self.dims,
            "total_time": self.total_time,       
        }


if __name__ == "__main__":
    optimizer = BayesianOptimizer(
        objective_fn=target_function,
        dimensions=3,
        bounds=[(-3, 3)] * 3,
        acquisition_type="EI",
        exploration_rate=0.1,
        random_state=42,
        n_tasks=8,
        task_weights=[1] * 8,
    )
    results = optimizer.optimize(n_iter=20)

    # Print results
    print("Total Time:", results["total_time"])

    import matplotlib.pyplot as plt
    plt.plot(np.max(results["best_values"], axis=1))
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value")
    plt.title("Optimization Progress")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
