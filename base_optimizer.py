import numpy as np
import torch
from typing import Callable, Sequence, List, Dict, Any
from sklearn.utils import check_random_state
from botorch.models.transforms import Normalize, Standardize

class BaseBayesianOptimizer:
    def __init__(self, objective_fn: Callable[[np.ndarray], np.ndarray],
                 dimensions: int,
                 bounds: Sequence[Sequence[float]],
                 n_tasks: int = 1,
                 acquisition_type: str = "EI",
                 exploration_rate: float = 0.1,
                 task_weights: Sequence[float] | None = None,
                 random_state: int | None = None):

        self.objective_fn = objective_fn
        self.dims = dimensions
        self.bounds = np.array(bounds, dtype=float)
        self.n_tasks = int(n_tasks)
        self.acq = acquisition_type.upper()
        self.beta = exploration_rate
        self.rng = check_random_state(random_state)

        self.task_weights = (np.array(task_weights, dtype=float) / np.sum(task_weights))             if task_weights is not None else np.ones(n_tasks) / n_tasks

        self.input_transform = Normalize(d=self.dims, bounds=torch.tensor(self.bounds.T, dtype=torch.double))
        self.output_transform = Standardize(m=1)

        self.X = np.empty((0, self.dims))
        self.Y = np.empty((0, self.n_tasks))
        self.best_values: List[np.ndarray] = []
        self._initialize_samples()

    def _initialize_samples(self):
        while len(self.X) < 5:
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = self._safe_eval(x)
            if np.isfinite(y).all():
                self.X = np.vstack([self.X, x]) if self.X.size else x.reshape(1, -1)
                self.Y = np.vstack([self.Y, y]) if self.Y.size else y
        self.best_values.append(np.max(self.Y, axis=0))

    def _safe_eval(self, x: np.ndarray) -> np.ndarray:
        try:
            y = np.asarray(self.objective_fn(x), dtype=float).reshape(1, -1)
            if y.shape[1] != self.n_tasks:
                raise ValueError(f"Expected {self.n_tasks} outputs, got {y.shape[1]}")
            return y
        except Exception as e:
            print(f"Evaluation failed at {x}: {e}")
            return np.full((1, self.n_tasks), np.nan)

    def optimize(self, n_iter=20):
        for iteration in range(1, n_iter+1):
            try:
                model = self._fit_model()
                candidate = self._suggest_next(model)
                y_new = self._safe_eval(candidate)
                if np.isfinite(y_new).all():
                    self.X = np.vstack([self.X, candidate])
                    self.Y = np.vstack([self.Y, y_new])
                    new_best = np.maximum(self.best_values[-1], y_new.squeeze())
                    self.best_values.append(new_best)
                print(f"Iter {iteration}: Best = {self.best_values[-1]} | New = {y_new.squeeze()}")
            except Exception as e:
                print(f"Iteration {iteration} failed: {e}")
                break
        return self.history

    @property
    def history(self) -> Dict[str, Any]:
        return {
            "X": self.X,
            "Y": self.Y,
            "best_values": np.vstack(self.best_values),
            "bounds": self.bounds,
            "dims": self.dims,
        }
