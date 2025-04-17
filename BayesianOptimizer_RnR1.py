
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, Sum
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class BayesianOptimizer:
    """
    High‑performance Bayesian Optimization implementation that
    currently relies on the full Gaussian Process Regressor only.
    If you want to switch to a sparse approximation in the future,
    add the logic back into `_dynamic_gp` and create an appropriate
    `_init_sparse_gp` method.
    """
    def __init__(
        self,
        dimensions,
        bounds,
        noise_level=0.1,
        acquisition_type="EI",
        exploration_rate=0.1,
        random_state=None,
        kernel=None,
    ):
        self._validate_inputs(dimensions, bounds)
        self.dims = dimensions
        self.bounds = np.array(bounds, dtype=float)
        self.exploration_rate = exploration_rate
        self.acquisition_type = acquisition_type
        self.random_state = check_random_state(random_state)

        self._init_gp_components(noise_level, kernel)

        self.X = np.empty((0, dimensions))
        self.Y = np.empty(0)
        self.best_values = []
        self.iteration = 0

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, dimensions, bounds):
        if len(bounds) != dimensions:
            raise ValueError("Dimensions/bounds mismatch")
        for i, (low, high) in enumerate(bounds):
            if low >= high:
                raise ValueError(
                    f"Invalid bounds at index {i}: must be (min, max) with min < max"
                )

    def _init_gp_components(self, noise_level, kernel):
        if kernel is None:
            self.base_kernel = Matern(
                nu=0.5,
                length_scale_bounds=(1e-5, 1e3),
            )
        else:
            self.base_kernel = kernel

        # Always include an additive noise term
        self.noise_kernel = (
            WhiteKernel(noise_level=noise_level ** 2, noise_level_bounds=(1e-10, 1e2))
            if not isinstance(self.base_kernel, Sum)
            else None
        )

        full_kernel = (
            self.base_kernel + self.noise_kernel if self.noise_kernel else self.base_kernel
        )

        self.full_gp = GaussianProcessRegressor(
            kernel=full_kernel,
            n_restarts_optimizer=5,
            optimizer="fmin_l_bfgs_b",
            normalize_y=True,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def optimization_history(self):
        return {
            "X": self.X,
            "Y": self.Y,
            "best_values": self.best_values,
            "bounds": self.bounds,
            "dimensions": self.dims,
            "iteration": self.iteration,
        }

    def get_current_gp(self):
        """Return the (fitted) GP model from the last iteration."""
        return self.full_gp

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _dynamic_gp(self):
        """For future extensibility. Currently always returns full GP."""
        return self.full_gp

    def _acquisition(self, X, gp=None):
        gp = self.full_gp if gp is None else gp
        X = np.atleast_2d(X)

        try:
            mu, sigma = gp.predict(X, return_std=True)
        except Exception:
            mu, sigma = np.zeros(len(X)), np.ones(len(X))

        sigma = np.clip(sigma, 1e-6, None)

        if self.Y.size == 0:
            return np.zeros_like(mu)

        current_max = np.max(self.Y)

        if self.acquisition_type == "EI":
            improvement = mu - current_max - self.exploration_rate
            z = improvement / sigma
            return improvement * norm.cdf(z) + sigma * norm.pdf(z)
        elif self.acquisition_type == "UCB":
            return mu + self.exploration_rate * sigma
        else:
            raise ValueError(f"Unknown acquisition type: {self.acquisition_type}")

    # Candidate generation ------------------------------------------------
    def _generate_candidates(self):
        random_candidates = Parallel(n_jobs=-1)(
            delayed(self._random_candidate)() for _ in range(10 * self.dims)
        )
        optimized_candidates = [self._optimized_candidate()]
        candidates = np.vstack([random_candidates, optimized_candidates])
        return np.unique(candidates, axis=0)

    def _random_candidate(self):
        return self.random_state.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _optimized_candidate(self):
        try:
            res = minimize(
                fun=lambda x: -self._acquisition(x.reshape(1, -1)).item(),
                x0=self._random_candidate(),
                bounds=self.bounds,
                method="L-BFGS-B",
                options={"maxiter": 50},
            )
            return res.x if res.success else self._random_candidate()
        except Exception:
            return self._random_candidate()

    # Main optimization loop ---------------------------------------------
    def optimize(self, objective_fn, n_iter):
        if self.X.size == 0:
            self._initial_sample(objective_fn)

        for _ in range(n_iter):
            print(f"Iteration {self.iteration + 1}/{n_iter}")
            self.iteration += 1

            try:
                current_gp = self._dynamic_gp()
                current_gp.fit(self.X, self.Y)

                candidates = self._generate_candidates()
                scores = self._acquisition(candidates, current_gp)
                x_next = candidates[np.nanargmax(scores)]

                y_next = self._safe_evaluate(objective_fn, x_next)

                if np.isfinite(y_next):
                    self.X = np.vstack([self.X, x_next])
                    self.Y = np.append(self.Y, y_next)
                    self.best_values.append(
                        max(self.best_values[-1], y_next) if self.best_values else y_next
                    )
            except Exception as e:
                print(f"Iteration {self.iteration} failed: {e}")
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initial_sample(self, objective_fn):
        x_init = self._random_candidate()
        y_init = self._safe_evaluate(objective_fn, x_init)
        if not np.isfinite(y_init):
            raise RuntimeError("Initial evaluation returned non‑finite value")

        self.X = np.atleast_2d(x_init)
        self.Y = np.array([y_init])
        self.best_values.append(y_init)

    def _safe_evaluate(self, objective_fn, x):
        try:
            return float(objective_fn(x))
        except Exception as e:
            print(f"Evaluation failed at {x}: {e}")
            return np.nan


# ----------------------------------------------------------------------
# Visualization utilities (unchanged)
# ----------------------------------------------------------------------
class BayesianOptimizationVisualizer:
    @staticmethod
    def plot_optimization_progress(best_values):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(best_values)),
                y=best_values,
                mode="lines+markers",
                name="Best Observed Value",
            )
        )
        fig.update_layout(
            title="Optimization Progress",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            template="plotly_white",
            hovermode="x unified",
        )
        fig.show()

    @staticmethod
    def plot_model_vs_true(optimizer_history, true_function, gp_model, resolution=15):
        if optimizer_history["dimensions"] != 3:
            print("3D visualization requires 3D problem")
            return

        resolution = max(10, resolution - optimizer_history["iteration"] // 5)
        axes = [np.linspace(b[0], b[1], resolution) for b in optimizer_history["bounds"]]
        grid = np.meshgrid(*axes)
        X_test = np.vstack([g.ravel() for g in grid]).T

        mu, _ = gp_model.predict(X_test, return_std=True)
        true_vals = np.array([true_function(x) for x in X_test])
        errors = np.abs(true_vals - mu)

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}] * 3],
            subplot_titles=("True Function", "Model Predictions", "Error Map"),
        )

        for vals, cmap, cb_pos, name, col in [
            (true_vals, "Viridis", 0.3, "True", 1),
            (mu, "Viridis", 0.6, "Model", 2),
            (errors, "RdBu", 1.0, "Error", 3),
        ]:
            fig.add_trace(
                go.Scatter3d(
                    x=X_test[:, 0],
                    y=X_test[:, 1],
                    z=X_test[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=vals,
                        colorscale=cmap,
                        opacity=0.7,
                        colorbar=dict(x=cb_pos),
                    ),
                    name=name,
                ),
                row=1,
                col=col,
            )

        fig.update_layout(
            title_text="Model Performance Analysis",
            width=1400,
            height=500,
            margin=dict(r=100),
        )
        fig.show()


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    def sphere(x):
        return np.sin(x[0]) * x[1] + 0.5 * np.sqrt(abs(x[2]))

    def noisy_sphere(x):
        return sphere(x) + np.random.normal(0, 0.1)

    opt = BayesianOptimizer(
        dimensions=3,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        noise_level=0.1,
        acquisition_type="EI",
        exploration_rate=0.3,
        kernel=Matern(nu=1.5) + WhiteKernel(),
        random_state=42,
    )

    opt.optimize(noisy_sphere, n_iter=20)

    vis = BayesianOptimizationVisualizer()
    vis.plot_optimization_progress(opt.best_values)
    vis.plot_model_vs_true(
        optimizer_history=opt.optimization_history,
        true_function=sphere,
        gp_model=opt.get_current_gp(),
    )
