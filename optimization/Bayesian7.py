import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import gpytorch

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gpytorch.models import ApproximateGP
from gpytorch.kernels import ScaleKernel, MaternKernel, LinearKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal

from scipy.stats import qmc
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# -----------------------------
# Config
# -----------------------------
@dataclass
class GPConfig:
    # [Tuned] Increased to 2048 to saturate GPU (16GB VRAM) and stabilize gradients
    batch_size: int = 2048

    # Initial fit epochs (first time we build the GP)
    num_epochs_init: int = 300

    # Base update epochs (we add extra epochs as N grows)
    num_epochs_update: int = 50

    # Adam learning rate
    lr: float = 0.02

    # [Tuned] Increased to 2048 for better capacity with 100k target data.
    # 16GB VRAM can handle this.
    max_inducing_points: int = 2048

    # How many data points to use when selecting inducing points (subsample for speed)
    inducing_subset_size: int = 10000

    # [Critical] Keep False to maintain model persistence and stability
    refresh_inducing: bool = False

    # Refresh inducing points every K iterations (only used if refresh_inducing=True)
    inducing_refresh_interval: int = 10

    # Candidate pool size for acquisition
    candidates_pool_size: int = 10000

    # How many new points to add each BO iteration
    acq_batch_size: int = 500

    # [Tuned] Chunk size for GPU prediction during acquisition/evaluation
    acq_eval_batch_size: int = 2048

    # Cap for K_big to limit CPU FPS workload
    K_BIG_CAP: int = 8000

    # Scheduler knobs
    lr_patience: int = 15
    lr_factor: float = 0.5

    # Early stopping for GP training (set 0 to disable)
    early_stop_patience: int = 20

    # Optional cap to avoid extremely long runs
    max_epochs_cap: int = 500


# -----------------------------
# Simple FPS (diversification)
# -----------------------------
def farthest_point_sampling(X: torch.Tensor, m: int) -> torch.Tensor:
    """
    Greedy farthest point sampling (FPS).
    Note: O(N*M). Use on a subsample to avoid huge costs at large N.
    """
    n = X.shape[0]
    if m >= n:
        return X

    # Random start index
    idx0 = torch.randint(low=0, high=n, size=(1,), device=X.device).item()
    idxs = [idx0]

    # Distances from all points to the selected set (initially one point)
    dists = torch.cdist(X, X[idxs], p=2).squeeze(-1)

    for _ in range(1, m):
        farthest = torch.argmax(dists).item()
        idxs.append(farthest)

        # Update distances: min(current_dist, dist_to_new_point)
        new_d = torch.cdist(X, X[[farthest]], p=2).squeeze(-1)
        dists = torch.minimum(dists, new_d)

    return X[torch.tensor(idxs, device=X.device, dtype=torch.long)]


def select_inducing_points(train_x: torch.Tensor, m: int, subset_size: int) -> torch.Tensor:
    """
    Select inducing points on a subsample of train_x to control cost.
    """
    n = train_x.shape[0]
    if n <= subset_size:
        x_sub = train_x
    else:
        # Subsample indices on the same device for speed
        idx = torch.randperm(n, device=train_x.device)[:subset_size]
        x_sub = train_x[idx]

    m_eff = min(m, x_sub.shape[0])
    inducing = farthest_point_sampling(x_sub, m_eff)
    return inducing


# -----------------------------
# Batch SVGP (multi-output via batch_shape)
# -----------------------------
class BatchSVGP(ApproximateGP):
    def __init__(
        self,
        inducing_points_std: torch.Tensor,   # (M, D) in standardized-log space
        num_tasks: int,
        bounds_tensor: torch.Tensor,         # (2, D)
        x_log_mean: torch.Tensor,            # (1, D)
        x_log_std: torch.Tensor,             # (1, D)
    ):
        batch_shape = torch.Size([num_tasks])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points_std.size(0),
            batch_shape=batch_shape,
        )

        # Expand inducing points to batch mode: (T, M, D)
        inducing_batched = inducing_points_std.unsqueeze(0).expand(num_tasks, -1, -1)

        variational_strategy = VariationalStrategy(
            self,
            inducing_batched,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        self.num_tasks = num_tasks

        self.mean_module = ConstantMean(batch_shape=batch_shape)

        # Flexible kernel: Linear + Matern
        self.covar_module = ScaleKernel(
            LinearKernel(ard_num_dims=inducing_points_std.size(-1), batch_shape=batch_shape)
            + MaternKernel(nu=2.5, ard_num_dims=inducing_points_std.size(-1), batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

        # Store transforms as buffers for correct device/dtype handling
        self.register_buffer("bounds_tensor", bounds_tensor)
        self.register_buffer("x_log_mean", x_log_mean)
        self.register_buffer("x_log_std", x_log_std)

    def update_input_transform(self, x_log_mean: torch.Tensor, x_log_std: torch.Tensor):
        """
        Update the standardization stats without rebuilding the GP.
        This is important for stability: keep variational state, only refresh scaling.
        """
        self.x_log_mean.copy_(x_log_mean)
        self.x_log_std.copy_(x_log_std)

    def _transform_inputs(self, x_unit: torch.Tensor) -> torch.Tensor:
        """
        Unit [0,1] -> physical -> log -> standardized.
        """
        x_unit = x_unit.to(self.bounds_tensor)
        lower, upper = self.bounds_tensor[0], self.bounds_tensor[1]
        x_phys = x_unit * (upper - lower) + lower
        x_log = torch.log(x_phys.clamp(min=1e-6))
        x_std = (x_log - self.x_log_mean) / self.x_log_std
        return x_std

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# -----------------------------
# Main Optimizer
# -----------------------------
class BayesianOptimizer:
    def __init__(
        self,
        simulator,
        bounds_list: Sequence[Sequence[float]],
        output_dir: str,
        n_initial_points: int,
        n_batches: int,
        batch_size: int,
        num_outputs: int = 8,
        svgp_threshold: int = 100,  # kept for compatibility
        resume: bool = False,
        target_total: Optional[int] = None,
        device: Optional[torch.device] = None,
        gp_config: Optional[GPConfig] = None,
        test_csv_path: Optional[str] = None,
        **kwargs,
    ):
        self.gp_device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.config = gp_config or GPConfig()

        self.simulator = simulator
        self.physical_bounds = np.asarray(bounds_list, dtype=np.float32)  # (D,2)
        self.dim = int(self.physical_bounds.shape[0])
        self.num_outputs = int(num_outputs)

        self.n_initial_points = int(n_initial_points)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.target_total = target_total
        self.resume = resume

        self.objective_mode = str(kwargs.get("objective_mode", "min")).lower()
        self.objective_index = kwargs.get("objective_index", None)
        self.objective_weights = kwargs.get("objective_weights", None)

        self.train_X = torch.empty((0, self.dim), dtype=self.dtype, device=self.gp_device)
        self.train_Y_raw = torch.empty((0, self.num_outputs), dtype=self.dtype, device=self.gp_device)

        self.test_X = None
        self.test_Y_raw = None

        self.x_log_mean = None
        self.x_log_std = None
        self.y_log_mean = None
        self.y_log_std = None

        self.gp_model: Optional[BatchSVGP] = None
        self.likelihood: Optional[GaussianLikelihood] = None

        self.iteration_counter = 0  # used for optional inducing refresh

        os.makedirs(output_dir, exist_ok=True)
        self.results_csv_path = os.path.join(output_dir, "optimization_results.csv")
        self.val_log_path = os.path.join(output_dir, "validation_log.csv")
        self.model_save_path = os.path.join(output_dir, "batch_svgp.pt")

        self._init_data()
        if test_csv_path:
            self._load_test_set(test_csv_path)

        print(f"[BayesianOptimizer] Device: {self.gp_device} | SVGP batch tasks: {self.num_outputs}")

    # -----------------------------
    # CSV init / resume
    # -----------------------------
    def _init_data(self):
        cols = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{i:02d}" for i in range(1, self.num_outputs + 1)]

        if os.path.exists(self.results_csv_path) and self.resume:
            print("[Resume] Loading existing CSV data...")
            try:
                df = pd.read_csv(self.results_csv_path)
                if not df.empty:
                    X_phys = df[cols[:5]].to_numpy(dtype=np.float32)
                    Y_raw = df[cols[5:]].to_numpy(dtype=np.float32)

                    b = self.physical_bounds
                    X_unit = (X_phys - b[:, 0]) / (b[:, 1] - b[:, 0])

                    self.train_X = torch.from_numpy(X_unit).to(self.gp_device, dtype=self.dtype)
                    self.train_Y_raw = torch.from_numpy(Y_raw).to(self.gp_device, dtype=self.dtype)
                    print(f"  -> Loaded {len(df)} samples.")
            except Exception as e:
                print(f"[Resume] Failed to load CSV: {e}")
        else:
            with open(self.results_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")

        if not os.path.exists(self.val_log_path):
            with open(self.val_log_path, "w", encoding="utf-8") as f:
                f.write("iteration,dataset,mse,mae,max_err,r2\n")

    def _load_test_set(self, path: str):
        if not os.path.exists(path):
            print(f"[Validation] Test CSV not found: {path}")
            return

        print(f"[Validation] Loading test set: {path}")
        df = pd.read_csv(path)

        cols = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{i:02d}" for i in range(1, self.num_outputs + 1)]
        df = df.dropna(subset=cols)

        X_phys = df[cols[:5]].to_numpy(dtype=np.float32)
        Y_raw = df[cols[5:]].to_numpy(dtype=np.float32)

        b = self.physical_bounds
        X_unit = (X_phys - b[:, 0]) / (b[:, 1] - b[:, 0])

        self.test_X = torch.from_numpy(X_unit).to(self.gp_device, dtype=self.dtype)
        self.test_Y_raw = torch.from_numpy(Y_raw).to(self.gp_device, dtype=self.dtype)

    def _save_row(self, x_phys: np.ndarray, y_vals: np.ndarray):
        row = np.concatenate([x_phys, y_vals])
        with open(self.results_csv_path, "a", encoding="utf-8") as f:
            f.write(",".join([f"{v:.8f}" for v in row]) + "\n")

    def _save_val_metrics(self, iter_idx: int, dataset: str, df: pd.DataFrame):
        with open(self.val_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{iter_idx},{dataset},{df['MSE'].mean():.6f},{df['MAE'].mean():.6f},"
                f"{df['MaxErr'].max():.6f},{df['R2'].mean():.4f}\n"
            )

    # -----------------------------
    # Simulation wrapper
    # -----------------------------
    def run_simulation(self, params) -> Optional[np.ndarray]:
        if isinstance(params, torch.Tensor):
            x_unit = params.detach().cpu().numpy().flatten()
        else:
            x_unit = np.asarray(params).flatten()

        b = self.physical_bounds
        x_phys = b[:, 0] + x_unit * (b[:, 1] - b[:, 0])

        try:
            self.simulator.configure_geometry(float(x_phys[3]), float(x_phys[4]))
            disp = self.simulator.run_simulation(float(x_phys[0]), float(x_phys[1]), float(x_phys[2]))
            if disp is None:
                return None

            disp = np.array(disp, dtype=np.float32).flatten()
            if len(disp) < self.num_outputs:
                disp = np.pad(disp, (0, self.num_outputs - len(disp)))
            else:
                disp = disp[: self.num_outputs]
            return disp
        except Exception:
            return None

    def _scaled_to_original(self, x_unit: torch.Tensor) -> np.ndarray:
        """Unit [0,1] -> physical bounds."""
        b = self.physical_bounds
        x = x_unit.detach().cpu().numpy().flatten()
        return x * (b[:, 1] - b[:, 0]) + b[:, 0]

    # -----------------------------
    # Data transforms
    # -----------------------------
    def _update_stats(self):
        bounds_t = torch.tensor(self.physical_bounds, device=self.gp_device, dtype=self.dtype).T  # (2,D)

        X_phys = self.train_X * (bounds_t[1] - bounds_t[0]) + bounds_t[0]
        X_log = torch.log(X_phys.clamp(min=1e-6))
        self.x_log_mean = X_log.mean(dim=0, keepdim=True)
        self.x_log_std = X_log.std(dim=0, keepdim=True).clamp_min(1e-6)

        Y_log = torch.log(self.train_Y_raw + 1e-6)
        self.y_log_mean = Y_log.mean(dim=0, keepdim=True)
        self.y_log_std = Y_log.std(dim=0, keepdim=True).clamp_min(1e-6)

    def _get_train_data_std(self):
        bounds_t = torch.tensor(self.physical_bounds, device=self.gp_device, dtype=self.dtype).T

        X_phys = self.train_X * (bounds_t[1] - bounds_t[0]) + bounds_t[0]
        X_log = torch.log(X_phys.clamp(min=1e-6))
        X_std = (X_log - self.x_log_mean) / self.x_log_std

        Y_log = torch.log(self.train_Y_raw + 1e-6)
        Y_std = (Y_log - self.y_log_mean) / self.y_log_std

        return X_std, Y_std.T  # X:(N,D), Y_T:(T,N)

    # -----------------------------
    # GP init/update helpers
    # -----------------------------
    def _maybe_build_gp(self, train_x_std: torch.Tensor):
        """
        Build the GP only once (or when forcing inducing refresh).
        Keeping the same model preserves variational parameters and stabilizes training.
        """
        need_build = self.gp_model is None

        if (not need_build) and self.config.refresh_inducing:
            if self.iteration_counter % max(1, self.config.inducing_refresh_interval) == 0:
                need_build = True

        if not need_build:
            # Update input transform buffers only
            self.gp_model.update_input_transform(self.x_log_mean, self.x_log_std)
            return

        # Select inducing points on a subsample for speed
        n_data = train_x_std.shape[0]
        m = min(self.config.max_inducing_points, n_data)
        inducing = select_inducing_points(train_x_std, m, self.config.inducing_subset_size)

        bounds_t = torch.tensor(self.physical_bounds, device=self.gp_device, dtype=self.dtype).T
        model = BatchSVGP(
            inducing_points_std=inducing,
            num_tasks=self.num_outputs,
            bounds_tensor=bounds_t,
            x_log_mean=self.x_log_mean,
            x_log_std=self.x_log_std,
        ).to(self.gp_device, self.dtype)

        likelihood = GaussianLikelihood(batch_shape=torch.Size([self.num_outputs])).to(self.gp_device, self.dtype)

        # Warm start if we are rebuilding (optional inducing refresh)
        if self.gp_model is not None and self.likelihood is not None:
            old_state = self.gp_model.state_dict()
            new_state = model.state_dict()

            # We avoid copying variational/inducing states if shapes differ.
            # If you keep the same M and do not refresh inducing, you should not rebuild at all.
            filtered = {}
            for k, v in old_state.items():
                if k not in new_state:
                    continue
                if v.shape != new_state[k].shape:
                    continue
                # Copy everything that matches, INCLUDING variational params when shapes match
                filtered[k] = v

            new_state.update(filtered)
            model.load_state_dict(new_state, strict=False)
            try:
                likelihood.load_state_dict(self.likelihood.state_dict())
            except Exception:
                pass

        self.gp_model = model
        self.likelihood = likelihood

    # -----------------------------
    # Train SVGP
    # -----------------------------
    def fit_gp_model(self):
        self._update_stats()
        train_x, train_y_T = self._get_train_data_std()
        n_data = train_x.shape[0]

        # Build once (or refresh inducing if configured)
        self._maybe_build_gp(train_x)

        assert self.gp_model is not None and self.likelihood is not None

        # Dynamic epochs: add more as N grows, but cap to keep runtime reasonable
        if self.iteration_counter == 0:
            epochs = self.config.num_epochs_init
        else:
            base = self.config.num_epochs_update
            extra = int(n_data / 200)  # gentler growth than /100 (tune as needed)
            epochs = min(base + extra, self.config.max_epochs_cap)

        # Logging
        M = self.gp_model.variational_strategy.inducing_points.size(-2)
        print(f"[GP] Training SVGP: N={n_data}, M={M}, Epochs={epochs}")

        self.gp_model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [{"params": self.gp_model.parameters()}, {"params": self.likelihood.parameters()}],
            lr=self.config.lr,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.config.lr_factor, patience=self.config.lr_patience
        )

        mll = VariationalELBO(self.likelihood, self.gp_model, num_data=n_data)

        # DataLoader
        dataset = TensorDataset(train_x, train_y_T.T)  # (N,D), (N,T)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            # [Tuned] pin_memory is not needed as data is already on GPU
            pin_memory=False,
        )

        t0 = time.time()
        best_loss = float("inf")
        bad_epochs = 0
        final_loss = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for xb, yb in loader:
                # xb, yb are already on GPU because dataset tensors are on GPU.
                optimizer.zero_grad(set_to_none=True)
                out = self.gp_model(xb)
                loss = -mll(out, yb.T).sum()  # sum over tasks
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(1, batch_count)
            scheduler.step(avg_loss)
            final_loss = avg_loss

            # Simple early stopping
            if self.config.early_stop_patience > 0:
                if avg_loss + 1e-6 < best_loss:
                    best_loss = avg_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.config.early_stop_patience:
                        break

        print(f"[GP] Done in {time.time() - t0:.2f}s | Final Loss: {final_loss:.4f}")

        # [Tuned] Clear cache to free up VRAM for large batch acquisition
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.gp_model.eval()
        self.likelihood.eval()

    # -----------------------------
    # Evaluate helper
    # -----------------------------
    def evaluate_model(self, X_unit: torch.Tensor, Y_true_raw: torch.Tensor, dataset_name: str = "Vali"):
        if self.gp_model is None or self.likelihood is None or len(X_unit) == 0:
            return

        self.gp_model.eval()
        self.likelihood.eval()

        eval_bs = int(self.config.acq_eval_batch_size)
        Y_pred_list = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X_unit), eval_bs):
                batch_x = X_unit[i: i + eval_bs]
                X_std = self.gp_model._transform_inputs(batch_x)

                pred = self.likelihood(self.gp_model(X_std))
                mean_log_std = pred.mean.T  # (T,B)->(B,T)

                Y_log = mean_log_std * self.y_log_std + self.y_log_mean
                Y_pred_batch = torch.exp(Y_log) - 1e-6
                Y_pred_list.append(Y_pred_batch)

        Y_pred = torch.cat(Y_pred_list, dim=0)

        yt = Y_true_raw.detach().cpu().numpy()
        yp = Y_pred.detach().cpu().numpy()

        rows = []
        for i in range(self.num_outputs):
            r2 = r2_score(yt[:, i], yp[:, i]) if np.var(yt[:, i]) > 1e-9 else 0.0
            rows.append(
                {
                    "Target": f"x_{i+1:02d}",
                    "R2": r2,
                    "MSE": mean_squared_error(yt[:, i], yp[:, i]),
                    "MAE": mean_absolute_error(yt[:, i], yp[:, i]),
                    "MaxErr": float(np.max(np.abs(yt[:, i] - yp[:, i]))),
                }
            )

        df = pd.DataFrame(rows)
        print(f"\n--- {dataset_name} Performance ---")
        print(df.to_string(index=False, float_format="%.4f"))

        mean_r2 = df["R2"].mean()
        if dataset_name == "Train_Set" and mean_r2 < 0.85:
            print(f"⚠️ [Warning] Potential underfitting: Train mean R2={mean_r2:.4f}")
            print("   -> Consider: more epochs, larger M, or check data noise/quality.")

        self._save_val_metrics(len(self.train_X), dataset_name, df)

    # -----------------------------
    # Objective -> scalar best_value
    # -----------------------------
    def _compute_objective(self, Y_raw: torch.Tensor) -> torch.Tensor:
        if Y_raw is None or Y_raw.numel() == 0:
            return torch.empty((0,), device=self.gp_device, dtype=self.dtype)

        if self.objective_weights is not None:
            w = torch.tensor(self.objective_weights, device=Y_raw.device, dtype=Y_raw.dtype).view(1, -1)
            obj = (Y_raw * w).sum(dim=1)
        elif self.objective_index is not None:
            obj = Y_raw[:, int(self.objective_index)]
        else:
            obj = Y_raw.sum(dim=1)

        return obj

    # -----------------------------
    # Main loop
    # -----------------------------
    def optimize(self):
        assert self.target_total is not None, "target_total must be provided (e.g., 100000)"
        print("[BayesianOptimizer] Starting optimization...")

        # 1) Initial LHS
        if self.train_X.shape[0] < self.n_initial_points:
            print(f"[Init] Collecting {self.n_initial_points} LHS points...")
            sampler = qmc.LatinHypercube(d=self.dim)
            samples = sampler.random(n=self.n_initial_points - self.train_X.shape[0])

            for s in samples:
                x_u = torch.tensor(s, dtype=self.dtype, device=self.gp_device)
                disp = self.run_simulation(x_u)
                if disp is not None:
                    self.train_X = torch.cat([self.train_X, x_u.unsqueeze(0)])
                    self.train_Y_raw = torch.cat(
                        [self.train_Y_raw, torch.tensor(disp, device=self.gp_device, dtype=self.dtype).unsqueeze(0)]
                    )
                    self._save_row(self._scaled_to_original(x_u), disp)

        # 2) Active learning loop
        while len(self.train_X) < self.target_total:
            self.iteration_counter += 1
            print(f"\n=== Iteration: {len(self.train_X)} samples ===")

            self.fit_gp_model()

            self.evaluate_model(self.train_X, self.train_Y_raw, "Train_Set")
            if self.test_X is not None:
                self.evaluate_model(self.test_X, self.test_Y_raw, "Test_Set")

            # ------------------------------------------------------
            # 3) Acquisition: CPU candidates + GPU chunked pred + CPU TopK + CPU FPS
            # ------------------------------------------------------
            print("[Acquisition] Scanning candidate pool (chunked + CPU FPS)...")

            sampler = qmc.LatinHypercube(d=self.dim)
            cand_unit_cpu = torch.tensor(
                sampler.random(n=self.config.candidates_pool_size),
                dtype=self.dtype,
                device="cpu",
            )

            eval_bs = int(self.config.acq_eval_batch_size)
            unc_list = []

            assert self.gp_model is not None and self.likelihood is not None
            self.gp_model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for i in range(0, self.config.candidates_pool_size, eval_bs):
                    batch_cand = cand_unit_cpu[i: i + eval_bs].to(self.gp_device, non_blocking=True)
                    batch_cand_std = self.gp_model._transform_inputs(batch_cand)
                    pred = self.likelihood(self.gp_model(batch_cand_std))

                    # Uncertainty score: sum variances across tasks -> shape (B,)
                    batch_unc = pred.variance.sum(dim=0).detach().cpu()
                    unc_list.append(batch_unc)

            unc = torch.cat(unc_list, dim=0)  # CPU tensor (N_cand,)

            batch_k = min(self.config.acq_batch_size, self.target_total - len(self.train_X))

            # Limit K_big to avoid heavy CPU work during FPS
            K_big = int(min(max(5000, 20 * batch_k), self.config.K_BIG_CAP, self.config.candidates_pool_size))

            _, idxs_big = torch.topk(unc, K_big)  # CPU
            cand_big_cpu = cand_unit_cpu[idxs_big]  # CPU

            # FPS on CPU for diversification
            batch_X_cpu = farthest_point_sampling(cand_big_cpu, batch_k)
            batch_X = batch_X_cpu.to(self.gp_device)

            print(f"[Acquisition] Selected {len(batch_X)} points (top{K_big} -> fps{batch_k}).")

            # 4) Run simulations
            new_cnt = 0
            for x_u in batch_X:
                disp = self.run_simulation(x_u)
                if disp is not None:
                    self.train_X = torch.cat([self.train_X, x_u.unsqueeze(0)])
                    self.train_Y_raw = torch.cat(
                        [self.train_Y_raw, torch.tensor(disp, device=self.gp_device, dtype=self.dtype).unsqueeze(0)]
                    )
                    self._save_row(self._scaled_to_original(x_u), disp)
                    new_cnt += 1

            if new_cnt == 0:
                print("[Stop] No valid simulations returned in this batch.")
                break

            # Save checkpoint
            try:
                torch.save(
                    {"model": self.gp_model.state_dict(), "likelihood": self.likelihood.state_dict()},
                    self.model_save_path,
                )
            except Exception:
                pass

        print("[Done] Optimization finished.")

        if self.train_X is None or self.train_X.shape[0] == 0:
            return None, None

        obj = self._compute_objective(self.train_Y_raw)
        if obj.numel() == 0:
            return None, None

        if self.objective_mode == "max":
            best_idx = int(torch.argmax(obj).item())
        else:
            best_idx = int(torch.argmin(obj).item())

        best_x_unit = self.train_X[best_idx]
        best_params = self._scaled_to_original(best_x_unit)
        best_value = float(obj[best_idx].detach().cpu().item())

        return best_params, best_value