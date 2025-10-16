import os
import torch
import numpy as np
import taichi as ti
from typing import List, Tuple, Optional, Union

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

# Initialize Taichi to use GPU if available
ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

class BayesianOptimizer:
    """
    This class handles the Bayesian Optimization process.
    It trains two types of models:
    1. A simple model (`gp_scalar`) to guide the optimization search.
    2. A detailed model (`gp_vector`) to predict all 8 outputs for analysis.
    """

    def __init__(
        self,
        simulator,
        bounds_list: List[List[float]],
        output_dir: str,
        n_initial_points: int,
        n_batches: int,
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        # Set up device (CPU or GPU)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Store main settings
        self.simulator = simulator
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.n_initial_points = n_initial_points
        self.n_batches = n_batches
        self.batch_size = batch_size

        # Define parameter bounds and dimensions
        self.bounds = torch.tensor(bounds_list, dtype=torch.float64, device=self.device).t()
        self.dim = int(self.bounds.shape[1])
        self.output_dim = 8

        # Prepare empty tensors to store training data
        self.train_X = torch.empty((0, self.dim), dtype=torch.float64, device=self.device)
        self.train_Y_scalar = torch.empty((0, 1), dtype=torch.float64, device=self.device)
        self.train_Y_full = torch.empty((0, self.output_dim), dtype=torch.float64, device=self.device)

        # Prepare empty lists for record-keeping
        self.original_X: List[np.ndarray] = []
        self.displacements_list: List[np.ndarray] = []

        # Define the path for the results file
        self.results_file = os.path.join(output_dir, "optimization_results.csv")
        
        # Initialize model placeholders
        self.gp_scalar: Optional[SingleTaskGP] = None
        self.gp_outputs: List[SingleTaskGP] = []
        self.gp_vector: Optional[ModelListGP] = None

    # ---------------------- I/O helpers ----------------------
    def _init_results_file(self) -> None:
        """Creates a new, empty CSV file with a header."""
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_0{i+1}" for i in range(self.output_dim)]
        with open(self.results_file, "w") as f:
            f.write(",".join(headers) + "\n")

    def _save_iteration_data(self, params_numpy: np.ndarray, displacements_numpy: np.ndarray) -> None:
        """Appends one row of data to the CSV file."""
        row = params_numpy.tolist() + displacements_numpy.tolist()
        with open(self.results_file, "a") as f:
            f.write(",".join([f"{v:.16f}" for v in row]) + "\n")

    # ---------------------- Initial sampling ----------------------
    def collect_initial_points(self) -> torch.Tensor:
        """Creates the initial set of points to test."""
        from botorch.utils.sampling import draw_sobol_samples
        unit_bounds = torch.stack(
            [
                torch.zeros(self.dim, dtype=torch.float64, device=self.device),
                torch.ones(self.dim, dtype=torch.float64, device=self.device),
            ]
        )
        X = draw_sobol_samples(bounds=unit_bounds, n=1, q=self.n_initial_points).squeeze(0)
        return X.to(dtype=torch.float64, device=self.device)

    # ---------------------- Simulation wrapper ----------------------
    def run_simulation(self, x_orig_numpy: np.ndarray) -> np.ndarray:
        """
        --- THIS IS THE FIX ---
        Runs the simulation by first configuring geometry and then the physics.
        """
        # 1. Unpack ALL five parameters from the array
        n, eta, sigma_y, width, height = x_orig_numpy

        # 2. Call the geometry configuration method with width and height
        self.simulator.configure_geometry(width, height)

        # 3. Call the physics simulation method with the other parameters
        displacements = self.simulator.run_simulation(n, eta, sigma_y)

        # Handle potential simulation failures gracefully
        if displacements is None or len(displacements) == 0 or np.isnan(displacements).any():
            return np.zeros(self.output_dim)
        if len(displacements) < self.output_dim:
            return np.concatenate([displacements, np.zeros(self.output_dim - len(displacements))])
        
        return np.array(displacements[:self.output_dim])


    # ---------------------- GP fitting ----------------------
    def fit_gp_scalar(self) -> SingleTaskGP:
        """Trains the simple model for the average displacement."""
        if self.train_X.shape[0] == 0:
            raise RuntimeError("No training data for gp_scalar.")
        gp = SingleTaskGP(self.train_X, self.train_Y_scalar)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.gp_scalar = gp
        return gp

    def fit_gp_vector(self) -> ModelListGP:
        """Trains 8 separate models, one for each displacement output."""
        if self.train_X.shape[0] == 0:
            raise RuntimeError("No training data for gp_vector.")
        outputs: List[SingleTaskGP] = []
        for j in range(self.output_dim):
            yj = self.train_Y_full[:, j:j+1]
            gp_j = SingleTaskGP(self.train_X, yj)
            mll_j = ExactMarginalLogLikelihood(gp_j.likelihood, gp_j)
            fit_gpytorch_mll(mll_j)
            outputs.append(gp_j)
        self.gp_outputs = outputs
        self.gp_vector = ModelListGP(*outputs)
        return self.gp_vector

    # ---------------------- Acquisition optimization ----------------------
    def optimize_acquisition_function(self, gp: SingleTaskGP) -> torch.Tensor:
        """Finds the best next points to evaluate."""
        if self.train_Y_scalar.numel() == 0:
            raise RuntimeError("Cannot form acquisition without observations.")
        best_Y = self.train_Y_scalar.max()
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        qEI = qLogExpectedImprovement(model=gp, best_f=best_Y, sampler=qmc_sampler)
        unit_bounds = torch.stack(
            [
                torch.zeros(self.dim, dtype=torch.float64, device=self.device),
                torch.ones(self.dim, dtype=torch.float64, device=self.device),
            ]
        )
        candidates, _ = optimize_acqf(
            acq_function=qEI,
            bounds=unit_bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates.detach()

    # ---------------------- Best sample (by scalar objective) ----------------------
    def return_best_result(self) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the best parameters based on the lowest average displacement."""
        if len(self.displacements_list) == 0:
            raise RuntimeError("No evaluated points yet.")
        vals = np.array([np.mean(d) for d in self.displacements_list])
        idx = int(np.argmin(vals))
        return self.original_X[idx], self.displacements_list[idx]

    # ---------------------- Prediction APIs ----------------------
    @torch.no_grad()
    def predict_scalar(self, X_orig_numpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the average displacement for new inputs."""
        if self.gp_scalar is None:
            raise RuntimeError("gp_scalar is not trained.")
        X = torch.as_tensor(X_orig_numpy, dtype=torch.float64, device=self.device)
        X = X if X.ndim == 2 else X.unsqueeze(0)
        Xn = normalize(X, bounds=self.bounds)
        post = self.gp_scalar.posterior(Xn)
        mu = post.mean.squeeze(-1).cpu().numpy()
        sd = post.variance.clamp_min(1e-18).sqrt().cpu().numpy()
        return mu, sd

    @torch.no_grad()
    def predict_vector(self, X_orig_numpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts all 8 displacements for new inputs."""
        if not self.gp_outputs:
            raise RuntimeError("gp_outputs are not trained.")
        X = torch.as_tensor(X_orig_numpy, dtype=torch.float64, device=self.device)
        X = X if X.ndim == 2 else X.unsqueeze(0)
        Xn = normalize(X, bounds=self.bounds)
        mus, sds = [], []
        for gp in self.gp_outputs:
            post = gp.posterior(Xn)
            mus.append(post.mean.squeeze(-1))
            sds.append(post.variance.clamp_min(1e-18).sqrt().squeeze(-1))
        mu = torch.stack(mus, dim=1).cpu().numpy()
        sd = torch.stack(sds, dim=1).cpu().numpy()
        return mu, sd

    # ---------------------- Save / Load ----------------------
    def save_models(self, path: str) -> None:
        """Saves the trained models to a .pt file."""
        n_train = int(self.train_X.shape[0])
        ckpt = {
            "bounds": self.bounds.detach().cpu(),
            "dim": self.dim,
            "output_dim": self.output_dim,
            "n_train": n_train,
            "gp_scalar": self.gp_scalar.state_dict() if self.gp_scalar is not None else None,
            "gp_outputs": [gp.state_dict() for gp in self.gp_outputs] if self.gp_outputs else None,
        }
        torch.save(ckpt, path)
        print(f"[GP models saved] -> {path}")

    def load_models(self, path: str, map_location: Union[str, torch.device] = "cpu") -> None:
        """Loads trained models from a .pt file."""
        ckpt = torch.load(path, map_location=map_location)
        self.bounds = ckpt["bounds"].to(self.device, dtype=torch.float64)
        self.dim = int(ckpt["dim"])
        self.output_dim = int(ckpt["output_dim"])
        n_train = int(ckpt.get("n_train", 1))
        if ckpt.get("gp_scalar") is not None:
            X_placeholder = torch.zeros((max(n_train, 1), self.dim), dtype=torch.float64, device=self.device)
            Y_placeholder = torch.zeros((max(n_train, 1), 1), dtype=torch.float64, device=self.device)
            self.gp_scalar = SingleTaskGP(X_placeholder, Y_placeholder)
            self.gp_scalar.load_state_dict(ckpt["gp_scalar"])
        if ckpt.get("gp_outputs") is not None:
            self.gp_outputs = []
            for state in ckpt["gp_outputs"]:
                Xp = torch.zeros((max(n_train, 1), self.dim), dtype=torch.float64, device=self.device)
                Yp = torch.zeros((max(n_train, 1), 1), dtype=torch.float64, device=self.device)
                gp = SingleTaskGP(Xp, Yp)
                gp.load_state_dict(state)
                self.gp_outputs.append(gp)
            self.gp_vector = ModelListGP(*self.gp_outputs)
        print(f"[GP models loaded] <- {path}")

    # ---------------------- Main BO loop ----------------------
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the entire optimization process."""
        # Create a new results file only when a new optimization is started.
        self._init_results_file()

        print("Starting Bayesian Optimization...")
        
        # 1. Evaluate initial points
        initial_X_scaled = self.collect_initial_points()
        for x_scaled in initial_X_scaled:
            x_orig_numpy = unnormalize(x_scaled, bounds=self.bounds).cpu().numpy()
            displacements = self.run_simulation(x_orig_numpy)
            avg_displacement = float(np.mean(displacements))
            self.train_X = torch.cat([self.train_X, x_scaled.unsqueeze(0)], dim=0)
            self.train_Y_scalar = torch.cat(
                [self.train_Y_scalar, torch.tensor([[avg_displacement]], dtype=torch.float64, device=self.device)], dim=0
            )
            self.train_Y_full = torch.cat(
                [self.train_Y_full, torch.as_tensor(displacements, dtype=torch.float64, device=self.device).unsqueeze(0)], dim=0
            )
            self.original_X.append(x_orig_numpy)
            self.displacements_list.append(displacements)
            self._save_iteration_data(x_orig_numpy, displacements)
            print(f"  Init eval @ {x_orig_numpy} | avg={avg_displacement:.6f}")

        # 2. Run optimization batches
        for b in range(self.n_batches):
            print(f"\n[Batch {b+1}/{self.n_batches}] Finding next candidates...")
            gp = self.fit_gp_scalar()
            X_new_scaled = self.optimize_acquisition_function(gp)
            print(f"  Evaluating {self.batch_size} new candidates...")
            for x_scaled in X_new_scaled:
                x_orig_numpy = unnormalize(x_scaled, bounds=self.bounds).cpu().numpy()
                displacements = self.run_simulation(x_orig_numpy)
                avg_displacement = float(np.mean(displacements))
                self.train_X = torch.cat([self.train_X, x_scaled.unsqueeze(0)], dim=0)
                self.train_Y_scalar = torch.cat(
                    [self.train_Y_scalar, torch.tensor([[avg_displacement]], dtype=torch.float64, device=self.device)], dim=0
                )
                self.train_Y_full = torch.cat(
                    [self.train_Y_full, torch.as_tensor(displacements, dtype=torch.float64, device=self.device).unsqueeze(0)], dim=0
                )
                self.original_X.append(x_orig_numpy)
                self.displacements_list.append(displacements)
                self._save_iteration_data(x_orig_numpy, displacements)
                print(f"    Eval @ {x_orig_numpy} | avg={avg_displacement:.6f}")
        
        # 3. Fit the final, detailed model and save it
        print("\nFitting final vector-output GPs for prediction...")
        self.fit_gp_vector()
        models_path = os.path.join(self.output_dir, "gp_models.pt")
        self.save_models(models_path)

        # Return the best result found
        best_params, best_disps = self.return_best_result()
        print("Optimization completed.")
        return best_params, best_disps