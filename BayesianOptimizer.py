import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process.kernels import WhiteKernel, Sum
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class BayesianOptimizer:
    def __init__(self, dimensions, bounds, 
                 noise_level=0.1, 
                 acquisition_type='EI',
                 exploration_rate=0.1,
                 fast_mode=True,
                 random_state=None,
                 kernel=None):
        """
        High-performance Bayesian Optimization with adaptive complexity control
        
        Args:
            dimensions (int): Number of input dimensions
            bounds (list): Search space boundaries [(min, max), ...]
            noise_level (float): Initial noise level estimate
            acquisition_type (str): Acquisition function type ('EI' or 'UCB')
            exploration_rate (float): Exploration-exploitation balance parameter
            fast_mode (bool): Enable computational optimizations
            random_state (int): Seed for reproducible results
            kernel (Kernel): Custom kernel configuration (optional)
        """
        self._validate_inputs(dimensions, bounds)
        self.dims = dimensions
        self.bounds = np.array(bounds)
        self.fast_mode = fast_mode
        self.exploration_rate = exploration_rate
        self.acquisition_type = acquisition_type
        self.random_state = check_random_state(random_state)
        
        self._init_gp_components(noise_level, kernel)
        self.X = np.empty((0, dimensions))
        self.Y = np.empty(0)
        self.best_values = []
        self.iteration = 0

    def _validate_inputs(self, dimensions, bounds):
        """Ensure valid input parameters"""
        if len(bounds) != dimensions:
            raise ValueError("Dimensions/bounds mismatch")
        for i, b in enumerate(bounds):
            if len(b) != 2 or b[0] >= b[1]:
                raise ValueError(f"Invalid bounds at index {i}: must be (min, max) with min < max")

    def _init_gp_components(self, noise_level, kernel):
        """Initialize GP components with adaptive configuration"""
        if kernel is None:
            self.base_kernel = Matern(
                nu=0.5,  # Exponential kernel for faster computations
                length_scale_bounds=(1e-5, 1e3)
            )
            self.noise_kernel = WhiteKernel(
                noise_level=noise_level**2,
                noise_level_bounds=(1e-10, 1e2)
            )
            kernel = self.base_kernel + self.noise_kernel
        else:
            self.base_kernel = kernel
            if isinstance(kernel, Sum):
                parts = [kernel.k1, kernel.k2]
                self.noise_kernel = next(
                    (p for p in parts if isinstance(p, WhiteKernel)), None
                )
            elif isinstance(kernel, WhiteKernel):
                self.noise_kernel = kernel
            else:
                self.noise_kernel = None
            
            if self.noise_kernel is None:
                self.noise_kernel = WhiteKernel(
                    noise_level=noise_level**2,
                    noise_level_bounds=(1e-10, 1e2)
                )

            kernel = self.base_kernel + self.noise_kernel
            
                
        self.full_gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            optimizer='fmin_l_bfgs_b' if not self.fast_mode else None,
            normalize_y=True,
            random_state=self.random_state
        )
        
        self.sparse_gp = None
        self.inducing_points = None

    def _init_sparse_gp(self):
        """Initialize sparse GP with inducing points"""
        self.inducing_points = self.X[::2]  # Simple subset selection
        
        self.sparse_gp = GaussianProcessRegressor(
            kernel=self.base_kernel,
            optimizer=None,
            alpha=self.noise_kernel.noise_level,  
            normalize_y=True,
            random_state=self.random_state
        )
        self.sparse_gp.fit(self.inducing_points, self.Y[::2])

    def _dynamic_gp(self):
        """Select appropriate GP model based on iteration count"""
        if len(self.X) < 1000000:  # Use full GP for first 1000 iterations
            return self.full_gp
        else:  # Switch to sparse approximation
            if self.sparse_gp is None:
                self._init_sparse_gp()
            return self.sparse_gp



    def _acquisition(self, X, gp=None):
        """Numerically stable acquisition function calculation"""
        gp = self._dynamic_gp() if gp is None else gp
        X = np.atleast_2d(X)
        
        try:
            mu, sigma = gp.predict(X, return_std=True)
        except:
            mu, sigma = np.zeros(len(X)), np.ones(len(X))
            
        sigma = np.clip(sigma, 1e-6, None)
        
        if self.Y.size == 0:
            return np.zeros_like(mu)
        
        current_max = np.max(self.Y)
        
        if self.acquisition_type == 'EI':
            improvement = mu - current_max - self.exploration_rate
            Z = improvement / sigma
            return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        elif self.acquisition_type == 'UCB':
            return mu + self.exploration_rate * sigma
        else:
            raise ValueError(f"Unknown acquisition type: {self.acquisition_type}")

    def _generate_candidates(self):
        """Parallel candidate generation with hybrid strategy"""
        random_candidates = Parallel(n_jobs=-1)(
            delayed(self._random_candidate)()
            for _ in range(10 * self.dims)
        )
        
        optimized_candidates = [self._optimized_candidate()]
        candidates = np.vstack([random_candidates, optimized_candidates])
        return np.unique(candidates, axis=0)

    def _random_candidate(self):
        """Generate single random candidate"""
        return self.random_state.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _optimized_candidate(self):
        """Generate candidate through local optimization"""
        try:
            res = minimize(
                fun=lambda x: -self._acquisition(x.reshape(1, -1)).item(),
                x0=self._random_candidate(),
                bounds=self.bounds,
                method='L-BFGS-B',
                options={'maxiter': 50}
            )
            return res.x if res.success else self._random_candidate()
        except:
            return self._random_candidate()

    def optimize(self, objective_fn, n_iter):
        """Optimization loop with adaptive complexity control"""
        if self.X.size == 0:
            self._initial_sample(objective_fn)
        
        for _ in range(n_iter):
            print(f"Iteration {self.iteration + 1}/{n_iter}")
            self.iteration += 1
            try:
                current_gp = self._dynamic_gp()
                if len(self.X) < 1e10 or not self.fast_mode:
                    current_gp.fit(self.X, self.Y)
                
                candidates = self._generate_candidates()
                scores = self._acquisition(candidates, current_gp)
                x_next = candidates[np.nanargmax(scores)]
                
                y_next = self._safe_evaluate(objective_fn, x_next)
                
                if np.isfinite(y_next):
                    self.X = np.vstack([self.X, x_next])
                    self.Y = np.append(self.Y, y_next)
                    self.best_values.append(
                        max(self.best_values[-1], y_next) 
                        if self.best_values else y_next
                    )
            except Exception as e:
                print(f"Iteration {self.iteration} failed: {str(e)}")
                break

    def _initial_sample(self, objective_fn):
        """Generate initial sample with validation"""
        x_init = self._random_candidate()
        try:
            y_init = float(objective_fn(x_init))
            if not np.isfinite(y_init):
                raise ValueError("Initial evaluation returned non-finite value")
        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")
        
        self.X = np.atleast_2d(x_init)
        self.Y = np.array([y_init])
        self.best_values.append(y_init)

    def _safe_evaluate(self, objective_fn, x):
        """Robust function evaluation with error handling"""
        try:
            return float(objective_fn(x))
        except Exception as e:
            print(f"Evaluation failed at {x}: {str(e)}")
            return np.nan

    @property
    def optimization_history(self):
        """Get optimization history data for visualization"""
        return {
            'X': self.X,
            'Y': self.Y,
            'best_values': self.best_values,
            'bounds': self.bounds,
            'dimensions': self.dims,
            'iteration': self.iteration
        }

    def get_current_gp(self):
        """Get current Gaussian Process model for visualization"""
        return self._dynamic_gp()


class BayesianOptimizationVisualizer:
    @staticmethod
    def plot_optimization_progress(best_values):
        """
        Interactive optimization progress visualization
        
        Args:
            best_values (list): Historical best objective values
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(best_values)),
            y=best_values,
            mode='lines+markers',
            name='Best Observed Value'
        ))
        fig.update_layout(
            title='Optimization Progress',
            xaxis_title='Iteration',
            yaxis_title='Objective Value',
            template='plotly_white',
            hovermode='x unified'
        )
        fig.show()

    @staticmethod
    def plot_model_vs_true(optimizer_history, true_function, gp_model, resolution=15):
        """
        3D model comparison visualization
        
        Args:
            optimizer_history (dict): Contains X, Y, bounds, dimensions, iteration
            true_function (callable): Ground truth function for comparison
            gp_model (GaussianProcessRegressor): Trained GP model
            resolution (int): Grid resolution
        """
        if optimizer_history['dimensions'] != 3:
            print("3D visualization requires 3D problem")
            return
            
        resolution = max(10, resolution - optimizer_history['iteration'] // 5)
        axes = [np.linspace(b[0], b[1], resolution) for b in optimizer_history['bounds']]
        grid = np.meshgrid(*axes)
        X_test = np.vstack([g.ravel() for g in grid]).T
        
        mu, _ = gp_model.predict(X_test, return_std=True)
        true_vals = np.array([true_function(x) for x in X_test])
        errors = np.abs(true_vals - mu)
        
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scatter3d'}] * 3],
            subplot_titles=('True Function', 'Model Predictions', 'Error Map')
        )
        
        traces = [
            go.Scatter3d(
                x=X_test[:,0],
                y=X_test[:,1],
                z=X_test[:,2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=vals,
                    colorscale=cmap,
                    opacity=0.7,
                    colorbar=dict(x=pos)
                ),
                name=name
            ) for vals, cmap, pos, name in [
                (true_vals, 'Viridis', 0.3, 'True'),
                (mu, 'Viridis', 0.6, 'Model'),
                (errors, 'RdBu', 1.0, 'Error')
            ]
        ]
        
        for i, trace in enumerate(traces):
            fig.add_trace(trace, row=1, col=i+1)
            
        fig.update_layout(
            title_text='Model Performance Analysis',
            width=1400,
            height=500,
            margin=dict(r=100)
        )
        fig.show()


# Example Usage
if __name__ == "__main__":
    # Benchmark function: 3D Sphere
    def sphere(x):
        return np.sin(x[0]) * x[1] + 0.5 * np.sqrt(abs(x[2]))
    
    # Noisy objective
    def noisy_sphere(x):
        return sphere(x) + np.random.normal(0, 0.1)
    
    # Initialize optimizer
    opt = BayesianOptimizer(
        dimensions=3,
        bounds=[(-5,5), (-5,5), (-5,5)],
        noise_level=0.1,
        acquisition_type='EI',
        exploration_rate=0.3,
        kernel=Matern(nu=1.5) + WhiteKernel(),
        fast_mode=False,
        random_state=42
    )
    
    # Run optimization
    opt.optimize(noisy_sphere, n_iter=200)
    
    # Visualization using separated class
    vis = BayesianOptimizationVisualizer()
    vis.plot_optimization_progress(opt.best_values)
    vis.plot_model_vs_true(
        optimizer_history=opt.optimization_history,
        true_function=sphere,
        gp_model=opt.get_current_gp()
    )