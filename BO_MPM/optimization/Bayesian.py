import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import norm
from scipy.optimize import minimize
from config import *
from simulation.file_ops import FileOperations

class BayesianOptimizer:
    def __init__(self, simulator, output_dir):
        self.simulator = simulator
        self.output_dir = output_dir
        self.file_ops = FileOperations()
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-8, 1e4))
        self.gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10))
        
    def optimize(self, sampling_number, seed=42):
        """Run Bayesian optimization"""
        np.random.seed(seed)
        print(f"Starting Bayesian optimization. Samples: {sampling_number}, Random seed: {seed}")
        
        # Initial sample
        n = np.random.uniform(MIN_N, MAX_N)
        eta = np.random.uniform(MIN_ETA, MAX_ETA)
        sigma_y = np.random.uniform(MIN_SIGMA_Y, MAX_SIGMA_Y)
        
        # Run initial simulation
        start_time = time.time()
        y_new = self._run_simulation(n, eta, sigma_y)
        elapsed = time.time() - start_time
        
        # Prepare data
        X_train = np.array([n, eta, sigma_y]).reshape(1, -1)
        Y_train = y_new.reshape(1, -1)
        
        # Save initial data
        self._save_iteration_data(0, n, eta, sigma_y, y_new, elapsed)
        print(f"Initial sample completed: n={n:.4f}, eta={eta:.4f}, σ_y={sigma_y:.4f}")
        print(f"Displacements: {y_new} | Time: {elapsed:.1f}s")
        
        # Optimization loop
        for i in range(1, sampling_number):
            print(f"\nIteration {i+1}/{sampling_number}")
            
            # Fit Gaussian process
            self.gp.fit(X_train, Y_train)
            
            # Find next sample point
            x_new, acquisition_time = self._find_next_sample(X_train, Y_train)
            
            # Run new simulation
            sim_start = time.time()
            y_new = self._run_simulation(*x_new)
            sim_time = time.time() - sim_start
            
            # Save results
            self._save_iteration_data(i, *x_new, y_new, sim_time)
            
            # Update training set
            X_train = np.vstack((X_train, x_new))
            Y_train = np.vstack((Y_train, y_new.reshape(1, -1)))
            
            # Print progress
            print(f"Completed iteration {i+1} | n={x_new[0]:.4f}, eta={x_new[1]:.4f}, σ_y={x_new[2]:.4f}")
            print(f"Displacements: {y_new} | Simulation time: {sim_time:.1f}s")
        
        # Complete optimization
        total_time = time.time() - start_time
        print(f"\nOptimization completed! Total time: {total_time:.1f} seconds")
        
        # Find and return best result
        best_idx = np.argmax(Y_train[:, -1])
        best_params = X_train[best_idx]
        best_displacement = Y_train[best_idx, -1]
        
        print("\nBest parameters found:")
        print(f"  n = {best_params[0]:.4f}")
        print(f"  eta = {best_params[1]:.4f}")
        print(f"  yield_stress = {best_params[2]:.4f}")
        print(f"  Maximum displacement: {best_displacement:.4f}")
        
        return X_train, Y_train
    
    def _run_simulation(self, n, eta, sigma_y):
        """Run simulation and return displacement results"""
        print(f"  Running simulation: n={n:.4f}, eta={eta:.4f}, σ_y={sigma_y:.4f}")
        return self.simulator.run_simulation(n, eta, sigma_y)
    
    def _find_next_sample(self, X_train, Y_train):
        """Find next sample using EI acquisition function"""
        start_time = time.time()
        
        bounds = [(MIN_N, MAX_N), (MIN_ETA, MAX_ETA), (MIN_SIGMA_Y, MAX_SIGMA_Y)]
        result = minimize(
            self._negative_EI, 
            x0=np.array([
                np.random.uniform(MIN_N, MAX_N), 
                np.random.uniform(MIN_ETA, MAX_ETA), 
                np.random.uniform(MIN_SIGMA_Y, MAX_SIGMA_Y)
            ]),
            args=(X_train, Y_train), 
            bounds=bounds, 
            method='L-BFGS-B'
        )
        
        elapsed = time.time() - start_time
        print(f"  Acquisition time: {elapsed:.2f}s")
        return result.x, elapsed
    
    def _negative_EI(self, x, X_train, Y_train):
        """Calculate negative expected improvement"""
        x = x.reshape(1, -1)
        mu = self.gp.predict(x)
        std = np.std(mu, axis=0)
        var = std ** 2
        best_y = np.max(Y_train, axis=0)
        var[var == 0] = 1e-10  # Prevent division by zero
        Z = (mu - best_y) / np.sqrt(var)
        ei = (mu - best_y) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)
        return -np.sum(ei)
    
    def _save_iteration_data(self, iteration, n, eta, sigma_y, displacements, sim_time):
        """Save iteration data to file"""
        data = [n, eta, sigma_y, *displacements, sim_time]
        # Append to main results file
        with open(f"{self.output_dir}/optimization_results.csv", 'a') as f:
            f.write(','.join(map(str, data)) + '\n')