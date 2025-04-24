import numpy as np
import time
from scipy.special import erf
import matplotlib.pyplot as plt
from .base_optimizer import BaseBayesianOptimizer
from .multitask_optimizer import MultiTaskBayesianOptimizer
from .independent_optimizer import IndependentBayesianOptimizer

def target_function(x: np.ndarray) -> np.ndarray:
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

if __name__ == "__main__":
    bounds = [(-3, 3)] * 3
    task_weights = [1] * 8

    print("Running MultiTask Optimizer...")
    mt_start = time.time()
    multitask_opt = MultiTaskBayesianOptimizer(
        objective_fn=target_function,
        dimensions=3,
        bounds=bounds,
        n_tasks=8,
        task_weights=task_weights,
        acquisition_type="EI",
        random_state=0,
    )
    result_mt = multitask_opt.optimize(n_iter=200)
    
    mt_end = time.time()
    print(f"MultiTask Optimization Time: {mt_end - mt_start:.2f} seconds")

    plt.plot(np.max(result_mt["best_values"], axis=1), label="MultiTask")
    # plt.plot(np.max(result_ind["best_values"], axis=1), label="Independent")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value")
    plt.title("Bayesian Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
