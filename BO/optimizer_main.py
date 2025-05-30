import numpy as np
import time
import streamlit as st
import torch
from multitask_optimizer import MultiTaskBayesianOptimizer
from botorch.models.transforms.outcome import Standardize
from visualizer import prepare_dataframes, plot_mse, visualize_streamlit

# def target_function(x: np.ndarray) -> np.ndarray:
#     x1, x2, x3 = x
#     y1 = x1**2 * np.sin(5 * x2) + 0.5 * x3 * np.exp(-0.1 * (x1**2 + x2**2))
#     y2 = np.where(x1 > 0, x1 + 0.3 * x2 - 0.5 * x3, 2 * x1 - x2 + 0.7 * np.abs(x3))
#     y3 = x1**2 / 4 + x2**2 / 9 - x3**2 / 16 + 0.2 * x1 * x2 * x3
#     y4 = float(torch.erf(torch.tensor(0.5 * x1))) + np.cos(3 * x2 + 0.5 * x3)
#     y5 = np.where(x3 > 1, np.exp(0.3 * (x1 + x2)), np.log1p(np.abs(x1 - x2)))
#     r = np.sqrt(x1**2 + x2**2 + x3**2)
#     y6 = 2 * np.exp(-0.2 * r**2) + 0.5 * np.exp(-0.05 * (r - 5)**2)
#     y7 = 3.9 * x1 * (1 - x1 / 5) * np.sin(x2) + 0.1 * np.mod(x3, 2.5)
#     threshold = np.sin(x1 + x2) + np.cos(x2 + x3)
#     y8 = np.where(threshold > 0.5,
#                  1.5 * x1 - 0.8 * x2 + 0.3 * x3**2,
#                  -0.5 * x1 + 1.2 * np.sqrt(np.abs(x2)) + 0.4 * x3)
#     return np.array([y1, y2, y3, y4, y5, y6, y7, y8])

def target_function(x: np.ndarray) -> np.ndarray:
    return [
        np.sin(x[0]) + np.cos(x[1]) + 0.5 * x[2] + np.random.normal(0, 0.1) for _ in range(8)
    ]

def main():
    st.title("Bayesian Optimization")
    n_iter = st.sidebar.number_input("Number of iterations", 10, 10000, 100, 10)
    acquisition = st.sidebar.selectbox("Acquisition function", ["EI", "UCB"])
    seed = st.sidebar.number_input("Random seed", value=0)

    # Run optimization on button click
    if "results" not in st.session_state and st.sidebar.button("Run Optimization"):
        start = time.time()
        optimizer = MultiTaskBayesianOptimizer(
            objective_fn=target_function,
            dimensions=3,
            bounds=[(-3, 3)] * 3,
            n_tasks=8,
            task_weights=[1.0] * 8,
            acquisition_type=acquisition,
            random_state=seed
        )
        history = optimizer.optimize(n_iter=n_iter)
        elapsed = time.time() - start
        st.success(f"Optimization completed in {elapsed:.2f} seconds.")

        # Prepare test grid and true values
        grid = np.linspace(-5, 5, 15)
        X_test = np.stack(np.meshgrid(grid, grid, grid), -1).reshape(-1, 3)
        Y_true = np.array([target_function(x) for x in X_test])
        X_t = torch.tensor(X_test, dtype=torch.double)

        # Compute MSE history over iterations
        X_all, Y_all = optimizer.X.copy(), optimizer.Y.copy()
        initial_samples = X_all.shape[0] - n_iter
        mse_history = []
        for j in range(n_iter):
            optimizer.X = X_all[:initial_samples + j + 1]
            optimizer.Y = Y_all[:initial_samples + j + 1]
            model_j = optimizer._fit_model()
            _ = model_j.eval()
            _ = model_j.likelihood.eval()
            with torch.no_grad():
                posterior_j = model_j.posterior(X_t)
            means_j = posterior_j.mean.detach().cpu().numpy()
            mse_history.append(np.mean((Y_true - means_j) ** 2))
        optimizer.X, optimizer.Y = X_all, Y_all

        # Final model fit and predictions
        model = optimizer._fit_model()
        _ = model.eval()
        _ = model.likelihood.eval()

        with torch.no_grad():
            posterior = model.posterior(X_t)

        if hasattr(model, "outcome_transform"):
            try:
                posterior = model.outcome_transform.untransform(posterior)
            except Exception as e:
                print(f"[Warning] Failed to untransform posterior: {e}")

        means = posterior.mean.detach().cpu().numpy()

        # Ensure correct orientation
        if means.shape[0] != Y_true.shape[0]:
            means = means.T

        # Store results in session state
        st.session_state.results = {
            "mse_history": mse_history,
            "final_mse": mse_history[-1],
            "means": means,
            "X_test": X_test,
            "Y_true": Y_true
        }

    # Render results if available
    if "results" in st.session_state:
        data = st.session_state.results
        steps = np.arange(1, len(data["mse_history"]) + 1)

        st.markdown(f"**Final Test Grid MSE:** {data['final_mse']:.6f}")
        
        visualize_streamlit(
            data["X_test"],
            data["Y_true"],
            data["means"],
            steps,
            data["mse_history"]
        )

if __name__ == "__main__":
    main()
