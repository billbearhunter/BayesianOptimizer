import streamlit as st
import torch
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import logging

# logging.basicConfig(level=logging.DEBUG)

@st.cache_data
def compute_gp_predictions(_gp_model, _X_test_tensor, _task_idx):
    with torch.no_grad():
        posterior = _gp_model.posterior(_X_test_tensor)
        return posterior.mean.numpy()[:, _task_idx]

@st.cache_data
def compute_true_function(_true_function, _X_test, _task_idx):
    return np.array([_true_function(x) for x in _X_test])[:, _task_idx]

class BayesianOptimizationVisualizer:
    def __init__(self):
        self.figsize = (10, 6)

    def plot_task_3d_visualizations_all(self, history, gp_model=None, resolution=20, true_function=None):
        st.subheader("3D Optimization Visualization")
        task_idx = st.selectbox("Select Task", range(history["Y"].shape[1]))
        logging.debug(f"Selected task index: {task_idx}")

        X_test = self._create_test_grid(history["bounds"], resolution)
        logging.debug(f"Test grid shape: {X_test.shape}")
        X_test_tensor = torch.tensor(X_test, dtype=torch.double)

        if gp_model is not None:
            Y_pred = compute_gp_predictions(gp_model, X_test_tensor, task_idx)
            logging.debug(f"Predicted values shape: {Y_pred.shape}")
        else:
            Y_pred = np.zeros(len(X_test))
            logging.debug("No GP model provided, using zeros for predictions")

        if true_function is not None:
            try:
                Y_real = compute_true_function(true_function, X_test, task_idx)
                logging.debug(f"Real values shape: {Y_real.shape}")
            except Exception as e:
                logging.error(f"Error evaluating true function: {e}")
                Y_real = np.zeros(len(X_test))
        else:
            Y_real = np.zeros(len(X_test))
            logging.debug("No true function provided, using zeros for ground truth")

        df = pd.DataFrame({
            "x1": X_test[:, 0],
            "x2": X_test[:, 1],
            "x3": X_test[:, 2],
            "Predicted": Y_pred,
            "Real": Y_real,
            "Absolute Difference": np.abs(Y_pred - Y_real)
        })
        logging.debug("DataFrame for visualization created")

        self._create_3d_plots(df, task_idx)

    def _create_test_grid(self, bounds, resolution):
        logging.debug(f"Creating test grid with resolution {resolution} and bounds {bounds}")
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        z = np.linspace(bounds[2][0], bounds[2][1], resolution)
        return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    def _create_3d_plots(self, df, task_idx):
        logging.debug("Creating 3D plots")
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=[
                f"Task {task_idx+1} - Predictions",
                f"Task {task_idx+1} - Ground Truth",
                f"Task {task_idx+1} - Absolute Error"
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=df.x1, y=df.x2, z=df.x3,
                mode="markers",
                marker=dict(
                    size=3,
                    color=df.Predicted,
                    colorscale="Viridis",
                    opacity=0.7,
                    colorbar=dict(title="Prediction", len=0.5, x=0.29)
                ),
                name="Predictions"
            ), row=1, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=df.x1, y=df.x2, z=df.x3,
                mode="markers",
                marker=dict(
                    size=3,
                    color=df.Real,
                    colorscale="Viridis",
                    opacity=0.7,
                    colorbar=dict(title="Ground Truth", len=0.5, x=0.6)
                ),
                name="Ground Truth"
            ), row=1, col=2
        )

        fig.add_trace(
            go.Scatter3d(
                x=df.x1, y=df.x2, z=df.x3,
                mode="markers",
                marker=dict(
                    size=3,
                    color=df["Absolute Difference"],
                    colorscale="Hot",
                    opacity=0.7,
                    colorbar=dict(title="Abs Error", len=0.5, x=0.95)
                ),
                name="Absolute Error"
            ), row=1, col=3
        )

        fig.update_layout(
            height=500,
            margin=dict(r=10, l=10, b=10, t=40),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_optimization_progress(self, history):
        st.subheader("Optimization Progress")
        logging.debug("Plotting optimization progress")

        progress_data = pd.DataFrame({
            "Iteration": np.arange(len(history["best_values"])),
            "Best Value": np.maximum.accumulate(history["best_values"].max(axis=1))
        })

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=progress_data.Iteration,
                y=progress_data["Best Value"],
                mode="lines+markers",
                name="Best Value"
            )
        )
        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Best Objective Value",
            height=300,
            margin=dict(r=20, l=20, b=20, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)