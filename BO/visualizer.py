import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go

def prepare_dataframes(X_test: np.ndarray, Y_real: np.ndarray, Mu: np.ndarray):
    dims = X_test.shape[1]
    n_tasks = Y_real.shape[1]
    cols = [f'X_test_{i+1}' for i in range(dims)]
    df_real = pd.DataFrame(X_test, columns=cols)
    df_mu = df_real.copy()
    df_diff = df_real.copy()

    for i in range(n_tasks):
        df_real[f'Y_real_{i+1}'] = Y_real[:, i]
        df_mu[f'Mu_{i+1}'] = Mu[:, i]
        denominator = np.where(np.abs(Y_real[:, i]) < 1e-8, 1e-8, np.abs(Y_real[:, i]))
        # df_diff[f'diff_{i+1}'] = 100 * np.abs(Y_real[:, i] - Mu[:, i]) / denominator

        df_diff[f'diff_{i+1}'] = np.clip(
            100 * np.abs(Y_real[:, i] - Mu[:, i]) / denominator,
            0, 20
        )

    return df_real, df_mu, df_diff

def plot_mse(step: np.ndarray, MSEs: np.ndarray,
             xlabel: str = 'Iteration',
             ylabel: str = 'Mean Squared Error (MSE)',
             title: str = 'MSE over Iterations',
             log_x: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(step, MSEs, marker='o', linestyle='-')
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    return fig

def visualize_streamlit(X_test: np.ndarray, Y_real: np.ndarray, Mu: np.ndarray,
                        step: np.ndarray, MSEs: np.ndarray):
    import streamlit as st

    df_real, df_mu, df_diff = prepare_dataframes(X_test, Y_real, Mu)

    # Show MSE line plot
    fig_mse = plot_mse(step, MSEs, log_x=False)
    st.pyplot(fig_mse)

    # Select which task to show
    n_tasks = Y_real.shape[1]
    task_index = st.selectbox("Select Task Index", list(range(1, n_tasks + 1)))
    key_real = f'Y_real_{task_index}'
    key_mu = f'Mu_{task_index}'
    key_diff = f'diff_{task_index}'

    st.markdown(f"**Overall MSE**: {np.mean((Y_real - Mu)**2):.6f}")

    # Plot true values
    fig_real = go.Figure(go.Scatter3d(
        x=df_real['X_test_1'],
        y=df_real['X_test_2'],
        z=df_real['X_test_3'],
        mode='markers',
        marker=dict(size=2, color=df_real[key_real], colorscale='Viridis', opacity=0.5),
        name='True Values'
    ))
    fig_real.update_layout(title=f"True Values: Task {task_index}", margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_real, use_container_width=True)

    # Plot predicted values
    fig_pred = go.Figure(go.Scatter3d(
        x=df_mu['X_test_1'],
        y=df_mu['X_test_2'],
        z=df_mu['X_test_3'],
        mode='markers',
        marker=dict(size=2, color=df_mu[key_mu], colorscale='Viridis', opacity=0.5),
        name='Predicted Values'
    ))
    fig_pred.update_layout(title=f"Predicted Values: Task {task_index}", margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_pred, use_container_width=True)

    # Plot prediction error
    fig_diff = go.Figure(go.Scatter3d(
        x=df_diff['X_test_1'],
        y=df_diff['X_test_2'],
        z=df_diff['X_test_3'],
        mode='markers',
        marker=dict(size=2, color=df_diff[key_diff], colorscale='Viridis', opacity=0.5,
                    colorbar=dict(title='Error (%)')),
        name='Prediction Error'
    ))
    fig_diff.update_layout(title=f"Prediction Error (%): Task {task_index}", margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_diff, use_container_width=True)
