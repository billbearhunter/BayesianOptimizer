import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_dataframes(X_test: np.ndarray, Y_real: np.ndarray, Mu: np.ndarray):
    """
    Given test inputs X_test and corresponding real outputs Y_real and GP predictions Mu,
    returns three DataFrames:
      - df_real: contains X_test cols and Y_real for each task
      - df_mu: contains X_test cols and Mu for each task
      - df_diff: contains X_test cols and absolute difference per task
    """
    dims = X_test.shape[1]
    n_tasks = Y_real.shape[1]
    cols = [f'X_test_{i+1}' for i in range(dims)]
    df_real = pd.DataFrame(X_test, columns=cols)
    df_mu = df_real.copy()
    df_diff = df_real.copy()

    for i in range(n_tasks):
        df_real[f'Y_real_{i+1}'] = Y_real[:, i]
        df_mu[f'Mu_{i+1}'] = Mu[:, i]
        df_diff[f'diff_{i+1}'] = np.abs(Y_real[:, i] - Mu[:, i])

    return df_real, df_mu, df_diff

def plot_mse(step: np.ndarray, MSEs: np.ndarray,
             xlabel: str = 'Number of Steps',
             ylabel: str = 'Mean Squared Error (MSE)',
             title: str = 'Effect of Step Increase on Error (MSE)') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(step, MSEs, marker='o', linestyle='-')
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    return fig

def plot_3d_scatter(df_real: pd.DataFrame,
                    df_mu: pd.DataFrame,
                    df_diff: pd.DataFrame,
                    index: int) -> go.Figure:
    key_real = f'Y_real_{index}'
    key_mu = f'Mu_{index}'
    key_diff = f'diff_{index}'

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[key_real, key_mu, key_diff]
    )

    fig.add_trace(
        go.Scatter3d(
            x=df_real['X_test_1'], y=df_real['X_test_2'], z=df_real['X_test_3'],
            mode='markers',
            marker=dict(size=2, color=df_real[key_real], colorscale='Viridis', opacity=0.5),
            name=key_real
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=df_mu['X_test_1'], y=df_mu['X_test_2'], z=df_mu['X_test_3'],
            mode='markers',
            marker=dict(size=2, color=df_mu[key_mu], colorscale='Viridis', opacity=0.5),
            name=key_mu
        ), row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(
            x=df_diff['X_test_1'], y=df_diff['X_test_2'], z=df_diff['X_test_3'],
            mode='markers',
            marker=dict(size=2, color=df_diff[key_diff], colorscale='Viridis', opacity=0.5,
                        colorbar=dict(title='Absolute Difference')),
            name=key_diff
        ), row=1, col=3
    )

    fig.update_layout(
        height=350,
        margin=dict(r=5, l=5, b=10, t=60),
        showlegend=False
    )
    return fig

def visualize_streamlit(X_test: np.ndarray, Y_real: np.ndarray, Mu: np.ndarray,
                        step: np.ndarray, MSEs: np.ndarray):
    import streamlit as st
    df_real, df_mu, df_diff = prepare_dataframes(X_test, Y_real, Mu)
    # st.pyplot(plot_mse(step, MSEs))
    st.markdown(f"**Overall MSE**: {np.mean((Y_real - Mu)**2):.6f}")
    n_tasks = Y_real.shape[1]
    index = st.selectbox('Select Output Index', list(range(1, n_tasks+1)))
    fig3d = plot_3d_scatter(df_real, df_mu, df_diff, index)
    st.plotly_chart(fig3d, use_container_width=True)
