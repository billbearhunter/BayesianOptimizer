import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import norm
from scipy.optimize import minimize
import streamlit as st
from scipy.special import erf
import time

# Streamlit App
st.set_page_config(layout="wide", page_title="3D Scatter Plot Viewer")

n = 8  # Output dimensions

def target_function(x: np.ndarray) -> np.ndarray:
    x1, x2, x3 = x
    y1 = x1**2 * np.sin(5 * x2) + 0.5 * x3 * np.exp(-0.1 * (x1**2 + x2**2))
    y2 = np.where(x1 > 0,
                  x1 + 0.3 * x2 - 0.5 * x3,
                  2 * x1 - x2 + 0.7 * np.abs(x3))
    y3 = x1**2 / 4 + x2**2 / 9 - x3**2 / 16 + 0.2 * x1 * x2 * x3
    y4 = erf(0.5 * x1) + np.cos(3 * x2 + 0.5 * x3)
    y5 = np.where(x3 > 1,
                  np.exp(0.3 * (x1 + x2)),
                  np.log1p(np.abs(x1 - x2)))
    r = np.sqrt(x1**2 + x2**2 + x3**2)
    y6 = 2 * np.exp(-0.2 * r**2) + 0.5 * np.exp(-0.05 * (r - 5)**2)
    y7 = 3.9 * x1 * (1 - x1 / 5) * np.sin(x2) + 0.1 * np.mod(x3, 2.5)
    threshold = np.sin(x1 + x2) + np.cos(x2 + x3)
    y8 = np.where(threshold > 0.5,
                  1.5 * x1 - 0.8 * x2 + 0.3 * x3**2,
                  -0.5 * x1 + 1.2 * np.sqrt(np.abs(x2)) + 0.4 * x3)
    return np.array([y1, y2, y3, y4, y5, y6, y7, y8])

def negative_EI(x, gp, X_train, Y_train):
    x = x.reshape(1, -1)
    mu = gp.predict(x)
    std = np.std(mu, axis=0)
    var = std ** 2
    best_y = np.max(Y_train, axis=0)
    var[var == 0] = 1e-10
    Z = (mu - best_y) / np.sqrt(var)
    ei = (mu - best_y) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)
    return -np.sum(ei)

@st.cache_data
def iterative_optimization(sampling_number):
    start_time = time.time()

    X_train = np.random.uniform(-5, 5, (5, 3))
    Y_train = np.array([target_function(x) for x in X_train])

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-8, 1e4))
    gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20))
    gp.fit(X_train, Y_train)

    mse_list = []

    for i in range(sampling_number):
        gp.fit(X_train, Y_train)
        Y_pred = gp.predict(X_train)
        mse = np.mean((Y_train - Y_pred) ** 2)
        mse_list.append(mse)

        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        result = minimize(negative_EI, x0=np.random.uniform(-5, 5, 3), args=(gp, X_train, Y_train), bounds=bounds, method='L-BFGS-B')
        x_new = result.x
        y_new = target_function(x_new)
        X_train = np.vstack((X_train, x_new))
        Y_train = np.vstack((Y_train, y_new))

    end_time = time.time()
    duration = end_time - start_time

    return gp, X_train, Y_train, duration, mse_list

# Set number of iterations
n_iter = st.sidebar.number_input("Number of iterations", 10, 1000, 100, 10)
gp, X_train, Y_train, duration, mse_list = iterative_optimization(n_iter)

st.markdown(f"**Execution Time**: {duration:.2f} seconds")

# Plot MSE over iterations
fig_mse, ax = plt.subplots(figsize=(6, 3))
ax.plot(range(1, len(mse_list) + 1), mse_list, marker='o', linestyle='-', color='b')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Squared Error (MSE)')
ax.set_title('MSE over Iterations')
ax.grid(True, linestyle='--', linewidth=0.5)
st.pyplot(fig_mse)

# Prediction grid
x_range = np.linspace(-5, 5, 15)
y_range = np.linspace(-5, 5, 15)
z_range = np.linspace(-5, 5, 15)
X_test = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

Y_real = np.array([target_function(x) for x in X_test])
Mu = gp.predict(X_test)

df_1 = pd.DataFrame({'X_test_1': X_test[:, 0], 'X_test_2': X_test[:, 1], 'X_test_3': X_test[:, 2]})
df_2 = df_1.copy()
df_3 = df_1.copy()

for i in range(n):
    df_1[f'Y_real_{i+1}'] = Y_real[:, i]
    df_2[f'Mu_{i+1}'] = Mu[:, i]
    denominator = np.where(np.abs(Y_real[:, i]) < 1e-8, 1e-8, np.abs(Y_real[:, i]))
    df_3[f'diff_{i+1}'] = np.clip(
            100 * np.abs(Y_real[:, i] - Mu[:, i]) / denominator,
            0, 100
        )

# Output selection
index = st.selectbox('Select Output Index', range(1, n + 1))
st.markdown(f"Overall MSE: {np.mean((Y_real - Mu) ** 2):.5f}")

# 3D plots
trace_real = go.Scatter3d(
    x=df_1['X_test_1'],
    y=df_1['X_test_2'],
    z=df_1['X_test_3'],
    mode='markers',
    marker=dict(size=2, color=df_1[f'Y_real_{index}'], colorscale='Viridis', opacity=0.5),
    name=f'Y_real_{index}'
)
trace_mu = go.Scatter3d(
    x=df_2['X_test_1'],
    y=df_2['X_test_2'],
    z=df_2['X_test_3'],
    mode='markers',
    marker=dict(size=2, color=df_2[f'Mu_{index}'], colorscale='Viridis', opacity=0.5),
    name=f'Mu_{index}'
)
trace_diff = go.Scatter3d(
    x=df_3['X_test_1'],
    y=df_3['X_test_2'],
    z=df_3['X_test_3'],
    mode='markers',
    marker=dict(size=2, color=df_3[f'diff_{index}'], colorscale='Viridis', opacity=0.5,
                colorbar=dict(title='Percentage Error (%)')),
    name=f'diff_{index}'
)

fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scatter3d'}] * 3],
                    subplot_titles=[f'True Values Y_real_{index}', f'Predicted Mu_{index}', f'Percentage Error'])

fig.add_trace(trace_real, row=1, col=1)
fig.add_trace(trace_mu, row=1, col=2)
fig.add_trace(trace_diff, row=1, col=3)

fig.update_layout(height=350, margin=dict(r=5, l=5, b=10, t=60), showlegend=False)
st.plotly_chart(fig, use_container_width=True)
