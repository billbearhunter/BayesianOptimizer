import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, Sum
from scipy.stats import norm
from BayesianOptimizer_RnR1 import BayesianOptimizer 
import time


# Configure Streamlit page
st.set_page_config(page_title="Bayesian Optimization Demo", layout="wide")
st.title("Interactive Bayesian Optimization Demo")

# Define target function (modified for 2D input handling)
def target_function(x):
    x = x.ravel()  # Flatten input to 1D array
    return -np.sin(3 * x) - x**2 + 0.7 * x

# Initialize optimizer
opt = BayesianOptimizer(
        dimensions=1,
        bounds=[(-5,5)],
        noise_level=0.1,
        acquisition_type='EI',
        exploration_rate=0.3,
        kernel=Matern(nu=1.5) + WhiteKernel(),
        random_state=42
    )

#initialize time
elapsed_time = 0

# Create plot containers
col1 = st.columns(1)[0]
plot_placeholder1 = col1.empty()

# Sidebar controls
with st.sidebar:
    st.header("Control Panel")
    n_iter = st.slider("Number of Iterations", 10, 100, 50)
    run_button = st.button("Start Optimization")
    reset_button = st.button("Reset")

# Initialize Session State
if 'opt' not in st.session_state or reset_button:
    st.session_state.opt = opt
    st.session_state.history = []

# Optimization loop
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()

    #calculate progress time
    start_time = time.time()
    st.session_state.history = []

    
    for i in range(n_iter):
        # Execute optimization iteration
        # st.session_state.opt.optimize(target_function, n_iter=1)

        with st.spinner(f"Running iteration {i+1}/{n_iter}..."):
            st.session_state.opt.optimize(target_function, n_iter=1)

        # Update progress
        progress = (i+1)/n_iter
        progress_bar.progress(progress)
        status_text.text(f"Completed: {i+1}/{n_iter} iterations")
        
        # Get current state
        X = st.session_state.opt.X
        Y = st.session_state.opt.Y
        x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot target function and predictions
        if len(X) > 0:
            gp = st.session_state.opt._dynamic_gp()
            mu, sigma = gp.predict(x_test, return_std=True)
            
            ax1.plot(x_test, target_function(x_test), 'r:', label='True Function')
            ax1.plot(X, Y, 'ro', markersize=8, label='Observations')
            ax1.plot(x_test, mu, 'b-', label='GP Prediction')
            ax1.fill_between(x_test.ravel(), 
                           mu - 1.96*sigma, 
                           mu + 1.96*sigma,
                           alpha=0.2)
            ax1.set_title(f'Iteration {i+1} - Function Approximation')
            ax1.legend()
        
        # Plot acquisition function
        if len(X) > 0:
            ei = st.session_state.opt._acquisition(x_test)
            ax2.plot(x_test, ei, 'g-', label='Expected Improvement (EI)')
            ax2.set_title('Acquisition Function')
            ax2.legend()
        
        # Update plots
        plot_placeholder1.pyplot(fig)
        # plot_placeholder2.line_chart({
        #     'Best Value': st.session_state.opt.best_values,
        #     'Current Values': Y[-10:] if len(Y) > 0 else []
        # })
        
        plt.close(fig)

    progress_bar.empty()
    status_text.empty()
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    st.success("Optimization finished!")

# Display final results
if len(st.session_state.opt.X) > 0:
    st.subheader("Optimization Results")
    best_idx = np.argmax(st.session_state.opt.Y)
    st.write(f"Optimal Solution: x = {st.session_state.opt.X[best_idx][0]:.3f}")
    st.write(f"Optimal Value: {st.session_state.opt.Y[best_idx]:.3f}")
    st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

# Display historical data
with st.expander("View Optimization History"):
    if len(st.session_state.opt.X) > 0:
        st.write("Observation Data:")
        st.dataframe({
            "x": st.session_state.opt.X.ravel(),
            "y": st.session_state.opt.Y.ravel()
        })