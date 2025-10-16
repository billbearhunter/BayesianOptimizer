import os
import pandas as pd
import numpy as np
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize

# --- Step 1: Configuration ---
# This section defines the input data, output path, and parameter bounds.
# Please ensure these paths and bounds match your project setup.

# Path to the CSV file containing the simulation results.
DATA_FILE_PATH = 'results/optimization_results.csv'

# Path where the trained model will be saved.
MODEL_SAVE_PATH = 'results/gp_models.pt'

# The physical bounds of your input parameters.
# This MUST be identical to the bounds used during optimization.
BOUNDS_LIST = [
    [0.3, 1.0],       # n
    [0.001, 300.0],   # eta
    [0.001, 400.0],   # sigma_y
    [2.0, 7.0],       # width
    [2.0, 7.0],       # height
]

def train_and_save_gp_model():
    """
    Main function to load data, train a multi-output GP model, and save it.
    """
    print(f"Loading data from '{DATA_FILE_PATH}'...")
    
    # --- Step 2: Load and Prepare Data ---
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'.")
        return

    df = pd.read_csv(DATA_FILE_PATH)

    if df.empty:
        print(f"Error: The data file '{DATA_FILE_PATH}' is empty. No model can be trained.")
        return
    
    print(f"Loaded {len(df)} data points.")

    # Define feature (input) and target (output) column names
    feature_cols = ['n', 'eta', 'sigma_y', 'width', 'height']
    target_cols = [col for col in df.columns if col.startswith('x_')]
    
    # Separate features and targets, ensuring they are numeric
    X = df[feature_cols].astype(np.float64)
    Y = df[target_cols].astype(np.float64)
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_X_orig = torch.tensor(X.values, dtype=torch.float64, device=device)
    train_Y = torch.tensor(Y.values, dtype=torch.float64, device=device)

    # Normalize the input features to the [0, 1] range
    bounds_tensor = torch.tensor(BOUNDS_LIST, dtype=torch.float64, device=device).t()
    train_X_normalized = normalize(train_X_orig, bounds=bounds_tensor)

    # --- Step 3: Train the Multi-Output GP Model ---
    print("\nTraining 8 independent Gaussian Process models...")
    
    output_models = []
    # Train one independent GP model for each of the 8 output dimensions
    for i in range(Y.shape[1]):
        print(f"  Training model for target '{target_cols[i]}'...")
        
        # Select the i-th output column
        train_Y_i = train_Y[:, i].unsqueeze(-1)
        
        # Create and fit the model
        gp = SingleTaskGP(train_X_normalized, train_Y_i)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        output_models.append(gp)
        
    print("All models trained successfully.")

    # --- Step 4: Save the Models to a .pt File ---
    # We save the state dictionaries, which is the recommended way for PyTorch models.
    # This format is compatible with the `load_models` function in your BayesianOptimizer.
    
    print(f"\nSaving trained models to '{MODEL_SAVE_PATH}'...")
    
    checkpoint = {
        "bounds": bounds_tensor.detach().cpu(),
        "dim": len(feature_cols),
        "output_dim": len(target_cols),
        "n_train": len(df),
        "gp_scalar": None, # Not trained in this script, so we save None
        "gp_outputs": [gp.state_dict() for gp in output_models]
    }
    
    torch.save(checkpoint, MODEL_SAVE_PATH)
    
    print(f"Successfully saved model checkpoint to '{MODEL_SAVE_PATH}'.")
    print("You can now use this file with your evaluation.py script.")


if __name__ == '__main__':
    # Ensure the 'results' directory exists before saving
    os.makedirs('results', exist_ok=True)
    train_and_save_gp_model()