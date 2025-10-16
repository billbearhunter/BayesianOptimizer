import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Step 1: Import the necessary classes ---
from optimization.Bayesian import BayesianOptimizer
from simulation.taichi import MPMSimulator 

# --- Step 2: Modified Evaluation Function for Multi-Output ---
def evaluate_and_print_metrics(y_true_df, y_pred_matrix, model_name):
    """
    Calculates and prints regression metrics for each target variable.

    Args:
        y_true_df (pd.DataFrame): DataFrame with true target values (n_samples, 8).
        y_pred_matrix (np.ndarray): NumPy array with predicted values (n_samples, 8).
        model_name (str): The name/path of the model for printing.
    """
    print(f"\n--- Model Performance for '{model_name}' ---")
    
    metrics_list = []
    
    for i, target_col in enumerate(y_true_df.columns):
        y_true = y_true_df[target_col].values
        y_pred = y_pred_matrix[:, i]
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {
            'Target': target_col,
            'R²': r2,
            'MSE': mse,
            'MAE': mae
        }
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False, float_format="%.6f"))
    return metrics_df


# --- NEW: Function to Predict on a New Sample using the Optimizer ---
def predict_new_sample(optimizer: BayesianOptimizer):
    """
    Uses the loaded optimizer to make a prediction on a new data sample.
    """
    print("\n# ===================================================================")
    print("# EXAMPLE PREDICTION WITH SAVED MODEL")
    print("# ===================================================================")
    
    feature_cols = ['n', 'eta', 'sigma_y', 'width', 'height']
    
    new_sample_raw = np.array([[0.3, 15, 20, 7.0, 7.0]])
    new_sample_df = pd.DataFrame(new_sample_raw, columns=feature_cols)
    print(f"\nInput sample for prediction:\n{new_sample_df.to_string(index=False)}")
    
    predicted_mean, predicted_std = optimizer.predict_vector(new_sample_raw)
    
    print("\n--- Predicted Displacements (Vector Output) ---")
    predictions_df = pd.DataFrame([predicted_mean[0]], columns=[f'pred_x_{i+1:02d}' for i in range(8)])
    print(predictions_df.to_string(index=False, float_format="%.6f"))


# --- Step 3: Main Logic to Load, Predict, and Evaluate ---
def main():
    """
    Main function: loads the .pt model, data, predicts, and evaluates.
    """
    model_path = 'results/gp_models.pt'
    validation_data_path = 'validation_data/split_x_diff_data.csv'

    print(f"Loading GP models from: {model_path}")
    
    dummy_simulator = MPMSimulator("config/setting.xml")
    dummy_bounds_list = [[0.3, 1.0], [0.001, 300.0], [0.001, 400.0], [2.0, 7.0], [2.0, 7.0]]

    optimizer = BayesianOptimizer(
        simulator=dummy_simulator,
        bounds_list=dummy_bounds_list,
        output_dir='results',
        n_initial_points=1, n_batches=1, batch_size=1
    )

    try:
        optimizer.load_models(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please check the path.")
        return

    print(f"Loading validation data from: {validation_data_path}")
    val_df = pd.read_csv(validation_data_path)

    # --- THIS IS THE FIX ---
    # Check if the DataFrame is empty after loading.
    if val_df.empty:
        print(f"\nError: The validation data file '{validation_data_path}' is empty.")
        print("Cannot perform evaluation without any data. Please run the optimization first.")
        return # Exit the script gracefully

    feature_cols = ['n', 'eta', 'sigma_y', 'width', 'height']
    disp_cols = [col for col in val_df.columns if col.startswith('x_')]
    
    # Ensure both input features and true labels are numeric
    X_val = val_df[feature_cols].astype(np.float64)
    y_true_df = val_df[disp_cols].astype(np.float64)

    print("\nMaking predictions on the validation set...")
    y_pred_mean, y_pred_std = optimizer.predict_vector(X_val.values)

    evaluate_and_print_metrics(y_true_df, y_pred_mean, model_path)

    predict_new_sample(optimizer)


if __name__ == '__main__':
    main()