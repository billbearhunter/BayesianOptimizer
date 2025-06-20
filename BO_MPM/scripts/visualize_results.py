import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_results(file_path):
    """Visualize optimization results"""
    print(f"Loading results from: {file_path}")
    # Load data
    data = np.loadtxt(file_path, delimiter=',')
    
    # Extract parameters and displacements
    n_values = data[:, 0]
    eta_values = data[:, 1]
    sigma_y_values = data[:, 2]
    final_displacements = data[:, -2]  # Final displacement
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(n_values, eta_values, sigma_y_values, 
                         c=final_displacements, cmap='viridis', s=50)
    
    ax.set_xlabel('n')
    ax.set_ylabel('eta')
    ax.set_zlabel('Yield Stress')
    ax.set_title('Parameter Optimization Results')
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Final Displacement')
    
    plt.tight_layout()
    
    # Save and show
    output_file = file_path.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to: {output_file}")
    plt.show()