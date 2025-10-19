import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_gaussian_ellipsoid(df, column_names, confidence_levels=[0.95, 0.6827]):
    """
    Analyzes 3D continuous data from a Pandas DataFrame, fits a Trivariate
    Normal distribution, and plots the data with confidence ellipsoids.

    Args:
        df (pd.DataFrame): Input DataFrame with 3 continuous columns.
        column_names (list): List of three column names to use (e.g., ['X', 'Y', 'Z']).
        confidence_levels (list): Confidence levels for the ellipsoids (e.g., 0.95 for 95%).
    """

    # 1. Prepare and Check Data
    if len(column_names) != 3:
        raise ValueError("Must provide exactly three column names for 3D analysis.")

    data = df[column_names].values
    n_points = data.shape[0]
    if n_points < 4:
        raise ValueError("Requires at least 4 data points for covariance calculation.")

    # 2. Calculate Trivariate Normal Parameters

    # Center (Mean Vector: mu)
    center = np.mean(data, axis=0)

    # Covariance Matrix (Sigma)
    # The warning from before is handled by the data generation, but real data
    # may sometimes yield numerical issues. rowvar=False ensures columns are variables.
    cov_matrix = np.cov(data, rowvar=False)

    # Eigen-decomposition for shape and orientation
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Degrees of freedom for Chi-squared distribution (k=3 for 3D)
    k = 3

    print("\n--- Trivariate Normal Distribution Parameters ---")
    print(f"Mean Vector (mu):\n{center}")
    print(f"\nCovariance Matrix (Sigma):\n{cov_matrix}")
    print(f"\nEigenvalues (Lambda):\n{eigenvalues}")

    # 3. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               s=10, alpha=0.6, label='Data Points')
    ax.scatter(center[0], center[1], center[2],
               s=100, color='k', marker='o', label='Mean Center')


    # 4. Plot Ellipsoids for specified confidence levels

    # Generate points on a unit sphere (helper for plotting)
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    for conf_level in confidence_levels:

        # Calculate critical Chi-square value (c)
        c = chi2.ppf(conf_level, k)

        # Calculate Radii for this confidence level: sqrt(c * eigenvalue)
        radii = np.sqrt(c * eigenvalues)

        # Ellipsoid coordinates generation
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Combine unit points into a matrix
        points = np.vstack([x.ravel(), y.ravel(), z.ravel()])

        # Rotate and translate points to form the scaled ellipsoid
        # (rotation matrix is the eigenvectors)
        transformed_points = eigenvectors @ points

        # Reshape and translate
        x_rotated = transformed_points[0].reshape(x.shape) + center[0]
        y_rotated = transformed_points[1].reshape(y.shape) + center[1]
        z_rotated = transformed_points[2].reshape(z.shape) + center[2]

        # Plot the surface
        ax.plot_surface(x_rotated, y_rotated, z_rotated, rstride=3, cstride=3,
                        color='r' if conf_level == 0.95 else 'b',
                        alpha=0.1 + (1-conf_level)/2, # Tighter ellipse is slightly less transparent
                        linewidth=0, shade=False,
                        label=f'{conf_level*100:.0f}% Confidence Ellipsoid')

    # 5. Final Plot Customization
    ax.set_xlabel(column_names[0])
    ax.set_ylabel(column_names[1])
    ax.set_zlabel(column_names[2])
    ax.set_title(f'Trivariate Normal Fit of 3D Data (N={n_points})')

    # Create a dummy element for the legend since plot_surface doesn't support labels well
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='r', lw=4, alpha=0.3),
                    Line2D([0], [0], color='b', lw=4, alpha=0.4)]
    ax.legend(custom_lines, ['95.0% Confidence', '68.3% Confidence'], loc='upper right')

    plt.savefig('stat_plot.png')

# ----------------------------------------------------
#               EXAMPLE USAGE
# ----------------------------------------------------

# Fix the covariance matrix to be positive-definite for robust generation
mu_example = np.array([5, 10, 15])
cov_fixed = np.array([
    [4, 1.5, 0.5],
    [1.5, 2, -0.2],  # Adjusted covariance
    [0.5, -0.2, 1]
])

# Generate 500 data points
n_points_example = 500
np.random.seed(42) # for reproducibility
sample_data_array = np.random.multivariate_normal(mu_example, cov_fixed, size=n_points_example)

# Create the Pandas DataFrame input
df_data = pd.DataFrame(sample_data_array, columns=['Feature_X', 'Feature_Y', 'Feature_Z'])

# Run the function
plot_3d_gaussian_ellipsoid(df_data, column_names=['Feature_X', 'Feature_Y', 'Feature_Z'])
