import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

# --- Helper Functions from Original Code ---

# ----------------------------------------------------
#               NEW FUNCTIONS FOR ASSIGNMENT 3
# ----------------------------------------------------

def chebyshev_outlier_removal(df, column_names, threshold_sigma=2.0):
    """
    Identifies and removes outliers from the DataFrame columns based on
    Chebyshev's Theorem (all points outside mean +/- threshold_sigma * std_dev).

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
        pd.DataFrame: A DataFrame containing the detected outliers.
    """
    df_filtered = df.copy()
    outlier_indices = set()
    print(f"\n--- Outlier Detection (Chebyshev {threshold_sigma}\u03c3) ---")

    for col in column_names:
        data = df[col].values
        mu = np.mean(data)
        sigma = np.std(data)
        lower_bound = mu - threshold_sigma * sigma
        upper_bound = mu + threshold_sigma * sigma

        # Find indices where data is outside the range
        col_outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(col_outlier_indices)
        print(f"Column '{col}': {len(col_outlier_indices)} outliers found.")

    outliers_df = df.loc[list(outlier_indices)]
    df_filtered = df.drop(list(outlier_indices))

    print(f"Total points: {len(df)}. Total outliers removed: {len(outliers_df)}. Filtered points: {len(df_filtered)}")
    return df_filtered, outliers_df


def chi_square_gaussian_test(data, col_name, num_bins=10, filename='chi_square_histogram.png'):
    """
    Performs the Chi-square goodness-of-fit test for a 1D Gaussian distribution
    and plots the histogram with the fitted Gaussian PDF.

    Args:
        data (np.array): 1D array of data points.
        col_name (str): Name of the data column for plotting.
        num_bins (int): Number of bins for the histogram and test.
    """
    mu, std = norm.fit(data)
    n = len(data)

    # 1. Calculate Observed Frequencies (O_i) and Bin Edges
    O_i, bin_edges, _ = plt.hist(data, bins=num_bins, density=False, label='Observed Data')
    plt.close() # Close the temporary plot

    # 2. Calculate Expected Frequencies (E_i)
    # The probability of falling in bin i is P_i = CDF(upper_edge) - CDF(lower_edge)
    P_i = norm.cdf(bin_edges[1:], loc=mu, scale=std) - norm.cdf(bin_edges[:-1], loc=mu, scale=std)
    E_i = n * P_i

    # Handle bins with very low expected frequency (rule of thumb: E_i >= 5)
    # Combine the last bins until the criterion is met.
    # Note: For simplicity and robustness with small datasets, we'll proceed
    # with the standard bins, but a warning is warranted if E_i is small.

    # 3. Calculate Chi-square statistic
    # Exclude bins where E_i is zero to avoid division by zero (or combine bins)
    valid_bins = E_i > 0
    O_i_valid = O_i[valid_bins]
    E_i_valid = E_i[valid_bins]

    if len(E_i_valid) < 3:
        print(f"\n[WARNING] Not enough valid bins for Chi-square test on '{col_name}'. Skipping.")
        return None, None, None

    chi2_stat = np.sum((O_i_valid - E_i_valid)**2 / E_i_valid)

    # Degrees of freedom: k - p - 1, where k = number of valid bins, p = 2 (for mu and sigma)
    # The degree of freedom is reduced by one for each estimated parameter.
    df = len(E_i_valid) - 2 - 1

    # 4. Critical Value and p-value
    if df < 1:
        print(f"\n[WARNING] Degrees of freedom for '{col_name}' is {df}. Skipping test.")
        return None, None, None

    alpha = 0.05
    critical_value = chi2.ppf(1 - alpha, df)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    # 5. Plotting (Histogram and PDF)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=num_bins, density=True, alpha=0.6, label=f'Data: {col_name}')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf, 'r-', linewidth=2, label='Fitted Gaussian PDF')
    plt.title(f'Gaussian Fit & $\chi^2$ Test for {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'figures/{filename}_{col_name}.png', dpi=300)
    plt.close()

    # 6. Conclusion
    is_gaussian = p_value > alpha # Null hypothesis is that data is Gaussian
    result_str = "ACCEPTED (Data is likely Gaussian)" if is_gaussian else "REJECTED (Data is NOT Gaussian)"

    print(f"\n--- Chi-square Test for {col_name} ---")
    print(f"Fitted Parameters: Mean={mu:.4f}, Std Dev={std:.4f}")
    print(f"Calculated Chi^2 Statistic: {chi2_stat:.4f}")
    print(f"Degrees of Freedom: {df}")
    print(f"Critical Value (\u03b1=0.05): {critical_value:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Null Hypothesis (Data is Gaussian) is: {result_str}")

    return chi2_stat, df, p_value


def plot_2d_pca_and_ellipse(df, column_names=['X', 'Y'], conf_level=0.95, filename='pca_ellipse_plot.png'):
    """
    Performs PCA on 2D data, plots the data in the new space, and adds the
    confidence ellipse (the statistical uncertainty).

    Args:
        df (pd.DataFrame): Input DataFrame with X and Y columns.
        column_names (list): The two column names to use (e.g., ['X', 'Y']).
        conf_level (float): Confidence level for the ellipse (e.g., 0.95 for 95%).
    """
    if len(column_names) != 2:
        raise ValueError("Must provide exactly two column names for 2D analysis.")

    data = df[column_names].values
    center = np.mean(data, axis=0) # Mean vector (\mu)
    cov_matrix = np.cov(data, rowvar=False) # Covariance matrix (\Sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # PCA: Eigen-decomposition

    # Sort in descending order (largest eigenvalue first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    k = 2 # Degrees of freedom for Chi-squared distribution (2D)
    c = chi2.ppf(conf_level, k) # Critical Chi-square value

    # Radii: sqrt(c * eigenvalue)
    radii = np.sqrt(c * eigenvalues)

    print("\n--- 2D PCA and Statistical Uncertainty ---")
    print(f"Mean (X, Y):\n{center}")
    print(f"Covariance Matrix (Sigma):\n{cov_matrix}")
    print(f"Principal Components (Eigenvectors):\n{eigenvectors}")
    print(f"Eigenvalues (Variance along PC):\n{eigenvalues}")
    print(f"Ellipse Radii ({conf_level*100:.0f}%):\n{radii}")
    print(f"Angle of largest PC: {np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])):.2f} degrees")

    # Angle of rotation for the ellipse (angle of the first principal component)
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Generate points on a unit circle
    t = np.linspace(0, 2 * np.pi, 100)
    xy_unit = np.array([np.cos(t), np.sin(t)])

    # Scale by radii
    xy_scaled = np.diag(radii) @ xy_unit

    # Rotate and translate
    xy_rotated = rotation_matrix @ xy_scaled
    x_ellipse = xy_rotated[0, :] + center[0]
    y_ellipse = xy_rotated[1, :] + center[1]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=20, alpha=0.6, label='Filtered End-Poses')
    plt.plot(center[0], center[1], 'ko', markersize=8, label='Mean Center')
    plt.plot(x_ellipse, y_ellipse, 'r-', linewidth=2, label=f'{conf_level*100:.1f}% Uncertainty Ellipse')

    # Plot Principal Axes (scaled by the radii for visualization)
    for i in range(k):
        v = eigenvectors[:, i] * radii[i]
        plt.plot([center[0], center[0] + v[0]], [center[1], center[1] + v[1]],
                 'g--' if i == 0 else 'm--', linewidth=1.5,
                 label=f'PC {i+1} Axis' if i == 0 else None)

    plt.title(f'PCA and Uncertainty Ellipse ({conf_level*100:.1f}% Confidence)')
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.savefig(f'figures/{filename}', dpi=300)
    plt.close()

    # Return covariance matrix for comparison with model uncertainty
    return cov_matrix, center

def plot_ellipsoid(a_vec, b_vec, c_vec):
    # 1. Define the Ellipsoid Vectors
    # NOTE: Replace these sample vectors with your actual vector data.
    # The code will extract the magnitudes A, B, C.
    # For this example, A=3, B=2, C=1.

    # Extract semi-axis lengths (A, B, C) from the vector magnitudes
    A = np.linalg.norm(a_vec)
    B = np.linalg.norm(b_vec)
    C = np.linalg.norm(c_vec)

    # 2. Create Grid and Calculate Depth
    # Define the range for the 2D plot (plane of a and b)
    # We use 300 points for a smooth image
    points = 300
    x_range = np.linspace(-A, A, points)
    y_range = np.linspace(-B, B, points)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the fractional squared sum of the (x, y) coordinates
    # This is the left-hand side of the projection ellipse equation: (x/A)^2 + (y/B)^2
    frac_sum_sq = (X/A)**2 + (Y/B)**2

    # Initialize the depth matrix Z
    Z = np.zeros_like(X)

    # Find the points inside the ellipse projection (where the sum is <= 1)
    inside = frac_sum_sq <= 1

    # Calculate the depth Z = C * sqrt(1 - (x/A)^2 - (y/B)^2)
    # Use np.maximum(0, ...) to clamp tiny negative values to zero, preventing the sqrt warning.
    depth_arg = np.maximum(0, 1 - frac_sum_sq) 

    Z = np.where(inside, C * np.sqrt(depth_arg), np.nan)

    # 3. Plot the Heatmap
    plt.figure(figsize=(8, 8))

    # Use imshow to display the Z matrix as a heatmap
    plt.imshow(Z, origin='lower', extent=[-A, A, -B, B], cmap='viridis',
            interpolation='nearest')

    # Add a color bar to show the depth scale
    cbar = plt.colorbar(label='Depth along $\\mathbf{c}$ $|z|$')

    # Set labels for the axes (aligned with vectors a and b)
    plt.xlabel('Component along $\\mathbf{a}$ ($x$)')
    plt.ylabel('Component along $\\mathbf{b}$ ($y$)')
    plt.title(f'Ellipsoid Projection (A={A:.2f}, B={B:.2f}, C={C:.2f}) with Depth Heatmap')
    # Ensure the aspect ratio is correct for the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('ellipsoid_projection_heatmap.png')

if __name__ == '__main__':
    a_vec = np.array([3, 3, 0])
    b_vec = np.array([-1, 1, 0])
    c_vec = np.array([0, 0, 1])

    plot_ellipsoid(a_vec, b_vec, c_vec)