import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from mpl_toolkits.mplot3d import Axes3D
import itertools

# --- Helper Functions from Original Code ---

def plot_3d_gaussian_ellipsoid(df, column_names, confidence_levels=[0.95, 0.6827]):
    """
    Analyzes 3D continuous data from a Pandas DataFrame, fits a Trivariate
    Normal distribution, and plots the data with confidence ellipsoids.
    (Kept for reference, though 2D is likely needed for the assignment)
    """

    # 1. Prepare and Check Data
    if len(column_names) != 3:
        raise ValueError("Must provide exactly three column names for 3D analysis.")

    data = df[column_names].values
    n_points = data.shape[0]
    if n_points < 4:
        raise ValueError("Requires at least 4 data points for covariance calculation.")

    # 2. Calculate Trivariate Normal Parameters
    center = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
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

    plt.savefig('stat_plot_3d.png')
    plt.close()


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

# ----------------------------------------------------
#               EXAMPLE USAGE (Removed/Modified)
# ----------------------------------------------------
# (The original example usage is removed to keep the file as a module of functions.)