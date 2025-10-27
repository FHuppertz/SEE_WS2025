import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

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


def chi_square_gaussian_test(data, col_name, num_bins=10, filename='chi_square_histogram.png', direction='Left'):
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

    # 5. Conclusion
    is_gaussian = p_value > alpha # Null hypothesis is that data is Gaussian
    result_str = "ACCEPTED (Data is likely Gaussian)" if is_gaussian else "REJECTED (Data is NOT Gaussian)"

    print(f"\n--- Chi-square Test for {col_name} ---")
    print(f"Fitted Parameters: Mean={mu:.4f}, Std Dev={std:.4f}")
    print(f"Calculated Chi^2 Statistic: {chi2_stat:.4f}")
    print(f"Degrees of Freedom: {df}")
    print(f"Critical Value (\u03b1=0.05): {critical_value:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Null Hypothesis (Data is Gaussian) is: {result_str}")

    # 6. Plotting (Histogram and PDF)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=num_bins, density=True, alpha=0.6, label=f'Data: {col_name}')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf, 'r-', linewidth=2, label='Fitted Gaussian PDF')
    plt.title(f'Gaussian Fit ($H_0$: {is_gaussian}) & $\chi^2$ Test for {direction} Direction ({col_name}-Axis)')
    plt.xlabel(col_name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'../figures/chi_square_{direction}_{col_name}.png', dpi=300)
    plt.close()



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
    plt.savefig(f'../figures/{filename}', dpi=300)
    plt.close()

    # Return covariance matrix for comparison with model uncertainty
    return cov_matrix, center

def plot_ellipsoid_pca_fit(df, df_name, sigma_multiplier=2, x_col='X', y_col='Y', z_col='Theta'):
    """
    Performs PCA on 3D data (X, Y, Theta) to define a best-fit ellipsoid.
    Plots the 2D projection of the ellipsoid onto the Y-X plane (heatmap for depth)
    and plots the original data points colored by their inclusion status.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        sigma_multiplier (int/float): Multiplier for the standard deviation 
                                      to define the size of the ellipsoid (e.g., 3 for ~99.7% confidence).
        x_col, y_col, z_col (str): Column names for the 3D data.
    """

    # --- 1. Data Preparation and PCA ---
    
    # Extract the data matrix and center the data
    X_data = df[[x_col, y_col, z_col]].values
    data_mean = X_data.mean(axis=0)
    X_centered = X_data - data_mean

    # Initialize and run PCA
    pca = PCA(n_components=3)
    pca.fit(X_centered)

    # Extract components and variances
    v1, v2, v3 = pca.components_ 

    # Calculate semi-axis lengths (A, B, C)
    A = sigma_multiplier * np.sqrt(pca.explained_variance_[0])
    B = sigma_multiplier * np.sqrt(pca.explained_variance_[1])
    C = sigma_multiplier * np.sqrt(pca.explained_variance_[2])

    # The final vectors a, b, and c for the ellipsoid
    a_vec = A * v1
    b_vec = B * v2
    c_vec = C * v3

    # Print results (optional)
    print("--- PCA Results for Ellipsoid ---")
    print(f"Center Point (Mean): {data_mean}")
    print(f"Lengths ({sigma_multiplier}-sigma): A={A:.2f}, B={B:.2f}, C={C:.2f}")

    # --- 2. Ellipsoid Surface Generation and Transformation ---

    # Define the Rotation Matrix (R) using the normalized principal components
    R = np.vstack([v1, v2, v3]).T 

    # Generate Ellipsoid Points in Canonical (Axis-Aligned) Frame
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)

    X_prime = A * np.cos(U) * np.sin(V)
    Y_prime = B * np.sin(U) * np.sin(V)
    Z_prime = C * np.cos(V)

    P_prime = np.array([X_prime.flatten(), Y_prime.flatten(), Z_prime.flatten()])

    # Rotate and Translate to World Frame
    P_world_centered = R @ P_prime
    P_world = P_world_centered + data_mean[:, np.newaxis]

    # Extract World Coordinates (P_world[0]=X, P_world[1]=Y, P_world[2]=Theta)
    X_world = P_world[0, :].reshape(X_prime.shape)
    Y_world = P_world[1, :].reshape(Y_prime.shape)
    # The depth information is the magnitude of the Z_prime component (along the c-axis)
    C_data = np.abs(Z_prime)

    # --- 3. Data Inclusion Check (Mahalanobis Distance) ---

    # Transform the centered data points to the Canonical (PCA) Frame
    X_prime_data = X_centered @ R # R transforms canonical -> world; R.T transforms world -> canonical. We need R.T

    # Correction: R is composed of column vectors (v1, v2, v3). 
    # To transform X_centered (World Frame) to X_prime_data (Canonical Frame), we use R inverse, which is R.T.
    # Since numpy's components_ are rows, R is built as [v1.T, v2.T, v3.T].T = [v1 | v2 | v3].
    # Transformation is: X_prime = X_centered @ R.T (or X_centered @ np.linalg.inv(R))
    # Note: R is built as np.vstack([v1, v2, v3]).T, so X_prime_data = X_centered @ np.linalg.inv(R) is the mathematically rigorous way, 
    # but since R is orthogonal, np.linalg.inv(R) == R.T. Let's use R.T for safety with array shapes.
    X_prime_data = X_centered @ R.T

    Xp_data = X_prime_data[:, 0]
    Yp_data = X_prime_data[:, 1]
    Zp_data = X_prime_data[:, 2]

    # Calculate Mahalanobis distance squared (D^2)
    D_sq = (Xp_data / A)**2 + (Yp_data / B)**2 + (Zp_data / C)**2

    # Determine inclusion status (True for inside, False for outside)
    is_inside = D_sq <= 1
    inclusion_status = is_inside.astype(int) # 0=Outside, 1=Inside

    # --- 4. Plotting ---

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Plot 1: Ellipsoid Heatmap (Depth) ---
    im = ax.pcolormesh(Y_world, X_world, C_data, cmap='viridis', shading='gouraud', alpha=0.6)
    
    # Add a color bar for the ellipsoid depth
    fig.colorbar(im, ax=ax, label=f'Ellipsoid Depth along $z\'$ (Max $C={C:.2f}$)', pad=0.1)


    # --- Plot 2: Scatter Plot (Inclusion Status) ---
    # Custom colormap for binary status: 0 (Outside) = Red, 1 (Inside) = Blue
    cmap_binary = ListedColormap(['red', 'blue']) 

    # Plotting Y vs X and coloring by inclusion_status
    scatter = ax.scatter(
        df[y_col], df[x_col],             # X-axis is Y, Y-axis is X (as per your custom frame)
        c=inclusion_status,               # Scalar data for coloring (0 or 1)
        cmap=cmap_binary,                 # Use the binary colormap
        marker='x',
        s=30,
        alpha=0.8
    )

    # Create custom legend handles for the binary colors
    custom_lines = [
        Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8),
        Line2D([0], [0], color='blue', marker='x', linestyle='None', markersize=8)
    ]
    
    # --- Plot 3: Center and Principal Axes ---
    
    # Add the center point (Y_mean vs X_mean)
    ax.plot(data_mean[1], data_mean[0], 'ko', markersize=8)

    # Add the principal axes (vectors a and b) projected onto the Y-X plane
    ax.quiver(data_mean[1], data_mean[0], a_vec[1], a_vec[0], 
            color='black', scale_units='xy', scale=1, width=0.005)
    ax.quiver(data_mean[1], data_mean[0], b_vec[1], b_vec[0], 
            color='darkgray', scale_units='xy', scale=1, width=0.005)

    # Final Legend
    ax.legend(
        custom_lines + [Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8),
                        Line2D([0], [0], color='black', linestyle='-', linewidth=2)],
        ['Outside Ellipsoid', 'Inside Ellipsoid', 'Ellipsoid Center', 'Principal Axes'],
        loc='best'
    )
    
    ax.set_xlabel(f'{y_col}-axis (Data)')
    ax.set_ylabel(f'{x_col}-axis (Data)')
    ax.set_title(f'PCA-Fitted Ellipsoid Projection for {df_name} Direction ({sigma_multiplier}-Sigma)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'../figures/{df_name}_ellipsoid_projection_inclusion.png')
    plt.close(fig) # Close the figure to free up memory

if __name__ == '__main__':
    pass