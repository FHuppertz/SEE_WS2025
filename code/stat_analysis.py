import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA
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



def plot_ellipsoid_pca_fit(df, df_name, sigma_multiplier=2, x_col='X', y_col='Y', z_col='Theta'):
    """
    Performs PCA on 3D data (X, Y, Theta) to define a best-fit ellipsoid.
    Plots a 3D view of the ellipsoid (no heatmap) and the original data points
    colored by their inclusion status, showing the principal axes as lines.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        df_name (str): Name for the dataset, used in titles and filenames.
        sigma_multiplier (int/float): Multiplier for the standard deviation
                                      to define the size of the ellipsoid.
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

    # Extract World Coordinates
    X_world = P_world[0, :].reshape(X_prime.shape)
    Y_world = P_world[1, :].reshape(Y_prime.shape)
    Theta_world = P_world[2, :].reshape(Z_prime.shape)

    # --- 3. Inclusion Check ---

    X_prime_data = R.T @ X_centered.T
    Xp_data = X_prime_data.T[:, 0]
    Yp_data = X_prime_data.T[:, 1]
    Zp_data = X_prime_data.T[:, 2]

    # Check if inside ellipsoid: (Xp/A)^2 + (Yp/B)^2 + (Zp/C)^2 <= 1
    D_sq = (Xp_data / A)**2 + (Yp_data / B)**2 + (Zp_data / C)**2
    is_inside = D_sq <= 1
    inclusion_status = is_inside.astype(int) # 0=Outside, 1=Inside

    # Custom colormap for binary status: 0 (Outside) = Red, 1 (Inside) = Blue
    cmap_binary = ListedColormap(['red', 'blue'])

    # Create custom legend handles for the binary colors
    custom_lines = [
        Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8),
        Line2D([0], [0], color='blue', marker='x', linestyle='None', markersize=8)
    ]

    # =========================================================================
    # --- 4. Plotting (3D View ONLY) ---
    # =========================================================================

    fig3d = plt.figure(figsize=(8, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')

    # a. Plot Ellipsoid Surface
    ax3d.plot_surface(
        X_world, Y_world, Theta_world,
        color='blue', alpha=0.1, linewidth=0.5, antialiased=False
    )

    # b. Plot Data Points (colored by inclusion status)
    ax3d.scatter(
        df[x_col], df[y_col], df[z_col],
        c=inclusion_status,
        cmap=cmap_binary,
        marker='x',
        s=30,
        alpha=0.8
    )

    # c. Plot Center and Principal Axes (as lines without arrowheads)

    # Center point
    ax3d.plot(data_mean[0:1], data_mean[1:2], data_mean[2:3], 'ko', markersize=8, label='Center')

    # Principal Axes (Lines drawn from -vector to +vector, passing through the mean)

    # Axis 1 (A) - Major Axis (Red)
    ax3d.plot(
        [data_mean[0], data_mean[0] + a_vec[0]], # X coordinates
        [data_mean[1], data_mean[1] + a_vec[1]], # Y coordinates
        [data_mean[2], data_mean[2] + a_vec[2]], # Z coordinates
        color='red', linewidth=2
    )

    # Axis 2 (B) - Mid Axis (Green)
    ax3d.plot(
        [data_mean[0] , data_mean[0] + b_vec[0]],
        [data_mean[1], data_mean[1] + b_vec[1]],
        [data_mean[2], data_mean[2] + b_vec[2]],
        color='green', linewidth=2
    )

    # Axis 3 (C) - Minor Axis (Black)
    ax3d.plot(
        [data_mean[0] , data_mean[0] + c_vec[0]],
        [data_mean[1], data_mean[1] + c_vec[1]],
        [data_mean[2], data_mean[2] + c_vec[2]],
        color='black', linewidth=2
    )

    # d. Set Labels, Title, and Save
    ax3d.set_xlabel(f'{x_col} (cm)')
    ax3d.set_ylabel(f'{y_col} (cm)')
    ax3d.set_zlabel(f'{z_col} (rad)')
    ax3d.set_title(f'3D PCA-Fitted Ellipsoid for {df_name} Direction ({sigma_multiplier} $\sigma$ PC length)')

    # Create custom legend for principal axes for the 3D plot
    custom_axis_lines = [
        Line2D([0], [0], color='red', linestyle='-', linewidth=2),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2)
    ]
    ax3d.legend(
        custom_lines + custom_axis_lines,
        ['Outside Ellipsoid', 'Inside Ellipsoid', 'PC 1', 'PC 2', 'PC 3'],
        loc='upper right'
    )

    plt.savefig(f'../figures/{df_name}_ellipsoid_3d_view.png')
    plt.close(fig3d)

if __name__ == '__main__':
    pass
