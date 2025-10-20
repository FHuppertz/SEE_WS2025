import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Functions import End_Pose, Cov


encoder_file_path = './data/robot_path.csv'
manual_file_path = './data/manual_measurements.csv'
manual_file_path2 = './data/rahul_measurements.csv'
manual_file_path3 = './data/bhavesh_measurements.csv'

try:
    encoder_df = pd.read_csv(encoder_file_path, sep=r'\s+', header=None)

    manual_df = pd.read_csv(manual_file_path, header=None)
    manual_df2 = pd.read_csv(manual_file_path2, header=None)
    manual_df3 = pd.read_csv(manual_file_path3)
except FileNotFoundError:
    print(f"Error: File was not found. Please double-check the path.")
    # Exiting the function if the file cannot be loaded
    exit()


# Rename columns for clarity (matching the data structure: X, Y, Angle)
encoder_df.columns = ['Y', 'X', 'Angle']
encoder_df['X'] *= -1.0
encoder_df['Y'] *= -1.0
encoder_df['Angle'] *= -1.0
encoder_df['Angle'] += np.pi/2

#print(encoder_df['Angle'])

manual_df.columns = ['Lx', 'Ly', 'Rx', 'Ry']
manual_df[['X', 'Y', 'Angle']] = manual_df.apply(End_Pose, axis=1, result_type='expand')

manual_df2.columns = ['Lx', 'Ly', 'Rx', 'Ry']
manual_df2['Lx'] -= 6.9
manual_df2['Ly'] += 4.75
manual_df2['Rx'] -= 6.9
manual_df2['Ry'] += 4.75
manual_df2[['X', 'Y', 'Angle']] = manual_df2.apply(End_Pose, axis=1, result_type='expand')

manual_df3['Theta'] *= np.pi/180

# Plot the results
plt.figure()

angles = [
    encoder_df['Angle'].min(),
    encoder_df['Angle'].max(),
    manual_df['Angle'].min(),
    manual_df['Angle'].max(),
    manual_df2['Angle'].min(),
    manual_df2['Angle'].max(),
    manual_df3['Theta'].min(),
    manual_df3['Theta'].max(),
]

max_angle = max(angles)
min_angle = min(angles)

manual_scatter = plt.scatter(
    manual_df['Y'],
    manual_df['X'],
    label='End Poses Team 1',
    marker='x',
    s=20,
    alpha=0.5,
    c=manual_df['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

manual_scatter2 = plt.scatter(
    manual_df2['Y'],
    manual_df2['X'],
    label='End Poses Team 2',
    marker='s',
    s=20,
    alpha=0.5,
    c=manual_df2['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

manual_scatter3 = plt.scatter(
    manual_df3['Y'],
    manual_df3['X'],
    label='End Poses Team 3',
    marker='^',
    s=20,
    alpha=0.5,
    c=manual_df3['Theta'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

manual_scatter3 = plt.scatter(
    manual_df3['Y'],
    manual_df3['X'],
    label='End Poses Team 4',
    marker='^',
    s=20,
    alpha=0.5,
    c=manual_df3['Theta'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

plt.title('Manual measured robot end pose')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Angle (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_angle, max_angle, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('figures/all_pose_scatter_plot.png', dpi=500)
plt.close()


plt.figure()

manual_scatter = plt.scatter(
    manual_df['Y'],
    manual_df['X'],
    label='Manual End Poses',
    marker='x',
    s=20,
    alpha=0.5,
    c=manual_df['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

encoder_scatter = plt.scatter(
    encoder_df['Y'],
    encoder_df['X'],
    label='Encoder Data',
    #marker='.',
    s=0.1,
    alpha=0.5,
    c=encoder_df['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)


plt.title('Robot end pose and path (Team 1)')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Angle (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_angle, max_angle, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('figures/team1_pose_scatter_plot.png', dpi=500)
plt.close()


print('Finished plotting')


# ====================================================
#               ASSIGNMENT 3 ANALYSIS STEPS
# ====================================================

from stat_analysis import chebyshev_outlier_removal, chi_square_gaussian_test, plot_2d_pca_and_ellipse
from Functions import End_Pose, Cov # Cov is needed for step 6

# --- 1. Combine All Manual Data for Statistical Analysis (Good Practice) ---
# Assuming you want to analyze all combined manual measurements
# You must ensure the coordinate frames (X, Y, Angle) are consistent across all manual_df
combined_df = pd.concat([
    manual_df[['X', 'Y', 'Angle']],
    manual_df2[['X', 'Y', 'Angle']],
    manual_df3[['X', 'Y', 'Theta']].rename(columns={'Theta': 'Angle'})
], ignore_index=True)

print('\n' + '='*50)
print('STARTING STATISTICAL ANALYSIS')
print('='*50)

# --- 2. Compare Manual vs. Encoder End-Poses (Assignment Point 2) ---
# Determine the estimated final pose from the encoder data (last point)
encoder_end_pose = encoder_df.iloc[-1][['X', 'Y', 'Angle']].values

# Calculate mean of manual end-poses
manual_mean_pose = combined_df[['X', 'Y', 'Angle']].mean().values

# Calculate the difference (Error)
error_vector = manual_mean_pose - encoder_end_pose
error_vector_df = pd.DataFrame([error_vector], columns=['Error_X', 'Error_Y', 'Error_Angle'])

print("\n--- Quantitative Comparison (Encoder vs. Manual Mean) ---")
print(f"Encoder End Pose (X, Y, Theta):\n{encoder_end_pose}")
print(f"Mean Manual End Pose (X, Y, Theta):\n{manual_mean_pose}")
print(f"Difference (Manual Mean - Encoder):\n{error_vector_df}")
# You should discuss causes for the magnitude of these errors (e.g., slippage, model error).


# --- 3. Outlier Removal using Chebyshev (Assignment Point 3, Deliverable 1) ---
# Use X, Y, and Angle for outlier detection
filtered_df, outliers_df = chebyshev_outlier_removal(
    combined_df,
    column_names=['X', 'Y', 'Angle'],
    threshold_sigma=2.0 # Chebyshev 2-sigma threshold
)

# --- 4. Gaussian Fit and Chi-square Test (Assignment Point 4, Deliverable 2 & 3) ---
# Fit individual 1D Gaussians and test for X and Y components.

print("\n--- Gaussian Fit and Chi-square Test Results ---")
for col in ['X', 'Y', 'Angle']:
    chi2_stat, df, p_value = chi_square_gaussian_test(
        filtered_df[col].values,
        col_name=col,
        num_bins=10,
        filename='chi_square_test'
    )

# --- 5. PCA and Uncertainty Ellipses (Assignment Point 5) ---
# PCA is typically performed on the 2D position (X, Y)
# The function returns the Observed Covariance Matrix (C_obs) and Mean
C_obs, mean_obs = plot_2d_pca_and_ellipse(
    filtered_df,
    column_names=['X', 'Y'],
    conf_level=0.95,
    filename='pca_and_ellipse_2d_plot'
)


# --- 6. Compare Uncertainty (Model vs. Statistical) (Assignment Point 6) ---

# a. Calculate Model Uncertainty (C_model) using Covariance Propagation
# We use the mean of the input variables (Lx, Ly, Rx, Ry) to evaluate the Jacobian J
# Note: A and T are assumed constant (6.9, 4.75) as per Functions.py
mean_input_data = pd.concat([manual_df, manual_df2])[['Lx', 'Ly', 'Rx', 'Ry']].mean()
# Add A and T as fixed values to the dictionary for input to the Cov function
input_for_cov = mean_input_data.to_dict()
input_for_cov['A'] = 6.9
input_for_cov['T'] = 4.75

# Call the Cov function (returns a flattened 3x3 matrix)
C_model_flat = Cov(pd.Series(input_for_cov)).flatten()
C_model_full = C_model_flat.reshape((3, 3))

# Extract the 2x2 position component of the model covariance
# Assuming the order is X, Y, Angle in the C_F matrix from Functions.py
C_model_pos = C_model_full[:2, :2]

print("\n--- Uncertainty Comparison (Model vs. Observed) ---")
print(f"\nObserved Statistical Covariance (C_obs) of (X, Y) end-poses:\n{C_obs}")
print(f"\nModel Propagated Covariance (C_model) of (X, Y):\n{C_model_pos}")

# Compare variances (diagonal elements) and the overall structure/magnitude
print("\nVariance Comparison:")
print(f"X: Observed (\u03c3\u00b2)={C_obs[0, 0]:.4f} vs. Model (\u03c3\u00b2)={C_model_pos[0, 0]:.4f}")
print(f"Y: Observed (\u03c3\u00b2)={C_obs[1, 1]:.4f} vs. Model (\u03c3\u00b2)={C_model_pos[1, 1]:.4f}")
# The report should discuss the difference (e.g., C_obs > C_model due to non-modeled robot behavior).

print('\n' + '='*50)
print('ANALYSIS COMPLETE. CHECK "figures" FOLDER FOR PLOTS.')
print('='*50)




#plot after outlier removed
original_team1_indices = manual_df.index
manual_df_filtered, _ = chebyshev_outlier_removal(
    manual_df,
    column_names=['X', 'Y', 'Angle'],
    threshold_sigma=2.0
)

plt.figure()

manual_filtered_scatter = plt.scatter(
    manual_df_filtered['Y'],
    manual_df_filtered['X'],
    label='Manual End Poses (Outliers Removed)',
    marker='x',
    s=20,
    alpha=0.7,
    c=manual_df_filtered['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

encoder_scatter = plt.scatter(
    encoder_df['Y'],
    encoder_df['X'],
    label='Encoder Data Path',
    s=0.1,
    alpha=0.5,
    c=encoder_df['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)


plt.title('Robot end pose and path (Team 1) - Outliers Removed')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_filtered_scatter)
cbar.set_label('Angle (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_angle, max_angle, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('figures/team1_filtered_pose_scatter_plot.png', dpi=500)
plt.close()
