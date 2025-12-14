import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Functions import End_Pose, Cov
from stat_analysis import plot_ellipsoid_pca_fit, chebyshev_outlier_removal, chi_square_gaussian_test

manual_file_path1 = '../data/group1/manual_measurements.csv'
encoder_file_path1 = '../data/group1/robot_path.csv'

manual_file_path2 = '../data/group2/calculated_translated_readings.csv'
encoder_file_path2 = '../data/group2/robot_path.csv'

manual_file_path3 = '../data/group3/manual_measurements_new.csv'
encoder_file_path3 = '../data/group3/combined_trajectories.csv'

manual_file_path4 = '../data/group4/Robot_End_Poses_Manual.csv'
encoder_file_path4 = '../data/group4/robot_poses.csv'


manual_df1 = pd.read_csv(manual_file_path1, header=None)
encoder_df1 = pd.read_csv(encoder_file_path1, sep=r'\s+', header=None)

manual_df2 = pd.read_csv(manual_file_path2, header=None)
encoder_df2 = pd.read_csv(encoder_file_path2, sep=r'\s+', header=None)

manual_df3 = pd.read_csv(manual_file_path3, header=None)
encoder_df3 = pd.read_csv(encoder_file_path3, header=None)

manual_df4 = pd.read_csv(manual_file_path4, header=None)
encoder_df4 = pd.read_csv(encoder_file_path4, sep=r'\s+', header=None)

# Rename columns for clarity (matching the data structure: X, Y, Theta)
encoder_df1.columns = ['Y', 'X', 'Theta']
encoder_df1['X'] *= -1.0
encoder_df1['Y'] *= -1.0
encoder_df1['Theta'] *= -1.0
encoder_df1['Theta'] += np.pi/2
manual_df1.columns = ['Lx', 'Ly', 'Rx', 'Ry']
manual_df1[['X', 'Y', 'Theta']] = manual_df1.apply(End_Pose, axis=1, result_type='expand')
manual_df1.to_csv('../data/group1/manual_end_poses_calc.csv', index=False)


manual_df2.columns = ['X', 'Y', 'Theta']
manual_df2['Theta'] *= np.pi/180
encoder_df2.columns = ['Y', 'X', 'Theta']
encoder_df2['X'] *= -1.0
encoder_df2['Y'] *= -1.0
encoder_df2['Theta'] *= -1.0
encoder_df2['Theta'] += np.pi/2

manual_df3.columns = ['X', 'Y', 'Theta']
manual_df3['Theta'] *= np.pi/180
encoder_df3.columns = ['Y', 'X', 'Theta']
encoder_df3['X'] *= -1.0
encoder_df3['Y'] *= -1.0
encoder_df3['Theta'] *= -1.0
encoder_df3['Theta'] += np.pi/2

manual_df4.columns = ['Y', 'X', 'Theta']
manual_df4['Theta'] *= np.pi/180
encoder_df4.columns = ['Y', 'X', 'Theta']
encoder_df4['Theta'] -= np.pi/2

#'''
# Plot the results
plt.figure()

Thetas = [
    encoder_df1['Theta'].min(),
    encoder_df1['Theta'].max(),
    manual_df1['Theta'].min(),
    manual_df1['Theta'].max(),
    encoder_df2['Theta'].min(),
    encoder_df2['Theta'].max(),
    manual_df2['Theta'].min(),
    manual_df2['Theta'].max(),
]

max_Theta = max(Thetas)
min_Theta = min(Thetas)

manual_scatter = plt.scatter(
    manual_df1['Y'],
    manual_df1['X'],
    label='End Poses Team 1',
    marker='x',
    s=20,
    alpha=0.5,
    c=manual_df1['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

manual_scatter2 = plt.scatter(
    manual_df2['Y'],
    manual_df2['X'],
    label='End Poses Team 2',
    marker='s',
    s=20,
    alpha=0.5,
    c=manual_df2['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

manual_scatter3 = plt.scatter(
    manual_df3['Y'],
    manual_df3['X'],
    label='End Poses Team 3',
    marker='o',
    s=20,
    alpha=0.5,
    c=manual_df3['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

manual_scatter4 = plt.scatter(
    manual_df4['Y'],
    manual_df4['X'],
    label='End Poses Team 4',
    marker='^',
    s=20,
    alpha=0.5,
    c=manual_df4['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)


plt.title('Pen measured robot end pose')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Theta (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_Theta, max_Theta, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('../figures/all_pose_scatter_plot.png', dpi=500)
plt.close()


# TEAM 1
plt.figure()

manual_scatter = plt.scatter(
    manual_df1['Y'],
    manual_df1['X'],
    label='Manual End Poses',
    marker='x',
    s=20,
    alpha=0.5,
    c=manual_df1['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

encoder_scatter = plt.scatter(
    encoder_df1['Y'],
    encoder_df1['X'],
    label='Encoder Data',
    #marker='.',
    s=0.1,
    alpha=0.5,
    c=encoder_df1['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)


plt.title('Robot end pose and path (Team 1)')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Theta (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_Theta, max_Theta, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('../figures/team1_pose_scatter_plot.png', dpi=500)
plt.close()



# TEAM 2
plt.figure()

manual_scatter = plt.scatter(
    manual_df2['Y'],
    manual_df2['X'],
    label='Manual End Poses',
    marker='s',
    s=20,
    alpha=0.5,
    c=manual_df2['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

encoder_scatter = plt.scatter(
    encoder_df2['Y'],
    encoder_df2['X'],
    label='Encoder Data',
    #marker='.',
    s=0.1,
    alpha=0.5,
    c=encoder_df2['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)


plt.title('Robot end pose and path (Team 2)')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Theta (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_Theta, max_Theta, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('../figures/team2_pose_scatter_plot.png', dpi=500)
plt.close()


# TEAM 3
plt.figure()

manual_scatter = plt.scatter(
    manual_df3['Y'],
    manual_df3['X'],
    label='Manual End Poses',
    marker='o',
    s=20,
    alpha=0.5,
    c=manual_df3['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

encoder_scatter = plt.scatter(
    encoder_df3['Y'],
    encoder_df3['X'],
    label='Encoder Data',
    #marker='.',
    s=0.1,
    alpha=0.5,
    c=encoder_df3['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)


plt.title('Robot end pose and path (Team 3)')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Theta (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_Theta, max_Theta, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('../figures/team3_pose_scatter_plot.png', dpi=500)
plt.close()


# TEAM 4
plt.figure()

manual_scatter = plt.scatter(
    manual_df4['Y'],
    manual_df4['X'],
    label='Manual End Poses',
    marker='^',
    s=20,
    alpha=0.5,
    c=manual_df4['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)

encoder_scatter = plt.scatter(
    encoder_df4['Y'],
    encoder_df4['X'],
    label='Encoder Data',
    #marker='.',
    s=0.1,
    alpha=0.5,
    c=encoder_df4['Theta'],
    cmap='plasma',
    vmin=float(min_Theta),
    vmax=float(max_Theta)
)


plt.title('Robot end pose and path (Team 4)')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')
plt.axis('equal')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Theta (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_Theta, max_Theta, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.savefig('../figures/team4_pose_scatter_plot.png', dpi=500)
plt.close()

print('Finished plotting')




# ====================================================
#               ASSIGNMENT 3 ANALYSIS STEPS
# ====================================================

raw_left_df1 = manual_df1[manual_df1['Theta'] < -0.5]
raw_straight_df1 = manual_df1[(manual_df1['Theta'] >= -0.5) & (manual_df1['Theta'] < 0.5)]
raw_right_df1 = manual_df1[manual_df1['Theta'] >= 0.5]

col_names = ['X', 'Y', 'Theta']
left_df1, _ = chebyshev_outlier_removal(raw_left_df1, col_names, threshold_sigma=2.0)
straight_df1, _ = chebyshev_outlier_removal(raw_straight_df1, col_names, threshold_sigma=2.0)
right_df1, _ = chebyshev_outlier_removal(raw_right_df1, col_names, threshold_sigma=2.0)


plot_ellipsoid_pca_fit(left_df1, "Left", 2)
plot_ellipsoid_pca_fit(straight_df1, "Straight", 2)
plot_ellipsoid_pca_fit(right_df1, "Right", 2)


for name in col_names:
    chi_square_gaussian_test(left_df1[name], col_name=name, num_bins=10, direction='Left')
    chi_square_gaussian_test(straight_df1[name], col_name=name, num_bins=10, direction='Straight')
    chi_square_gaussian_test(right_df1[name], col_name=name, num_bins=10, direction='Right')


results = []

# Loop through each motion direction and its corresponding dataframe
for direction, df in [('Left', left_df1), ('Right', right_df1), ('Straight', straight_df1)]:
    for axis in ['X', 'Y', 'Theta']:
        res = chi_square_gaussian_test(
            data=df[axis].values,
            col_name=axis,
            num_bins=10,
            direction=direction
        )
        if res is not None:
            results.append(res)

# Combine all results into one summary DataFrame
chi_summary_df = pd.DataFrame(results)
print(chi_summary_df)

# Save to a single summary file (with all directions)
chi_summary_df.to_csv('../data/chi_square_summary.csv', index=False)

