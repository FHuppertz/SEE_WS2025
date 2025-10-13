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
plt.figure(figsize=(10, 6))

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

encoder_scatter = plt.scatter(
    encoder_df['Y'],
    encoder_df['X'],
    label='Encoder Data',
    s=0.1,
    c=encoder_df['Angle'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

manual_scatter = plt.scatter(
    manual_df['Y'],
    manual_df['X'],
    label='Manual Data',
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
    label='Manual Data 2',
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
    label='Manual Data 2',
    marker='^',
    s=20,
    alpha=0.5,
    c=manual_df3['Theta'],
    cmap='plasma',
    vmin=float(min_angle),
    vmax=float(max_angle)
)

plt.title('Robot pose and path')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')

cbar = plt.colorbar(manual_scatter)
cbar.set_label('Angle (Radians)', rotation=270, labelpad=15)
cbar.set_ticks(list(np.linspace(min_angle, max_angle, 15)))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('figures/pose_scatter_plot.png', dpi=500)
plt.close()

print('Finished plotting')
