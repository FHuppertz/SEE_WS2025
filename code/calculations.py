import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Functions import End_Pose, Cov


# Using the file path provided by you
encoder_file_path = './data/robot_path.csv'
manual_file_path = './data/manual_measurements.csv'

# Read the data, assuming it's space-separated and has no header.
try:
    encoder_df = pd.read_csv(encoder_file_path, sep=r'\s+', header=None)
except FileNotFoundError:
    print(f"Error: The encoder file was not found at the expected path: {encoder_file_path}. Please double-check the path.")
    # Exiting the function if the file cannot be loaded
    exit()


# Read the data, assuming it's space-separated and has no header.
try:
    manual_df = pd.read_csv(manual_file_path, header=None)
except FileNotFoundError:
    print(f"Error: The manual file was not found at the expected path: {manual_file_path}. Please double-check the path.")
    # Exiting the function if the file cannot be loaded
    exit()


# Rename columns for clarity (matching the data structure: X, Y, Angle)
encoder_df.columns = ['Y', 'X', 'Angle']
encoder_df['X'] = -encoder_df['X']
encoder_df['Y'] = -encoder_df['Y']
encoder_df['Angle'] = -encoder_df['Angle'] + np.pi/2

print(encoder_df['Angle'])

manual_df.columns = ['Lx', 'Ly', 'Rx', 'Ry']

manual_df[['X', 'Y', 'Angle']] = manual_df.apply(End_Pose, axis=1, result_type='expand')

# Plot the first column (X) against the second column (Y)
plt.figure(figsize=(10, 6))

# Plotting X vs Y, using the 'Angle' column to color-code the points for extra insight
plt.scatter(encoder_df['Y'], encoder_df['X'], label='Encoder Data', s=0.1, c=encoder_df['Angle'], cmap='plasma')
plt.scatter(manual_df['Y'], manual_df['X'], label='Manual Data', marker='x', s=20, c=manual_df['Angle'], cmap='plasma')

plt.title('Robot pose and path')
plt.xlabel('Y (cm)')
plt.ylabel('X (cm)')

# Add a color bar
cbar = plt.colorbar()
cbar.set_label('Angle (Radians)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('figures/pose_scatter_plot.png')
plt.close()

print('Finished plotting')
