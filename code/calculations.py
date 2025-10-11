import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Functions import End_Pose, Cov

## TEST
'''
# [Lx, Ly, Rx, Ry, A]
data = [10, 0, 0, 0, 10]

print(End_Pose(data))

print(Cov(data))
'''

# Using the file path provided by you
file_path = './data/robot_path.csv'

# Read the data, assuming it's space-separated and has no header.
try:
    df = pd.read_csv(file_path, sep=r'\s+', header=None, skipinitialspace=True)
except FileNotFoundError:
    print(f"Error: The file was not found at the expected path: {file_path}. Please double-check the path.")
    # Exiting the function if the file cannot be loaded
    exit()

# Rename columns for clarity (matching the data structure: X, Y, Angle)
df.columns = ['Y', 'X', 'Angle']
df['X'] = -df['X']
df['Y'] = -df['Y']

# Display the head for verification
print("--- Data Loaded ---")
print(df.head())

# Plot the first column (X) against the second column (Y)
plt.figure(figsize=(10, 6))

# Plotting X vs Y, using the 'Angle' column to color-code the points for extra insight
plt.scatter(df['Y'], df['X'], label='Robot Path Points', s=0.1, c=df['Angle'], cmap='plasma')

plt.title('Robot Path: X vs. Y (Color-coded by Angle)')
plt.xlabel('Y')
plt.ylabel('X')

# Add a color bar
cbar = plt.colorbar()
cbar.set_label('Angle (Radians)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('figures/robot_path_x_vs_y_scatter_plot.png')
plt.close()
