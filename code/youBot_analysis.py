import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --- Configuration ---
# Define the canonical set of object keys (using lowercase for internal consistency)
CANONICAL_OBJECTS = ['large', 'medium', 'small']
DIRECTIONS = ['left', 'straight', 'right']

# Define the data sources and their specific directory casing
DATA_CONFIG = [
    {
        'source_key': 'opti', # Used as the dictionary key in the data_store
        'folder_core_path': '../data/optitrack/',
        'object_names_map': {
            'large': 'large',
            'medium': 'medium',
            'small': 'small',
        },
        'label': 'OptiTrack'
    },
    {
        'source_key': 'rob', # Used as the dictionary key in the data_store
        'folder_core_path': '../data/youBot/',
        'object_names_map': {
            'large': 'large',
            'medium': 'medium',
            'small': 'small',
        },
        'label': 'youBot'
    }
]

# Initialize storage
all_theta_values = []
# Structure: data_store['large']['left']['opti_start'] -> np.array of points
# FIX: Initialize all data entries with an empty NumPy array np.array([]). 
# This ensures that even if a directory is skipped, the variable will have a .size attribute (size 0).
data_store = {
    obj: {
        d: {
            'opti_start': np.array([]), 'opti_end': np.array([]),
            'rob_start': np.array([]), 'rob_end': np.array([])
        }
        for d in DIRECTIONS
    }
    for obj in CANONICAL_OBJECTS
}

# 1. Unified Data Loading and Global Min/Max Collection
print("--- Starting Unified Data Loading and Pre-analysis ---")

for config in DATA_CONFIG:
    source = config['source_key']
    base_path = config['folder_core_path']
    name_map = config['object_names_map']

    for canonical_obj in CANONICAL_OBJECTS:
        # Get the actual folder name based on the source's casing convention
        object_folder_name = name_map[canonical_obj]

        for dir_name in DIRECTIONS:
            folder_path = os.path.join(base_path, object_folder_name, dir_name)
            
            if not os.path.isdir(folder_path):
                print(f"Directory not found: {folder_path}. Skipping.")
                continue

            # Keys for storing data in the unified data_store
            start_key = f'{source}_start'
            end_key = f'{source}_end'
            
            points_start = []
            points_end = []

            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        
                        if source == 'opti':
                            # --- OPTITRACK DATA LOADING ---
                            # Original OptiTrack logic: skiprows=7, headered columns
                            df = pd.read_csv(file_path, skiprows=7) 
                            
                            # Column names for the required data components
                            col_x_plot = 'X.1'
                            col_theta_color = 'Y'
                            col_y_plot = 'Z.1'
                            
                            if df.empty: continue
                            
                            # Check columns before access
                            if not all(col in df.columns for col in [col_x_plot, col_theta_color, col_y_plot]):
                                raise KeyError("OptiTrack CSV missing required columns (X.1, Y, Z.1).")

                            first_row = df.iloc[0]
                            last_row = df.iloc[-1]
                            
                            # Mapped structure: [X_plot, Theta_color, Y_plot]
                            start_point = [first_row[col_x_plot], first_row[col_theta_color], first_row[col_y_plot]]
                            end_point = [last_row[col_x_plot], last_row[col_theta_color], last_row[col_y_plot]]
                            
                            theta_val_start = first_row[col_theta_color]
                            theta_val_end = last_row[col_theta_color]
                            
                        elif source == 'rob':
                            # --- YOUBOT DATA LOADING ---
                            # User requirement: no header, no skiprows, columns are [X, Y, theta] (indices 0, 1, 2)
                            # We map them to the expected plotting structure: [X_plot, Theta_color, Y_plot]
                            # X (Col 0) -> X_plot
                            # Y (Col 1) -> Y_plot
                            # Theta (Col 2) -> Theta_color
                            df = pd.read_csv(file_path, header=None) 
                            
                            # Check if the DataFrame has exactly 3 columns (0, 1, 2)
                            if len(df.columns) < 3:
                                raise ValueError(f"youBot CSV expected 3 columns (X, Y, Theta) but found {len(df.columns)}.")

                            first_row = df.iloc[0]
                            last_row = df.iloc[-1]
                            
                            
                            # Mapped structure: [X_plot (0), Theta_color (2), Y_plot (1)]
                            start_point = [first_row[0], first_row[2], first_row[1]]
                            end_point = [last_row[0], last_row[2], last_row[1]]
                            
                            theta_val_start = first_row[2]
                            theta_val_end = last_row[2]
                            
                        else:
                            # Should not happen based on DATA_CONFIG
                            continue

                        # Append points and theta values after successful loading and mapping
                        points_start.append(start_point)
                        points_end.append(end_point)
                        all_theta_values.append(theta_val_start)
                        all_theta_values.append(theta_val_end)

                    except pd.errors.EmptyDataError:
                        print(f"File {filename} is empty and was skipped.")
                    except KeyError as e:
                        print(f"Missing required column in {filename}: {e}. Skipping.")
                    except ValueError as e:
                        print(f"Data processing error in {filename}: {e}. Skipping.")
                    except Exception as e:
                        print(f"An error occurred while processing {filename}: {e}")
            
            # Convert list of lists to numpy arrays for storage
            data_store[canonical_obj][dir_name][start_key] = np.array(points_start)
            data_store[canonical_obj][dir_name][end_key] = np.array(points_end)

print("--- Data Loading Complete. Preparing for Plotting. ---")

# Calculate global normalization range
if all_theta_values:
    # Ensure min/max calculation is robust even if data is sparse
    global_theta_min = np.min(all_theta_values)
    global_theta_max = np.max(all_theta_values)
    # Define the normalization object based on global min/max
    norm = Normalize(vmin=global_theta_min, vmax=global_theta_max)
else:
    # Fallback if no data was processed
    norm = Normalize(vmin=-1, vmax=1) 
    print("Warning: No data loaded. Using default color normalization.")

# Create the scalar mappable object for the color bar
sm = ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([]) # Required for ScalarMappable

# 2. Generate the 2D plots using the unified data_store
print("\n--- Generating Start/End Point 2D Plots ---")

for object_name in CANONICAL_OBJECTS:
    for dir_name in DIRECTIONS:
        # --- Plot Start Points ---
        fig_start, ax_start = plt.subplots(figsize=(10, 8))
        
        # Access the arrays directly from the data_store
        start_opti = data_store[object_name][dir_name]['opti_start']
        start_rob = data_store[object_name][dir_name]['rob_start']

        # Plot OptiTrack Start Points
        if start_opti.size > 0:
            sc1 = ax_start.scatter(start_opti[:, 0], start_opti[:, 2], # X is col 0, Z is col 2 (Y-axis in plot)
                                c=start_opti[:, 1], # Y is col 1 (Color/Theta)
                                cmap='viridis', norm=norm,
                                marker='o', s=100, label='OptiTrack Start', alpha=0.7, edgecolors='black')
        else:
            sc1 = ax_start.scatter([], [], marker='o', s=100, label='OptiTrack Start')

        # Plot youBot (rob) Start Points
        if start_rob.size > 0:
            sc2 = ax_start.scatter(start_rob[:, 0], start_rob[:, 2], # X is col 0, Z is col 2 (Y-axis in plot)
                                c=start_rob[:, 1], # Y is col 1 (Color/Theta)
                                cmap='viridis', norm=norm,
                                marker='^', s=100, label='youBot Start', alpha=0.7, edgecolors='black')
        else:
             sc2 = ax_start.scatter([], [], marker='^', s=100, label='youBot Start')

        # Add color bar
        cbar_start = fig_start.colorbar(sm, ax=ax_start, orientation='vertical', pad=0.05)
        cbar_start.set_label('Theta (rad)')
        
        # Set labels, title, and aspect ratio
        ax_start.set_xlabel('X (cm)')
        ax_start.set_ylabel('Y (cm)')
        ax_start.set_title(f'Start Points Analysis: {object_name.capitalize()} / {dir_name.capitalize()}')
        ax_start.set_aspect('equal', adjustable='box')
        ax_start.grid(True, linestyle='--', alpha=0.6)
        
        # Create a legend based on marker type (source)
        ax_start.legend(handles=[sc1, sc2], 
                        labels=['OptiTrack', 'youBot'], 
                        loc='lower left', 
                        title="Source")
        
        plt.show()

        # --- Plot End Points ---
        fig_end, ax_end = plt.subplots(figsize=(10, 8))
        
        # Access the arrays directly from the data_store
        end_opti = data_store[object_name][dir_name]['opti_end']
        end_rob = data_store[object_name][dir_name]['rob_end']

        # Plot OptiTrack End Points
        if end_opti.size > 0:
            sc3 = ax_end.scatter(end_opti[:, 0], end_opti[:, 2],
                                c=end_opti[:, 1], 
                                cmap='viridis', norm=norm,
                                marker='s', s=100, label='OptiTrack End', alpha=0.7, edgecolors='black')
        else:
            sc3 = ax_end.scatter([], [], marker='s', s=100, label='OptiTrack End')


        # Plot youBot (rob) End Points
        if end_rob.size > 0:
            sc4 = ax_end.scatter(end_rob[:, 0], end_rob[:, 2],
                                c=end_rob[:, 1], 
                                cmap='viridis', norm=norm,
                                marker='X', s=100, label='youBot End', alpha=0.7, edgecolors='black')
        else:
            sc4 = ax_end.scatter([], [], marker='X', s=100, label='youBot End')


        # Add color bar
        cbar_end = fig_end.colorbar(sm, ax=ax_end, orientation='vertical', pad=0.05)
        cbar_end.set_label('Theta (rad)')

        # Set labels, title, and aspect ratio
        ax_end.set_xlabel('X (cm)')
        ax_end.set_ylabel('Y (cm)')
        ax_end.set_title(f'End Points Analysis: {object_name.capitalize()} / {dir_name.capitalize()}')
        ax_end.set_aspect('equal', adjustable='box')
        ax_end.grid(True, linestyle='--', alpha=0.6)
        
        # Create a legend based on marker type (source)
        ax_end.legend(handles=[sc3, sc4], 
                        labels=['OptiTrack', 'youBot'], 
                        loc='lower left', 
                        title="Source")
        
        plt.show()


print("\n--- Plotting Complete ---")