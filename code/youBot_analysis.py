import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def calculate_stats(data_array, source_label):
    """
    Calculates the mean and variance for the X (index 0) and Y (index 2) components 
    of the data array, assuming the format [X_plot, Theta_color, Y_plot].
    
    Returns a formatted string of the statistics.
    """
    if data_array.size == 0:
        return f"   {source_label} Stats: No data available."

    # X data is at index 0, Y data is at index 2
    X = data_array[:, 0]
    Y = data_array[:, 2]

    mean_X = np.mean(X)
    var_X = np.var(X)
    mean_Y = np.mean(Y)
    var_Y = np.var(Y)
    
    stats_str = (
        f"   {source_label} Stats:\n"
        f"     Mean X: {mean_X:.4f}, Variance X: {var_X:.4f}\n"
        f"     Mean Y: {mean_Y:.4f}, Variance Y: {var_Y:.4f}"
    )

    #print(stats_str)
    return np.array([mean_X, mean_Y]), np.array([var_X, var_Y])

# --- Function to generate Quiver Plots (Arrows Only) ---
def plot_quiver_data(ax, data_array, color, label_prefix, scale_factor=0.1):
    """
    Plots the given data array (X, Theta, Y) as a quiver plot (arrows only).
    Returns the quiver object for the legend.
    """
    if data_array.size > 0:
        X = data_array[:, 0]  # X position
        Y = data_array[:, 2]  # Y position (Z.1 in OptiTrack)
        Theta = data_array[:, 1] # Theta angle (in radians)

        # Convert polar (Theta) to Cartesian (U, V) for the arrow components
        U = np.cos(Theta)
        V = np.sin(Theta)

        # Plot the arrow (quiver)
        # scale: controls the length of the arrows. Higher scale = shorter arrows.
        # pivot='tail': centers the base of the arrow at (X, Y)
        quiv = ax.quiver(X, Y, U, V, 
                         color=color, 
                         scale=scale_factor, 
                         scale_units='xy', 
                         angles='xy',
                         width=0.005,
                         headwidth=3,
                         headlength=4,
                         label=f'{label_prefix} Orientation')

        return quiv
    else:
        # Return a dummy scatter plot for the legend entry if no data exists
        return ax.scatter([], [], color=color, label=f'{label_prefix} Orientation')

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
# FIX: Initialize all data entries with an empty NumPy array np.array([]). 
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

# New list to collect all X and Y coordinates for global min/max calculation
all_xy_values = [] 

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

                            # Get End Point
                            df_mean = df.mean()

                            # Extract the mean values for clarity
                            mean_x = df_mean[0]
                            mean_y = df_mean[1]
                            mean_theta = df_mean[2]
                            
                            
                            # Mapped structure: [X_plot (0), Theta_color (2), Y_plot (1)]
                            # Note: youBot data is scaled by 100 to match OptiTrack's cm scale
                            # youBot data provides only mean state, so we use it for both 'start' and 'end' points
                            start_point = [mean_x*100, mean_theta, mean_y*100] 
                            end_point = start_point
                            
                            theta_val_start = mean_theta
                            theta_val_end = mean_theta
                            
                        else:
                            # Should not happen based on DATA_CONFIG
                            continue

                        # Append points and theta values after successful loading and mapping
                        points_start.append(start_point)
                        points_end.append(end_point)
                        all_theta_values.append(theta_val_start)
                        all_theta_values.append(theta_val_end)

                        # --- Add X and Y values to the global accumulator ---
                        all_xy_values.append((start_point[0], start_point[2])) # Start Point: X (index 0), Y (index 2)
                        all_xy_values.append((end_point[0], end_point[2]))   # End Point: X (index 0), Y (index 2)
                        # --- End of Add X and Y values ---


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

# --- 2. Calculate Global X and Y Limits ---
if all_xy_values:
    all_xy_array = np.array(all_xy_values)
    
    # Get overall min/max for X and Y coordinates
    x_min_global = np.min(all_xy_array[:, 0])
    x_max_global = np.max(all_xy_array[:, 0])
    y_min_global = np.min(all_xy_array[:, 1])
    y_max_global = np.max(all_xy_array[:, 1])
    
    # Add a buffer/margin to the limits for better visualization
    BUFFER = 5.0 # 5 cm buffer
    X_LIM = (x_min_global - BUFFER, x_max_global + BUFFER)
    Y_LIM = (y_min_global - BUFFER, y_max_global + BUFFER)
    print(f"Global X Limits: {X_LIM}, Global Y Limits: {Y_LIM}")
else:
    # Set default limits if no data was loaded
    X_LIM = (-50, 50)
    Y_LIM = (-50, 50)
    print("No data found, using default plot limits.")
    
# --- End of Global Limit Calculation ---


# 3. Generate the 2D plots using the unified data_store
print("\n--- Generating Start/End Point 2D Quiver Plots (Arrows Only) ---")

# Define colors for clarity in the legend
COLOR_OPTI = 'darkblue'
COLOR_ROB = 'darkred'

all_start = []
for object_name in CANONICAL_OBJECTS:
    for dir_name in DIRECTIONS:
        
        # --- Plot End Points (Orientation) ---
        fig_end, ax_end = plt.subplots(figsize=(10, 8))
        
        # Access the arrays directly from the data_store
        end_opti = data_store[object_name][dir_name]['opti_end']
        end_rob = data_store[object_name][dir_name]['rob_end']

        mean_opti, var_opti = calculate_stats(end_opti, 'Opti_'+object_name+'/'+dir_name)
        mean_rob, var_rob = calculate_stats(end_rob, 'Rob_'+object_name+'/'+dir_name)

        print('DiffMean '+object_name+'/'+dir_name+':'+str(np.linalg.norm(mean_opti-mean_rob)))
        print('VarsOptiRob '+object_name+'/'+dir_name+':'+str(var_opti),str(var_rob))

        # Plot OptiTrack End Quivers
        q3 = plot_quiver_data(ax_end, end_opti, COLOR_OPTI, 'OptiTrack')

        # Plot youBot (rob) End Quivers
        q4 = plot_quiver_data(ax_end, end_rob, COLOR_ROB, 'youBot')

        # Set labels, title, and aspect ratio
        ax_end.set_xlabel('X (cm)')
        ax_end.set_ylabel('Y (cm)')
        ax_end.set_title(f'End Points of {object_name.capitalize()}/{dir_name.capitalize()}')
        ax_end.set_aspect('equal')
        ax_end.grid(True, linestyle='--', alpha=0.6)
        
        # --- APPLY UNIFORM LIMITS ---
        ax_end.set_xlim(X_LIM)
        ax_end.set_ylim(Y_LIM)
        # --- END APPLY UNIFORM LIMITS ---
        
        # Create a legend
        ax_end.legend(handles=[q3, q4], 
                        labels=['OptiTrack', 'youBot'], 
                        loc='lower left', 
                        )
        
        plt.savefig('../figures/youbot/EndPt_'+object_name +'_'+dir_name)
        plt.close(fig_end)


        all_start.append(data_store[object_name][dir_name]['opti_start'])


all_start = np.concatenate(all_start, axis=0)
# --- Plot Start Points (Orientation) ---
fig_end, ax_end = plt.subplots(figsize=(10, 8))

# Plot OptiTrack End Quivers
q = plot_quiver_data(ax_end, all_start, COLOR_OPTI, 'OptiTrack', scale_factor=5)

# Set labels, title, and aspect ratio
ax_end.set_xlabel('X (cm)')
ax_end.set_ylabel('Y (cm)')
ax_end.set_title(f'Start Points of all Trials')
ax_end.set_aspect('equal')
ax_end.grid(True, linestyle='--', alpha=0.6)

## --- APPLY UNIFORM LIMITS ---
#ax_end.set_xlim(X_LIM)
#ax_end.set_ylim(Y_LIM)
## --- END APPLY UNIFORM LIMITS ---

# Create a legend
ax_end.legend(handles=[q], 
                labels=['OptiTrack'], 
                loc='lower left', 
                )

plt.savefig('../figures/youbot/StPt_opti')
plt.close(fig_end)


print("\n--- Plotting Complete ---")