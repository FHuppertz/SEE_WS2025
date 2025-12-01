import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import cm # Import colormap module

# --- Helper Functions ---

def calculate_stats(data_array, source_label):
    """
    Calculates the mean and variance for the X (index 0) and Y (index 2) components 
    of the data array, assuming the format [X_plot, Theta_color, Y_plot].
    
    Returns the mean and variance numpy arrays.
    """
    if data_array.size == 0:
        # Return empty arrays for consistency when no data is available
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

    # X data is at index 0, Y data is at index 2
    X = data_array[:, 0]
    Y = data_array[:, 2]

    mean_X = np.mean(X)
    var_X = np.var(X)
    mean_Y = np.mean(Y)
    var_Y = np.var(Y)
    
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
        # Use alpha=0 to prevent drawing, but retain the object for the legend
        return ax.scatter([], [], color=color, alpha=0, label=f'{label_prefix} Orientation')

# --- Configuration ---
# Define the canonical set of object *bases*
CANONICAL_OBJECT_BASES = ['large', 'medium', 'small']
# Define the object indices (1 through 4)
OBJECT_INDICES = list(range(1, 5)) 

# Generate the new canonical object keys (e.g., 'large1', 'large2', ...)
CANONICAL_OBJECTS = [
    f'{base}{idx}' 
    for base in CANONICAL_OBJECT_BASES 
    for idx in OBJECT_INDICES
]

DIRECTIONS = ['left', 'straight', 'right']

# Define the data sources and their specific directory casing
DATA_CONFIG = [
    {
        'source_key': 'opti', # Used as the dictionary key in the data_store
        'folder_core_path': '../data/optitrack/',
        # Map the canonical key ('large1') to the folder name ('large1')
        'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS},
        'label': 'OptiTrack'
    },
    {
        'source_key': 'rob', # Used as the dictionary key in the data_store
        'folder_core_path': '../data/youBot/',
        # Map the canonical key ('large1') to the folder name ('large1')
        'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS},
        'label': 'youBot'
    }
]

# --- Color Mapping Setup ---
cmap_base = cm.get_cmap('inferno') # Good for distinct categories
# Normalizer scales the indices (1-4) to (0.0-1.0)
norm_indices = Normalize(vmin=min(OBJECT_INDICES), vmax=max(OBJECT_INDICES))

def get_object_color(canonical_obj_key):
    """Retrieves a unique color based on the object's base name and index."""
    for base in CANONICAL_OBJECT_BASES:
        if canonical_obj_key.startswith(base):
            # Extract the index number
            index_str = canonical_obj_key.replace(base, '')
            try:
                index = int(index_str)
                # Map the index to a color from the colormap
                return cmap_base(norm_indices(index))
            except ValueError:
                return 'gray'
    return 'black' # Default fallback


# Initialize storage using the new canonical keys
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

    for canonical_obj in CANONICAL_OBJECTS: # Iterate over the new list ('large1', 'large2', ...)
        object_folder_name = name_map[canonical_obj]

        for dir_name in DIRECTIONS:
            # The folder path is now base_path / large1 / left
            folder_path = os.path.join(base_path, object_folder_name, dir_name)
            
            if not os.path.isdir(folder_path):
                # print(f"Directory not found: {folder_path}. Skipping.")
                continue

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
                            df = pd.read_csv(file_path, skiprows=7) 
                            
                            col_x_plot = 'X.1'
                            col_theta_color = 'Y'
                            col_y_plot = 'Z.1'
                            
                            if df.empty: continue
                            
                            if not all(col in df.columns for col in [col_x_plot, col_theta_color, col_y_plot]):
                                raise KeyError("OptiTrack CSV missing required columns (X.1, Y, Z.1).")

                            first_row = df.iloc[0]
                            last_row = df.iloc[-1]
                            
                            # Mapped structure: [X_plot, Theta_color, Y_plot]
                            start_point = [first_row[col_x_plot], first_row[col_theta_color], first_row[col_y_plot]]
                            end_point = [last_row[col_x_plot], last_row[col_theta_color], last_row[col_y_plot]]
                            
                        elif source == 'rob':
                            # --- YOUBOT DATA LOADING ---
                            df = pd.read_csv(file_path, header=None) 
                            
                            if len(df.columns) < 3:
                                raise ValueError(f"youBot CSV expected 3 columns (X, Y, Theta) but found {len(df.columns)}.")

                            df_mean = df.mean()

                            mean_x = df_mean[0]
                            mean_y = df_mean[1]
                            mean_theta = df_mean[2]
                            
                            # Mapped structure: [X_plot (0), Theta_color (2), Y_plot (1)]
                            # youBot data is scaled by 100 
                            start_point = [mean_x*100, mean_theta, mean_y*100] 
                            end_point = start_point
                            
                        else:
                            continue

                        # Append points
                        points_start.append(start_point)
                        points_end.append(end_point)
                        all_xy_values.append((start_point[0], start_point[2])) # Start Point: X (index 0), Y (index 2)
                        all_xy_values.append((end_point[0], end_point[2]))   # End Point: X (index 0), Y (index 2)

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
    
    x_min_global = np.min(all_xy_array[:, 0])
    x_max_global = np.max(all_xy_array[:, 0])
    y_min_global = np.min(all_xy_array[:, 1])
    y_max_global = np.max(all_xy_array[:, 1])
    
    BUFFER = 5.0 # 5 cm buffer
    X_LIM = (x_min_global - BUFFER, x_max_global + BUFFER)
    Y_LIM = (y_min_global - BUFFER, y_max_global + BUFFER)
    print(f"Global X Limits: {X_LIM}, Global Y Limits: {Y_LIM}")
else:
    X_LIM = (-50, 50)
    Y_LIM = (-50, 50)
    print("No data found, using default plot limits.")
    
# --- End of Global Limit Calculation ---

# 3. Generate the 2D plots using the unified data_store (MODIFIED)
print("\n--- Generating Grouped End Point 2D Quiver Plots ---")

# Define base colors/alpha for OptiTrack and youBot source differentiation
COLOR_OPTI_BASE = 0.8 # Transparency/alpha for OptiTrack
COLOR_ROB_BASE = 1.0 # Transparency/alpha for youBot

# Iterate over the canonical object BASES (large, medium, small)
for object_base in CANONICAL_OBJECT_BASES:
    
    # Iterate over the DIRECTIONS (left, straight, right)
    for dir_name in DIRECTIONS:
        
        fig_end, ax_end = plt.subplots(figsize=(10, 6))
        
        # Store handles and labels for this specific plot
        current_handles = {}
        
        # Iterate over the numbered object indices (1, 2, 3, 4)
        for index in OBJECT_INDICES:
            
            # Construct the full canonical object key (e.g., 'large1')
            canonical_obj = f'{object_base}{index}'
            
            # Get the unique color for this object index (e.g., color for '1')
            dynamic_color = get_object_color(canonical_obj)
            
            # Access the arrays directly from the data_store
            end_opti = data_store[canonical_obj][dir_name]['opti_end']
            end_rob = data_store[canonical_obj][dir_name]['rob_end']

            # --- Calculate Stats ---
            mean_opti, var_opti = calculate_stats(end_opti, f'Opti_{canonical_obj}/{dir_name}')
            mean_rob, var_rob = calculate_stats(end_rob, f'Rob_{canonical_obj}/{dir_name}')

            print(f'DiffMean {canonical_obj}/{dir_name}: {np.linalg.norm(mean_opti-mean_rob) if not np.isnan(mean_opti[0]) else "NaN"}')
            print(f'VarsOptiRob {canonical_obj}/{dir_name}: {var_opti} {var_rob}')

            # --- Plotting ---
            
            # 1. Plot OptiTrack End Quivers (with transparency)
            opti_color = list(dynamic_color[:3]) + [COLOR_OPTI_BASE]
            q3 = plot_quiver_data(ax_end, end_opti, opti_color, f'OptiTrack {canonical_obj}')
            
            # 2. Plot youBot (rob) End Quivers (full color)
            rob_color = list(dynamic_color[:3]) + [COLOR_ROB_BASE]
            q4 = plot_quiver_data(ax_end, end_rob, rob_color, f'youBot {canonical_obj}')
            
            # --- Legend Management ---
            # Add handles for this specific plot.
            key_opti = f'OptiTrack - {canonical_obj}'
            if key_opti not in current_handles:
                 current_handles[key_opti] = q3
            
            key_rob = f'youBot - {canonical_obj}'
            if key_rob not in current_handles:
                 current_handles[key_rob] = q4


        # Set labels, title, and aspect ratio
        ax_end.set_xlabel('X (cm)')
        ax_end.set_ylabel('Y (cm)')
        ax_end.set_title(f'End Points of {object_base.capitalize()} Objects on {dir_name.capitalize()} Trajectories')
        ax_end.set_aspect('equal')
        ax_end.grid(True, linestyle='--', alpha=0.6)
        
        # Apply uniform limits
        ax_end.set_xlim(X_LIM)
        ax_end.set_ylim(Y_LIM)
        
        # Create the legend
        sorted_items = sorted(current_handles.items())
        handles = [item[1] for item in sorted_items]
        labels = [item[0] for item in sorted_items]

        ax_end.legend(handles=handles, 
                      labels=labels, 
                      loc='lower left', 
                      ncol=2, 
                      fontsize='small')
        
        # Save the figure using the object base name and direction name
        plt.savefig(f'../figures/youbot/EndPt_{object_base}_{dir_name}')
        plt.close(fig_end)

# --- Consolidated Start Points Plot (Kept from original) ---
print("\n--- Generating Consolidated Start Points Plot ---")

all_start = np.concatenate(
    [data_store[obj][dir_name]['opti_start'] 
     for obj in CANONICAL_OBJECTS 
     for dir_name in DIRECTIONS], 
    axis=0
)

fig_start, ax_start = plt.subplots(figsize=(10, 8))

# Plot OptiTrack Start Quivers (all in one color for simplicity, as per original code)
# Using a larger scale factor since it's a dense plot
q = plot_quiver_data(ax_start, all_start, 'green', 'All OptiTrack Starts', scale_factor=5)

ax_start.set_xlabel('X (cm)')
ax_start.set_ylabel('Y (cm)')
ax_start.set_title(f'Start Points of all Trials (OptiTrack)')
ax_start.set_aspect('equal')
ax_start.grid(True, linestyle='--', alpha=0.6)

# Apply uniform limits (optional for start points, but often helpful)
# ax_start.set_xlim(X_LIM)
# ax_start.set_ylim(Y_LIM)

ax_start.legend(handles=[q], 
                labels=['OptiTrack Start'], 
                loc='lower left', 
                )

plt.savefig('../figures/youbot/StPt_opti_all')
plt.close(fig_start)

print("\n--- Plotting Complete ---")