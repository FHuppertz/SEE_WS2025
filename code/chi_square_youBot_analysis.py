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
    Assumes X and Y data are already in cm.
    Returns the quiver object for the legend.
    """
    if data_array.size > 0:
        X = data_array[:, 0]  # X position (in cm)
        Y = data_array[:, 2]  # Y position (in cm)
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

# --- NEW Function to plot a single Ground Truth pose ---
def plot_ground_truth(ax, gt_pose, label, color, scale_factor_cm, marker='s', size=5):
    """
    Plots a single ground truth pose as a scatter point with a quiver arrow.

    gt_pose format: [x (m), y (m), theta (rad)]
    scale_factor_cm: The desired scale factor for the quiver arrow in cm units.

    The coordinates are converted from meters (m) to centimeters (cm).
    """
    # **Meters to Centimeters Conversion (This is the crucial step)**
    # This *correctly* converts the ground truth table (in meters) to cm
    x_plot = gt_pose[0] * 100
    y_plot = gt_pose[1] * 100
    theta = gt_pose[2]

    # Calculate arrow components
    u = np.cos(theta)
    v = np.sin(theta)


    # 2. Plot the quiver arrow (orientation)
    quiv = ax.quiver(x_plot, y_plot, u, v,
                     color=color,
                     scale=scale_factor_cm, # Use the scale factor appropriate for cm
                     scale_units='xy',
                     angles='xy',
                     width=0.005,
                     headwidth=3,
                     headlength=4,
                     zorder=10,
                     label=f'{label}')

    # Return the point (scatter) handle for the legend, as it's cleaner
    return quiv

# --- Configuration ---

# **NEW: Ground Truth Data from the image tables (in meters)**

# Table I: Ground truth object pose (Used for Start Point - 'Pick' pose)
GROUND_TRUTH_OBJECT_POSE = {
    'Pick': [0.143, -0.351, -1.65],
    'Straight': [0.150, -0.212, -1.65],
    'Left': [0.356, -0.283, -2.08],
    'Right': [-0.064, -0.352, -1.13]
}

# Table II: Ground truth end-effector pose (Used for End Points)
GROUND_TRUTH_END_EFFECTOR_POSE = {
    'Pick': [0.152, -0.457, -1.65],
    'Straight': [0.167, -0.317, -1.62],
    'Left': [0.331, -0.380, -2.11],
    'Right': [-0.01, -0.433, -1.11]
}

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


                            start_point = [first_row[col_x_plot], first_row[col_theta_color], first_row[col_y_plot]]
                            end_point = [last_row[col_x_plot]+212.4, last_row[col_theta_color], last_row[col_y_plot]-76.17]

                        elif source == 'rob':
                            # --- YOUBOT DATA LOADING ---
                            df = pd.read_csv(file_path, header=None)

                            if len(df.columns) < 3:
                                raise ValueError(f"youBot CSV expected 3 columns (X, Y, Theta) but found {len(df.columns)}.")

                            df_mean = df.mean()

                            mean_x = df_mean[0]
                            mean_y = df_mean[1]
                            mean_theta = df_mean[2]

                            # **FIXED: Removed *100. Assumes data in CSV is already in cm.**
                            # Mapped structure: [X_plot (0), Theta_color (2), Y_plot (1)]
                            start_point = [0, 0, 0]
                            end_point = [mean_x*100, mean_theta, mean_y*100]

                        else:
                            continue

                        # Append points
                        points_start.append(start_point)
                        points_end.append(end_point)

                        # Collect (cm) X and Y values for global limits
                        all_xy_values.append((start_point[0], start_point[2]))
                        all_xy_values.append((end_point[0], end_point[2]))

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

# --- 2. Calculate Global X and Y Limits, including Ground Truth ---
# Include Ground Truth values (converted to cm) in the limit calculation
for pose in GROUND_TRUTH_OBJECT_POSE.values():
    # Convert m to cm for limits calculation
    all_xy_values.append((pose[0]*100, pose[1]*100))
for pose in GROUND_TRUTH_END_EFFECTOR_POSE.values():
    # Convert m to cm for limits calculation
    all_xy_values.append((pose[0]*100, pose[1]*100))

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

# Define the common scale factor for quivers in the cm plots
# This value (0.1) worked for the previous data, assuming it was 100x too large
# Now that the data is 100x smaller, the scale factor needs to be 100x larger
# to make the arrows the same *visual* size.
# Let's try 100 * 0.1 = 10
CM_QUIVER_SCALE_FACTOR = 0.1

# 3. Generate the 2D plots using the unified data_store (MODIFIED)
print("\n--- Generating Grouped End Point 2D Quiver Plots ---")

# Define base colors/alpha for OptiTrack and youBot source differentiation
COLOR_OPTI_BASE = 0.8 # Transparency/alpha for OptiTrack
COLOR_ROB_BASE = 1.0 # Transparency/alpha for youBot
GT_COLOR = '#006600' # Dark Green for Ground Truth

# Iterate over the canonical object BASES (large, medium, small)
for object_base in CANONICAL_OBJECT_BASES:

    # Iterate over the DIRECTIONS (left, straight, right)
    for dir_name in DIRECTIONS:

        fig_end, ax_end = plt.subplots(figsize=(10, 5.5))

        # Store handles and labels for this specific plot
        current_handles = {}

        # Plot the **Ground Truth End-Effector Pose** for the current direction
        gt_pose_name = dir_name.capitalize()
        if gt_pose_name in GROUND_TRUTH_OBJECT_POSE:
            gt_pose = GROUND_TRUTH_OBJECT_POSE[gt_pose_name]
            h_gt = plot_ground_truth(ax_end,
                                     gt_pose,
                                     f'Ground Truth Object',
                                     GT_COLOR,
                                     # Use a prominent scale for the GT arrow
                                     scale_factor_cm=CM_QUIVER_SCALE_FACTOR,
                                     size=5)
            current_handles[h_gt.get_label()] = h_gt

        # Iterate over the numbered object indices (1, 2, 3, 4)
        #
        all_end_opti = []
        all_end_rob = []
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
            q3 = plot_quiver_data(ax_end, end_opti, opti_color, f'OptiTrack {canonical_obj}', scale_factor=CM_QUIVER_SCALE_FACTOR)

            # 2. Plot youBot (rob) End Quivers (full color)
            rob_color = list(dynamic_color[:3]) + [COLOR_ROB_BASE]
            q4 = plot_quiver_data(ax_end, end_rob, rob_color, f'youBot {canonical_obj}', scale_factor=CM_QUIVER_SCALE_FACTOR)

            # --- Legend Management ---
            # Add handles for this specific plot.
            key_opti = f'OptiTrack - Group{canonical_obj[-1]}'
            if key_opti not in current_handles:
                 current_handles[key_opti] = q3

            key_rob = f'youBot - Group{canonical_obj[-1]}'
            if key_rob not in current_handles:
                 current_handles[key_rob] = q4


        # TODO CHI SQUARE:


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
        # Need to sort based on a common string key to group the GT handle
        sorted_items = sorted(current_handles.items(), key=lambda item: item[0])
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
     for dir_name in DIRECTIONS if data_store[obj][dir_name]['opti_start'].size > 0],
    axis=0
)

fig_start, ax_start = plt.subplots(figsize=(10, 8))

# Store handles and labels for this specific plot
current_handles_start = {}

# Plot the **Ground Truth Object Pose** for 'Pick' (Start)
if 'Pick' in GROUND_TRUTH_OBJECT_POSE:
    gt_pose_pick = GROUND_TRUTH_OBJECT_POSE['Pick']
    h_gt_pick = plot_ground_truth(ax_start,
                                  gt_pose_pick,
                                  'Ground Truth Start',
                                  GT_COLOR,
                                  scale_factor_cm=CM_QUIVER_SCALE_FACTOR,
                                  size=2,
                                  marker='*')
    current_handles_start[h_gt_pick.get_label()] = h_gt_pick

# Plot OptiTrack Start Quivers
# Use the new, larger scale factor
q = plot_quiver_data(ax_start, all_start, 'green', 'All OptiTrack Starts', scale_factor=CM_QUIVER_SCALE_FACTOR)
current_handles_start[q.get_label()] = q

ax_start.set_xlabel('X (cm)')
ax_start.set_ylabel('Y (cm)')
ax_start.set_title(f'Start Points of all Trials (OptiTrack)')
ax_start.set_aspect('equal')
ax_start.grid(True, linestyle='--', alpha=0.6)

# Apply uniform limits
ax_start.set_xlim(X_LIM)
ax_start.set_ylim(Y_LIM)

# Create the legend
sorted_items_start = sorted(current_handles_start.items(), key=lambda item: item[0])
handles_start = [item[1] for item in sorted_items_start]
labels_start = [item[0] for item in sorted_items_start]

ax_start.legend(handles=handles_start,
                labels=labels_start,
                loc='lower left',
                )

plt.savefig('../figures/youbot/StPt_opti_all')
plt.close(fig_start)

print("\n--- Plotting Complete ---")
