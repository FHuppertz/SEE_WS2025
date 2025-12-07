import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Configuration Variables ---
CANONICAL_OBJECT_BASES = ['large', 'medium', 'small']
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
        'folder_core_path': '../data/optitrack/', # Adjust this relative path as needed
        # Map the canonical key ('large1') to the folder name ('large1')
        'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS},
        'label': 'OptiTrack'
    },
    {
        'source_key': 'rob', # Used as the dictionary key in the data_store
        'folder_core_path': '../data/youBot/', # Adjust this relative path as needed
        # Map the canonical key ('large1') to the folder name ('large1')
        'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS},
        'label': 'youBot'
    }
]

# Initialize storage using the new canonical keys
# Structure: data_store[object_key][direction][source_point_key] = np.array([[X, Theta, Y], ...])
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

# --- 1. Optimized CSV Reading Logic ---
print("--- Starting Optimized CSV Reading ---")

for config in DATA_CONFIG:
    source = config['source_key']
    base_path = config['folder_core_path']
    name_map = config['object_names_map']

    for canonical_obj in CANONICAL_OBJECTS:
        object_folder_name = name_map[canonical_obj]

        for dir_name in DIRECTIONS:
            # Construct the current sub-directory path: ../data/optitrack/large1/left
            folder_path = os.path.join(base_path, object_folder_name, dir_name)

            if not os.path.isdir(folder_path):
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
                            # Only read required columns and skip header rows
                            cols_of_interest = ['X.1', 'Y', 'Z.1']
                            df = pd.read_csv(file_path, skiprows=7, usecols=cols_of_interest)

                            if df.empty:
                                continue

                            # Get the start (index 0) and end (index -1) rows ONLY (speedup)
                            rows = df.iloc[[0, -1]]

                            first_row = rows.iloc[0]
                            last_row = rows.iloc[1]

                            # OptiTrack: [X_cm, Theta_rad, Y_cm]
                            start_point = [first_row['X.1'], first_row['Y'], first_row['Z.1']]
                            # Apply end point offset
                            end_point = [last_row['X.1'] + 212.4, last_row['Y'], last_row['Z.1'] - 76.17]

                        elif source == 'rob':
                            # --- YOUBOT DATA LOADING ---
                            # Read all rows for mean calculation
                            df = pd.read_csv(file_path, header=None)

                            if len(df.columns) < 3:
                                raise ValueError(f"youBot CSV expected 3 columns (X, Y, Theta) but found {len(df.columns)}.")

                            # Calculate the mean of all columns (X, Y, Theta)
                            df_mean = df.mean()

                            mean_x = df_mean[0]
                            mean_y = df_mean[1]
                            mean_theta = df_mean[2]

                            # Start point: [0, 0, 0]
                            start_point = [0, 0, 0]

                            # End point: [X_cm, Theta_rad, Y_cm] (Meters * 100 to cm)
                            end_point = [mean_x * 100, mean_theta, mean_y * 100]

                        else:
                            continue

                        # Append points
                        points_start.append(start_point)
                        points_end.append(end_point)


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

print("--- Data Loading Complete ---")
print("Format: X Theta Y")
#print(data_store['large1']['left']['opti_end'].shape)

# Define the three primary object size categories
COMBINED_OBJECT_BASES = ['large', 'medium', 'small']
DIRECTIONS = ['left', 'straight', 'right']
SOURCES = ['opti', 'rob']
POINTS = ['start', 'end']

# Initialize the new data store
# Structure: combined_data_store[size_base][direction][source_point_key] = np.array([[X, Theta, Y], ...])
combined_data_store = {
    size_base: {
        d: {
            f'{s}_{p}': []  # Initialize as a list to hold arrays for concatenation
            for s in SOURCES
            for p in POINTS
        }
        for d in DIRECTIONS
    }
    for size_base in COMBINED_OBJECT_BASES
}

print("--- Starting Data Aggregation by Object Size ---")

# 2. Iterate through the detailed data_store
for canonical_obj, dir_data in data_store.items():

    # Determine the object's base size (e.g., 'large1' -> 'large')
    object_base_size = ''
    for size in COMBINED_OBJECT_BASES:
        if canonical_obj.startswith(size):
            object_base_size = size
            break

    if not object_base_size:
        print(f"Warning: Could not categorize object {canonical_obj}. Skipping.")
        continue

    # Iterate through directions ('left', 'straight', 'right')
    for dir_name, point_data in dir_data.items():

        # Iterate through the source/point keys (e.g., 'opti_end')
        for key in point_data:
            data_array = point_data[key]

            # Check if the array contains data before trying to append
            if data_array.size > 0:
                # Append the NumPy array to the list associated with the combined category
                combined_data_store[object_base_size][dir_name][key].append(data_array)


# 3. Concatenate the lists of arrays into single NumPy arrays
combined_data = {}
total_concatenations = 0

for size_base in COMBINED_OBJECT_BASES:
    combined_data[size_base] = {}

    for dir_name in DIRECTIONS:
        combined_data[size_base][dir_name] = {}

        for key in combined_data_store[size_base][dir_name]:
            list_of_arrays = combined_data_store[size_base][dir_name][key]

            if list_of_arrays:
                # Concatenate all arrays along axis 0 (stacking rows)
                # This combines all 'large1' through 'large4' trials into one array.
                concatenated_array = np.concatenate(list_of_arrays, axis=0)
                combined_data[size_base][dir_name][key] = concatenated_array
                total_concatenations += 1
            else:
                # Keep an empty array if no data was found for this category
                combined_data[size_base][dir_name][key] = np.array([])

print(f"--- Aggregation Complete. {total_concatenations} arrays were concatenated. ---")
#(combined_data['large']['left']['opti_end'].shape)
