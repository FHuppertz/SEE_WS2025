import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats

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
        'source_key': 'opti', 
        'folder_core_path': '../data/optitrack/', 
        'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS},
        'label': 'OptiTrack'
    },
    {
        'source_key': 'rob', 
        'folder_core_path': '../data/youBot/', 
        'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS},
        'label': 'youBot'
    }
]

# Initialize storage
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
                            cols_of_interest = ['X.1', 'Y', 'Z.1']
                            df = pd.read_csv(file_path, skiprows=7, usecols=cols_of_interest)

                            if df.empty:
                                continue

                            rows = df.iloc[[0, -1]]
                            first_row = rows.iloc[0]
                            last_row = rows.iloc[1]

                            start_point = [first_row['X.1'], first_row['Y'], first_row['Z.1']]
                            end_point = [last_row['X.1'] + 212.4, last_row['Y'], last_row['Z.1'] - 76.17]

                        elif source == 'rob':
                            df = pd.read_csv(file_path, header=None)

                            if len(df.columns) < 3:
                                raise ValueError(f"youBot CSV expected 3 columns but found {len(df.columns)}.")

                            df_mean = df.mean()
                            mean_x = df_mean[0]
                            mean_y = df_mean[1]
                            mean_theta = df_mean[2]

                            start_point = [0, 0, 0]
                            end_point = [mean_x * 100, mean_theta, mean_y * 100]

                        else:
                            continue

                        points_start.append(start_point)
                        points_end.append(end_point)

                    except Exception as e:
                        print(f"Error in {filename}: {e}")

            data_store[canonical_obj][dir_name][start_key] = np.array(points_start)
            data_store[canonical_obj][dir_name][end_key] = np.array(points_end)

print("--- Data Loading Complete ---")

# Combine data by Size
COMBINED_OBJECT_BASES = ['large', 'medium', 'small']
DIRECTIONS = ['left', 'straight', 'right']
SOURCES = ['opti', 'rob']
POINTS = ['start', 'end']

combined_data_store = {
    size_base: {
        d: {
            f'{s}_{p}': [] 
            for s in SOURCES
            for p in POINTS
        }
        for d in DIRECTIONS
    }
    for size_base in COMBINED_OBJECT_BASES
}

print("--- Starting Data Aggregation by Object Size ---")

for canonical_obj, dir_data in data_store.items():
    object_base_size = ''
    for size in COMBINED_OBJECT_BASES:
        if canonical_obj.startswith(size):
            object_base_size = size
            break

    if not object_base_size:
        continue

    for dir_name, point_data in dir_data.items():
        for key in point_data:
            data_array = point_data[key]
            if data_array.size > 0:
                combined_data_store[object_base_size][dir_name][key].append(data_array)

combined_data = {}
total_concatenations = 0

for size_base in COMBINED_OBJECT_BASES:
    combined_data[size_base] = {}
    for dir_name in DIRECTIONS:
        combined_data[size_base][dir_name] = {}
        for key in combined_data_store[size_base][dir_name]:
            list_of_arrays = combined_data_store[size_base][dir_name][key]
            if list_of_arrays:
                concatenated_array = np.concatenate(list_of_arrays, axis=0)
                combined_data[size_base][dir_name][key] = concatenated_array
                total_concatenations += 1
            else:
                combined_data[size_base][dir_name][key] = np.array([])

print(f"--- Aggregation Complete. {total_concatenations} arrays concatenated. ---")


# --- STATISTICAL ANALYSIS SECTION ---

# 1. Define Ground Truths
GT_OBJ = {
    'pick': [0.143, -0.351, -1.65],
    'straight': [0.150, -0.212, -1.65],
    'left': [0.356, -0.283, -2.08],
    'right': [-0.064, -0.352, -1.13]
}

# 2. Statistical Functions

def check_chebyshev(data_points, axis_name, k_values=[2, 3]):
    """
    Checks if the data satisfies Chebyshev's inequality.
    """
    mu = np.mean(data_points)
    sigma = np.std(data_points)
    n = len(data_points)
    
    print(f"    [Chebyshev - {axis_name}] Mean: {mu:.2f}, Std: {sigma:.2f}")
    
    for k in k_values:
        lower_bound = mu - k * sigma
        upper_bound = mu + k * sigma
        count_within = np.sum((data_points >= lower_bound) & (data_points <= upper_bound))
        percentage_actual = (count_within / n) * 100
        chebyshev_bound = (1 - (1 / (k**2))) * 100
        
        status = "Pass" if percentage_actual >= chebyshev_bound else "Fail"
        print(f"      k={k}: {percentage_actual:.2f}% (Bound: >{chebyshev_bound:.2f}%) -> {status}")

def check_chi_squared_normality(data_points, axis_name, size, direction, output_dir="../figures/chi_squared/", alpha=0.05):
    """
    Performs Chi-Squared test and SAVES PLOT to output_dir.
    """
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n = len(data_points)
    if n < 8:
        print(f"    [Chi^2 - {axis_name}] Not enough data (n={n})")
        return

    # 1. Histogram / Bins
    k_bins = int(1 + 3.322 * np.log10(n))
    k_bins = max(k_bins, 5)
    
    observed_freq, bin_edges = np.histogram(data_points, bins=k_bins)
    
    # 2. Expected Frequencies
    mu = np.mean(data_points)
    sigma = np.std(data_points)
    
    expected_freq = []
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        prob = stats.norm.cdf(upper, mu, sigma) - stats.norm.cdf(lower, mu, sigma)
        expected_freq.append(prob * n)
    
    expected_freq = np.array(expected_freq)
    # Normalize expected to match observed sum
    if np.sum(expected_freq) > 0:
        expected_freq = expected_freq * (np.sum(observed_freq) / np.sum(expected_freq))

    # 3. Chi-Square Test
    chi2_stat, p_value = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
    significance = "Gaussian (H0)" if p_value > alpha else "Not Gaussian (Reject H0)"
    
    print(f"    [Chi^2 - {axis_name}] p={p_value:.4f} -> {significance}")

    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    
    # Plot Histogram (Density=True for comparison with PDF)
    plt.hist(data_points, bins=k_bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data Hist')
    
    # Plot Normal Distribution Curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit\n$\mu={mu:.2f}, \sigma={sigma:.2f}$')
    
    # Decoration
    plt.title(f'Chi^2 Test: {size.upper()} - {direction.upper()} ({axis_name})\n$p={p_value:.4f}$ ({significance})')
    plt.xlabel(f'{axis_name} Value (cm)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    filename = f"{size}_{direction}_{axis_name}_chi2.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close() # Important to close to free memory
    # print(f"      Saved plot to {save_path}")


# --- 3. Main Analysis Loop ---
print("\n=======================================================")
print("===       STATISTICAL ANALYSIS (YouBot vs GT)       ===")
print("=======================================================\n")

for size in COMBINED_OBJECT_BASES:
    for direction in DIRECTIONS:
        dataset_key = 'opti_end' 
        data = combined_data[size][direction][dataset_key]
        
        if data.size == 0:
            print(f"Skipping {size} - {direction}: No data found.")
            continue

        print(f"\n--- Analyzing: Object {size.upper()} | Pose {direction.upper()} ---")
        
        observed_x = data[:, 0] # X (cm)
        observed_y = data[:, 2] # Y (cm)
        
        # Ground Truth (Meters -> cm)
        gt_x = GT_OBJ[direction][0] * 100 
        gt_y = GT_OBJ[direction][1] * 100 
        
        # --- A. ACCURACY & PRECISION ---
        errors = np.sqrt((observed_x - gt_x)**2 + (observed_y - gt_y)**2)
        
        accuracy = np.mean(errors)
        precision_x = np.std(observed_x)
        precision_y = np.std(observed_y)
        
        print(f"  [Metrics] Acc: {accuracy:.3f}cm | Prec X: {precision_x:.3f}cm | Prec Y: {precision_y:.3f}cm")

        # --- B. CHEBYSHEV ---
        check_chebyshev(observed_x, "X-Axis")
        check_chebyshev(observed_y, "Y-Axis")
        
        # --- C. CHI-SQUARED + PLOTTING ---
        # Passing size and direction for filenames
        check_chi_squared_normality(observed_x, "X-Axis", size, direction)
        check_chi_squared_normality(observed_y, "Y-Axis", size, direction)