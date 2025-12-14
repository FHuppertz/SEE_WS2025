import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats

# --- Configuration Variables ---
CANONICAL_OBJECT_BASES = ['large', 'medium', 'small']
OBJECT_INDICES = list(range(1, 5))

CANONICAL_OBJECTS = [f'{base}{idx}' for base in CANONICAL_OBJECT_BASES for idx in OBJECT_INDICES]
DIRECTIONS = ['left', 'straight', 'right']

# Define Data Sources
DATA_CONFIG = [
    {'source_key': 'opti', 'folder_core_path': '../data/optitrack/', 'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS}, 'label': 'OptiTrack'},
    {'source_key': 'rob', 'folder_core_path': '../data/youBot/', 'object_names_map': {obj: obj for obj in CANONICAL_OBJECTS}, 'label': 'youBot'}
]

# Initialize storage
data_store = {
    obj: {d: {'opti_start': np.array([]), 'opti_end': np.array([]), 'rob_start': np.array([]), 'rob_end': np.array([])} for d in DIRECTIONS}
    for obj in CANONICAL_OBJECTS
}

# --- 1. CSV Reading Logic ---
print("--- Starting CSV Reading ---")
for config in DATA_CONFIG:
    source = config['source_key']
    base_path = config['folder_core_path']
    name_map = config['object_names_map']

    for canonical_obj in CANONICAL_OBJECTS:
        object_folder_name = name_map[canonical_obj]
        for dir_name in DIRECTIONS:
            folder_path = os.path.join(base_path, object_folder_name, dir_name)
            if not os.path.isdir(folder_path): continue

            start_key, end_key = f'{source}_start', f'{source}_end'
            points_start, points_end = [], []

            for filename in os.listdir(folder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if source == 'opti':
                            df = pd.read_csv(file_path, skiprows=7, usecols=['X.1', 'Y', 'Z.1'])
                            if df.empty: continue
                            rows = df.iloc[[0, -1]]
                            # Opti: [X, Y, Z] -> X and Z are horizontal plane
                            points_start.append([rows.iloc[0]['X.1'], rows.iloc[0]['Y'], rows.iloc[0]['Z.1']])
                            points_end.append([rows.iloc[1]['X.1'] + 212.4, rows.iloc[1]['Y'], rows.iloc[1]['Z.1'] - 76.17])
                        elif source == 'rob':
                            df = pd.read_csv(file_path, header=None)
                            if len(df.columns) < 3: continue
                            mean_val = df.mean()
                            points_start.append([0, 0, 0])
                            # Rob: [X, Theta, Y] -> Reordered to [X, Theta, Y]
                            points_end.append([mean_val[0] * 100, mean_val[2], mean_val[1] * 100])
                    except Exception: continue

            data_store[canonical_obj][dir_name][start_key] = np.array(points_start)
            data_store[canonical_obj][dir_name][end_key] = np.array(points_end)

print("--- Data Loading Complete ---")

# --- 2. Aggregation ---
COMBINED_OBJECT_BASES = ['large', 'medium', 'small']
combined_data = {size: {d: {} for d in DIRECTIONS} for size in COMBINED_OBJECT_BASES}

for size in COMBINED_OBJECT_BASES:
    for direction in DIRECTIONS:
        for source in ['opti', 'rob']:
            key = f'{source}_end'
            arrays_list = []
            
            # Collect all arrays for this Size + Direction + Source
            for obj in CANONICAL_OBJECTS:
                if obj.startswith(size):
                    arr = data_store[obj][direction][key]
                    if arr.size > 0: arrays_list.append(arr)
            
            if arrays_list:
                combined_data[size][direction][key] = np.concatenate(arrays_list, axis=0)
            else:
                combined_data[size][direction][key] = np.array([])

print("--- Aggregation Complete ---")


# --- 3. Statistical Analysis Setup ---

GT_OBJ = {
    'pick': [0.143, -0.351, -1.65],
    'straight': [0.150, -0.212, -1.65],
    'left': [0.356, -0.283, -2.08],
    'right': [-0.064, -0.352, -1.13]
}

def remove_outliers_chebyshev(data, k=3):
    """
    Filters 2D/3D points. A point is removed if X OR Y is > k std devs away.
    Expects data in format [X, Theta, Y] (based on previous logic).
    Returns cleaned data and count of removed points.
    """
    if data.size == 0: return data, 0
    
    # Extract X (col 0) and Y (col 2)
    x = data[:, 0]
    y = data[:, 2]
    
    mean_x, std_x = np.mean(x), np.std(x)
    mean_y, std_y = np.mean(y), np.std(y)
    
    # Create mask: True if point is VALID (within k sigma)
    mask_x = np.abs(x - mean_x) <= (k * std_x)
    mask_y = np.abs(y - mean_y) <= (k * std_y)
    
    # Combine masks (Must be valid in BOTH X and Y to stay)
    final_mask = mask_x & mask_y
    
    cleaned_data = data[final_mask]
    removed_count = len(data) - len(cleaned_data)
    
    return cleaned_data, removed_count

def check_chebyshev(data_points, axis_name, k_values=[2, 3]):
    mu = np.mean(data_points)
    sigma = np.std(data_points)
    n = len(data_points)
    
    # print(f"    [Chebyshev - {axis_name}] Mean: {mu:.2f}, Std: {sigma:.2f}")
    
    for k in k_values:
        lower, upper = mu - k*sigma, mu + k*sigma
        count = np.sum((data_points >= lower) & (data_points <= upper))
        pct = (count / n) * 100
        bound = (1 - 1/k**2) * 100
        status = "Pass" if pct >= bound else "Fail"
        print(f"      k={k}: {pct:.1f}% inside (Bound >{bound:.1f}%) -> {status}")

def check_chi_squared_and_plot(data_points, axis_name, size, direction, source_label, output_dir, alpha=0.05):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    n = len(data_points)
    if n < 8: return 

    # Chi-Squared Test
    k_bins = max(int(1 + 3.322 * np.log10(n)), 5)
    obs_freq, bin_edges = np.histogram(data_points, bins=k_bins)
    
    mu, sigma = np.mean(data_points), np.std(data_points)
    
    exp_freq = []
    for i in range(len(bin_edges)-1):
        cdf_upper = stats.norm.cdf(bin_edges[i+1], mu, sigma)
        cdf_lower = stats.norm.cdf(bin_edges[i], mu, sigma)
        exp_freq.append((cdf_upper - cdf_lower) * n)
    
    exp_freq = np.array(exp_freq) * (np.sum(obs_freq) / np.sum(exp_freq))
    
    chi2_stat, p_val = stats.chisquare(obs_freq, exp_freq)
    sig = "Gaussian" if p_val > alpha else "Not Gaussian"
    
    print(f"    [Chi^2 {axis_name}] p={p_val:.4f} -> {sig}")
    
    # Plotting
    plt.figure(figsize=(8,6))
    plt.hist(data_points, bins=k_bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    
    x_plot = np.linspace(min(data_points), max(data_points), 100)
    plt.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), 'r-', lw=2, label=f'Norm Fit\n$\mu={mu:.1f}, \sigma={sigma:.1f}$')
    
    plt.title(f'{source_label} | {size} - {direction} ({axis_name})\n{sig} (p={p_val:.3f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    filename = f"{source_label}_{size}_{direction}_{axis_name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# --- 4. Main Analysis Loop ---
print("\n=======================================================")
print("===   STATISTICAL ANALYSIS (Outlier Removal ON)     ===")
print("=======================================================\n")

# Configuration for the two passes
ANALYSIS_CONFIG = [
    {'name': 'OptiTrack', 'key': 'opti_end', 'save_folder': '../figures/chi_squared/optitrack/'},
    {'name': 'YouBot',    'key': 'rob_end',  'save_folder': '../figures/chi_squared/youbot/'}
]

for config in ANALYSIS_CONFIG:
    source_name = config['name']
    data_key = config['key']
    save_path = config['save_folder']
    
    print(f"\n>>>>>>>> STARTING ANALYSIS FOR: {source_name.upper()} <<<<<<<<")

    for size in COMBINED_OBJECT_BASES:
        for direction in DIRECTIONS:
            
            raw_data = combined_data[size][direction][data_key]
            
            if raw_data.size == 0:
                continue

            print(f"\n--- {size.upper()} | {direction.upper()} ---")

            # 1. Remove Outliers
            clean_data, num_removed = remove_outliers_chebyshev(raw_data, k=3) # Removing 3-sigma outliers
            
            print(f"  Data Points: {len(raw_data)} -> {len(clean_data)} (Removed {num_removed} outliers)")
            
            if len(clean_data) < 2:
                print("  Not enough data after cleaning.")
                continue

            # 2. Extract X and Y (Using indices 0 and 2 based on previous mapping)
            vals_x = clean_data[:, 0]
            vals_y = clean_data[:, 2]
            
            # Ground Truth
            gt_x = GT_OBJ[direction][0] * 100
            gt_y = GT_OBJ[direction][1] * 100
            
            # 3. Accuracy & Precision (On Clean Data)
            # Note: For YouBot (odometry), GT comparison might be relative to 0 or target depending on interpretation.
            # Here we assume the GT dictionary applies to physical space (OptiTrack). 
            # If YouBot data is odometry error, GT might effectively be the reported coordinates vs target.
            
            errors = np.sqrt((vals_x - gt_x)**2 + (vals_y - gt_y)**2)
            acc = np.mean(errors)
            prec_x = np.std(vals_x)
            prec_y = np.std(vals_y)
            
            print(f"  [Metrics] Acc: {acc:.2f}cm | Prec X: {prec_x:.2f} | Prec Y: {prec_y:.2f}")

            # 4. Chebyshev Check (On Clean Data - verifying distribution properties)
            check_chebyshev(vals_x, "X-Axis")
            check_chebyshev(vals_y, "Y-Axis")

            # 5. Chi-Squared & Plotting (Separate Folders)
            check_chi_squared_and_plot(vals_x, "X", size, direction, source_name, save_path)
            check_chi_squared_and_plot(vals_y, "Y", size, direction, source_name, save_path)

print("\n--- Analysis Complete. Plots saved to ../figures/chi_squared/ ---")