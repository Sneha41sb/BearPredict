import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 1. Folder path where .mat files are downloaded
data_dir = '/home/nyx/cwru_dataset'

# 2. Mapping labels to CWRU file names
file_names = {
    'Normal': '97.mat',
    'Inner Race Fault (0.007")': '105.mat',
    'Ball Fault (0.007")': '118.mat',
    'Outer Race Fault (0.007")': '130.mat'
}

# 3. Helper function to extract Drive End signal from dictionary
def get_de_signal(data_dict):
    for key in data_dict.keys():
        if key.endswith('_DE_time'):
            # Flatten 2D matrix column into a 1D array of floats
            return data_dict[key].ravel()
    raise KeyError("No Drive End (_DE_time) signal key found in file dictionary.")

print("--- Step 1: Inspecting Dictionary Keys ---")
for label, filename in file_names.items():
    filepath = os.path.join(data_dir, filename)
    data_dict = loadmat(filepath)
    
    # Filter out MATLAB metadata headers (__header__, __version__, __globals__)
    clean_keys = [k for k in data_dict.keys() if not k.startswith('__')]
    print(f"[{label}] File: {filename} -> Keys: {clean_keys}")

print("\n--- Step 2: Generating & Saving Plots ---")
fs = 12000          # Sampling frequency for Drive End data = 12,000 Hz
num_samples = 1200  # 1200 samples = 0.1 seconds of data

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

for i, (label, filename) in enumerate(file_names.items()):
    filepath = os.path.join(data_dir, filename)
    data_dict = loadmat(filepath)
    
    # Extract raw vibration signal
    signal = get_de_signal(data_dict)
    
    # Take first 0.1s segment
    segment = signal[:num_samples]
    
    # Calculate time array t in seconds (t = index / sampling_rate)
    time = np.arange(num_samples) / fs
    
    # Plotting
    color = 'green' if 'Normal' in label else 'red'
    axes[i].plot(time, segment, color=color, linewidth=0.8)
    axes[i].set_title(f"{label} (File: {filename})", fontsize=10, fontweight='bold')
    axes[i].set_ylabel("Accel (g)")
    axes[i].grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel("Time (seconds)", fontsize=11)
plt.tight_layout()

# Save image file
output_plot_path = os.path.join(data_dir, 'raw_vibration_signals.png')
plt.savefig(output_plot_path, dpi=300)
print(f"Plot saved successfully to: {output_plot_path}")
