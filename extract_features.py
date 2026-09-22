import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq

# Settings
data_dir = '/home/nyx/cwru_dataset'
file_names = {
    'Normal': '97.mat',
    'Inner Race Fault': '105.mat',
    'Ball Fault': '118.mat',
    'Outer Race Fault': '130.mat'
}

fs = 12000
window_size = 2048
overlap = 1024
step = window_size - overlap

def get_de_signal(data_dict):
    for key in data_dict.keys():
        if key.endswith('_DE_time'):
            return data_dict[key].ravel()
    raise KeyError("No Drive End signal found.")

def extract_window_features(window, fs=12000):
    # 1. Time-Domain Features
    mean_val = np.mean(window)
    std_val  = np.std(window)
    rms_val  = np.sqrt(np.mean(window**2))
    peak_val = np.max(np.abs(window))
    p2p_val  = np.max(window) - np.min(window)
    
    kurt_val = kurtosis(window)
    skew_val = skew(window)
    
    abs_mean = np.mean(np.abs(window))
    square_root_mean = (np.mean(np.sqrt(np.abs(window))))**2
    
    crest_factor   = peak_val / (rms_val + 1e-8)
    shape_factor   = rms_val / (abs_mean + 1e-8)
    impulse_factor = peak_val / (abs_mean + 1e-8)
    margin_factor  = peak_val / (square_root_mean + 1e-8)
    
    # 2. Frequency-Domain Features (FFT)
    N = len(window)
    fft_vals = np.abs(fft(window))[:N // 2]
    freqs = fftfreq(N, 1.0 / fs)[:N // 2]
    
    total_energy = np.sum(fft_vals**2)
    mean_freq    = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
    
    # Energy in BPFO zone (100 - 115 Hz) vs BPFI zone (155 - 170 Hz)
    bpfo_mask = (freqs >= 100) & (freqs <= 115)
    bpfi_mask = (freqs >= 155) & (freqs <= 170)
    
    bpfo_energy = np.sum(fft_vals[bpfo_mask]**2) / (total_energy + 1e-8)
    bpfi_energy = np.sum(fft_vals[bpfi_mask]**2) / (total_energy + 1e-8)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'rms': rms_val,
        'peak': peak_val,
        'p2p': p2p_val,
        'kurtosis': kurt_val,
        'skewness': skew_val,
        'crest_factor': crest_factor,
        'shape_factor': shape_factor,
        'impulse_factor': impulse_factor,
        'margin_factor': margin_factor,
        'spectral_energy': total_energy,
        'mean_frequency': mean_freq,
        'bpfo_energy_ratio': bpfo_energy,
        'bpfi_energy_ratio': bpfi_energy
    }

# Process signals and construct DataFrame
dataset_rows = []

print("--- Extracting Features from Segmented Signals ---")
for label, filename in file_names.items():
    filepath = os.path.join(data_dir, filename)
    data_dict = loadmat(filepath)
    signal = get_de_signal(data_dict)
    
    num_windows = (len(signal) - window_size) // step + 1
    print(f"[{label}] File: {filename} -> Total Samples: {len(signal)} | Windows Extracted: {num_windows}")
    
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        window = signal[start_idx:end_idx]
        
        feats = extract_window_features(window, fs)
        feats['label'] = label
        dataset_rows.append(feats)

# Convert to Pandas DataFrame
df = pd.DataFrame(dataset_rows)

# Save to CSV
csv_output_path = os.path.join(data_dir, 'cwru_features.csv')
df.to_csv(csv_output_path, index=False)
print(f"\nFeature extraction complete! Saved dataset with shape {df.shape} to: {csv_output_path}")
