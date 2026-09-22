import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, hilbert

# File and directory paths
data_dir = '/home/nyx/cwru_dataset'
file_names = {
    'Normal': '97.mat',
    'Inner Race Fault (0.007")': '105.mat',
    'Ball Fault (0.007")': '118.mat',
    'Outer Race Fault (0.007")': '130.mat'
}

def get_fault_frequencies(rpm):
    fr = rpm / 60.0
    bpfo = (9 / 2.0) * fr * (1 - (0.3126 / 1.537))
    bpfi = (9 / 2.0) * fr * (1 + (0.3126 / 1.537))
    bsf  = (1.537 / (2.0 * 0.3126)) * fr * (1 - (0.3126 / 1.537)**2)
    return {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf}

def get_de_signal(data_dict):
    for key in data_dict.keys():
        if key.endswith('_DE_time'):
            return data_dict[key].ravel()
    raise KeyError("No Drive End signal found.")

def get_rpm(data_dict):
    for key in data_dict.keys():
        if key.endswith('RPM'):
            return float(data_dict[key][0][0])
    return 1797.0

def bandpass_filter(signal, lowcut=2000, highcut=4000, fs=12000, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, signal)

def get_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope - np.mean(envelope)

def compute_fft(signal, fs=12000):
    N = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(N, 1.0 / fs)[:N // 2]
    amplitudes = (2.0 / N) * np.abs(fft_vals[:N // 2])
    return freqs, amplitudes

# Main execution loop
fs = 12000
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for i, (label, filename) in enumerate(file_names.items()):
    filepath = os.path.join(data_dir, filename)
    data_dict = loadmat(filepath)

    signal = get_de_signal(data_dict)[:12000] # 1 sec signal segment
    rpm = get_rpm(data_dict)
    freq_dict = get_fault_frequencies(rpm)

    # 1. Bandpass filter around resonance (2000-4000 Hz)
    filtered = bandpass_filter(signal, 2000, 4000, fs)

    # 2. Extract envelope using Hilbert transform
    envelope = get_envelope(filtered)

    # 3. FFT of envelope
    freqs, amplitudes = compute_fft(envelope, fs)

    # Plot up to 400 Hz
    axes[i].plot(freqs, amplitudes, color='darkgreen', linewidth=1.0)
    axes[i].set_title(f"{label} - Envelope Spectrum (Resonance Demodulated)", fontsize=10, fontweight='bold')
    axes[i].set_ylabel("Amplitude")
    axes[i].set_xlim(0, 400)
    axes[i].grid(True, linestyle='--', alpha=0.5)

    # Mark theoretical defect frequencies & harmonics
    if 'Outer Race' in label:
        bpfo = freq_dict['BPFO']
        axes[i].axvline(bpfo, color='red', linestyle='--', label=f"BPFO ({bpfo:.1f} Hz)")
        axes[i].axvline(2*bpfo, color='red', linestyle=':', alpha=0.7, label=f"2x BPFO ({2*bpfo:.1f} Hz)")
        axes[i].legend(loc='upper right')
    elif 'Inner Race' in label:
        bpfi = freq_dict['BPFI']
        axes[i].axvline(bpfi, color='magenta', linestyle='--', label=f"BPFI ({bpfi:.1f} Hz)")
        axes[i].axvline(2*bpfi, color='magenta', linestyle=':', alpha=0.7, label=f"2x BPFI ({2*bpfi:.1f} Hz)")
        axes[i].legend(loc='upper right')
    elif 'Ball' in label:
        bsf = freq_dict['BSF']
        axes[i].axvline(bsf, color='orange', linestyle='--', label=f"BSF ({bsf:.1f} Hz)")
        axes[i].legend(loc='upper right')

axes[-1].set_xlabel("Frequency (Hz)", fontsize=11)
plt.tight_layout()

output_plot_path = os.path.join(data_dir, 'envelope_spectrum_comparison.png')
plt.savefig(output_plot_path, dpi=300)
print(f"Envelope spectrum plot successfully generated and saved to: {output_plot_path}")
