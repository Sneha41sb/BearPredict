import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft, fftfreq


# Define path and file names
data_dir = '/home/nyx/cwru_dataset'

file_names = {
    'Normal': '97.mat',
    'Inner Race Fault (0.007")': '105.mat',
    'Ball Fault (0.007")': '118.mat',
    'Outer Race Fault (0.007")': '130.mat'
}


# Function to compute theoretical fault frequencies
def get_fault_frequencies(rpm):
    fr = rpm / 60  # rotational speed

    bpfo = (9 / 2) * fr * (1 - (0.3126 / 1.537))
    bpfi = (9 / 2.0) * fr * (1 + (0.3126 / 1.537))
    bsf = (1.537 / (2.0 * 0.3126)) * fr * (
        1 - (0.3126 / 1.537) ** 2
    )

    return {
        'BPFO': bpfo,
        'BPFI': bpfi,
        'BSF': bsf
    }


# Helper function to get DE signal
def get_de_signal(data_dict):
    for key in data_dict.keys():
        if key.endswith('_DE_time'):
            return data_dict[key].ravel()

    return None


# Function to get RPM
def get_rpm(data_dict):
    for key in data_dict.keys():
        if key.endswith('_RPM'):
            return float(data_dict[key][0][0])

    return 1797.0


# Function to compute FFT
def compute_fft(signal, fs=12000):
    N = len(signal)

    fft_vals = fft(signal)

    freqs = fftfreq(N, 1.0 / fs)[:N // 2]

    amplitudes = (2.0 / N) * np.abs(
        fft_vals[:N // 2]
    )

    return freqs, amplitudes


# Main loop for processing files and plotting
fs = 12000

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 10),
    sharex=True
)


for i, (label, filename) in enumerate(file_names.items()):

    filepath = os.path.join(data_dir, filename)

    data_dict = loadmat(filepath)

    signal = get_de_signal(data_dict)

    rpm = get_rpm(data_dict)

    freq_dict = get_fault_frequencies(rpm)

    # Take 1 second of signal (12,000 samples)
    segment = signal[:12000]

    freqs, amplitudes = compute_fft(segment, fs)

    # Plot frequency range 0 to 500 Hz
    axes[i].plot(
        freqs,
        amplitudes,
        color='navy',
        linewidth=1.0
    )

    axes[i].set_title(
        f"{label} - FFT Spectrum",
        fontsize=10,
        fontweight='bold'
    )

    axes[i].set_ylabel("Amplitude")

    axes[i].set_xlim(0, 500)

    axes[i].grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    # Mark theoretical frequencies on the plots
    if 'Outer Race' in label:

        axes[i].axvline(
            freq_dict['BPFO'],
            color='red',
            linestyle='--',
            label=f"BPFO ({freq_dict['BPFO']:.1f} Hz)"
        )

        axes[i].legend(loc='upper right')

    elif 'Inner Race' in label:

        axes[i].axvline(
            freq_dict['BPFI'],
            color='magenta',
            linestyle='--',
            label=f"BPFI ({freq_dict['BPFI']:.1f} Hz)"
        )

        axes[i].legend(loc='upper right')

    elif 'Ball' in label:

        axes[i].axvline(
            freq_dict['BSF'],
            color='orange',
            linestyle='--',
            label=f"BSF ({freq_dict['BSF']:.1f} Hz)"
        )

        axes[i].legend(loc='upper right')


# Final plot settings
axes[-1].set_xlabel(
    "Frequency (Hz)",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        data_dir,
        'fft_spectrum_comparison.png'
    ),
    dpi=300
)

print("FFT spectrum plot generated and saved successfully!")