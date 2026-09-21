import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft, fftfreq

#define pth and files names
data_dir='/home/nyx/cwru_dataset'
file_names = {
    'Normal': '97.mat',
    'Inner Race Fault (0.007")': '105.mat',
    'Ball Fault (0.007")': '118.mat',
    'Outer Race Fault (0.007")': '130.mat'
}

#fnctn to compute theoretical fault frequencies
def get_fault_frequencies(rpm):
    fr=rpm /60  #rotational speed
    bpfo=(9/2)* fr *(1 - (0.3126 / 1.537))
    bpfi = (9 / 2.0) * fr * (1 + (0.3126 / 1.537))
    bsf  = (1.537 / (2.0 * 0.3126)) * fr * (1 - 
    (0.3126 / 1.537)**2)
    return {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf}
