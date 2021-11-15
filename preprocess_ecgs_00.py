## Jessica Torres Soto, Aug. 15 2020
## Preprocessing ECGS, stanford - philips
'''
example usage: 

python preprocess_ecgs_00.py --input pathto/hha-ecgs-dir/Signals --output pathto/hha-ecgs-dir/SignalsProcessed

'''
import pandas as pd
import numpy as np 
from pickle import dump
import joblib
import h5py
import os
import tempfile
import argparse
import pdb
from pathlib import Path
import scipy
import glob
import scipy
import scipy.signal
import argparse


def get_command_line():
    '''
    # construct the argument parse and parse the arguments
    '''
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", type=str, default='hha-ecgs-dir/Signals', required=True, 
        help="input path")  

    ap.add_argument("-o", "--output", type=str, default="hha-ecgs-dir/SignalsProcessed", required=True,
        help="output path")

    args = vars(ap.parse_args())
    return args



def notch(data, fs):
    # Pre-processing
    # Moving averaging filter for power-line interference suppression:
    # averages samples in one period of the powerline
    # interference frequency with a first zero at this frequency.
    row,__ = data.shape
    #(data.shape)
    processed_data = np.zeros(data.shape)
    b = np.ones(int(0.02 * fs)) / 50.
    a = [1]
    for lead in range(0,row):
        X = scipy.signal.filtfilt(b, a, data[lead,:])
        processed_data[lead,:] = X
    return processed_data


def baseline_wander_removal(data, sampling_frequency):
    row,__ = data.shape
    processed_data = np.zeros(data.shape)

    win_size = int(np.round(0.2 * sampling_frequency)) + 1
    baseline = scipy.ndimage.median_filter(data, [1, win_size], mode='constant')
    win_size = int(np.round(0.6 * sampling_frequency)) + 1
    baseline = scipy.ndimage.median_filter(baseline, [1, win_size], mode='constant')
    filt_data = data - baseline
    
    return filt_data


def downsample_signal(df, Hz, NewHz):
    ten_secs = Hz * 10
    ecgs = df[:ten_secs]
    ecg_ds = scipy.signal.resample(x=ecgs.astype(np.float16), num=NewHz*10, axis=0)
    return ecg_ds


def loadecg(filename: str) -> np.ndarray:
    """Loads a ecg signal from a file.
    Args:
        filename (str): filename of ecg signal
    Returns:
        A np.ndarray with dimensions  [sequence_length, channels]. The
        values will be float16's.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    ecg_signal = np.load(filename)
    if ecg_signal.shape[1] == 12:
        ecg_signal = ecg_signal.T
    channels, sequence_length = ecg_signal.shape
    assert (channels == 12), "Channels are not set to 12"
    if sequence_length > 5000:
        ecg_signal = ecg_signal[:,:5000]
        channels, sequence_length = ecg_signal.shape
    if not channels == 12:
        raise NameError('Channel length is not 12')
    if not sequence_length == 5000:
        raise NameError('Length of signal must be 5000')
    return ecg_signal


def main():
    args = get_command_line()
    files = glob.glob(args['input'] + '/*.npy')
    os.makedirs(args['output'], exist_ok=True)
    print(len(files))
    for i,f in enumerate(files):
        ecg = loadecg(f)
        ecg_notch_removed = notch(ecg, 500)
        ecg_baseline_removed = baseline_wander_removal(ecg_notch_removed, 500)
        np.save(args['output'] + "/" + f.split("/")[-1], ecg_baseline_removed.astype(np.float16))
        print(i, "processed")
    print("COMPLETE")


if __name__ == '__main__':
    main()

