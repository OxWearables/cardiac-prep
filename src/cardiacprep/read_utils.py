import os
from datetime import datetime

import numpy as np
import pandas as pd
import pyedflib
from actipy.processing import calibrate_gravity
from scipy.signal import butter, filtfilt, iirnotch

from .logging_utils import get_logger

log = get_logger("read")



def readEDFECG_info(edfFile, signal_label='ECG'):
    log.info("Reading EDF: %s", os.path.basename(edfFile))
    f = pyedflib.EdfReader(edfFile)

    iECG = f.getSignalLabels().index(signal_label)
    units, fs = f.getSignalHeader(iECG)["dimension"], f.getSignalHeader(iECG)["sample_frequency"]

    L = f.getNSamples()[iECG]

    log.debug("Length: %s samples / %s hours / %s days",
              L, np.round(L / fs / 3600, 1), np.round(L / fs / 3600 / 24, 1))
    log.debug("Sample rate: %s Hz", fs)
    log.debug("Units: %s", units)
    
    # year, month, day, hour, minute, second, microsecond,
    start_time = datetime(
        year = f.startdate_year,
        month = f.startdate_month,
        day = f.startdate_day,
        hour = f.starttime_hour,
        minute = f.starttime_minute,
        second = f.starttime_second,
        microsecond = f.starttime_subsecond)
    
    f._close()
    log.debug("Finished reading header")

    # Tlim = Tlim*24*3600*fs
    # ecg = ecg[:Tlim]
    
    dat_info =  pd.DataFrame({
        'Name' : [os.path.basename(edfFile)],
        'Tstart': [start_time],
        'fs_ecg': [fs],
        'units_ecg': [units],
        'N_ecg': [L]
    })
    
    return fs, start_time, dat_info

def mean_amplitude_deviation(vm, epoch_samples):
    """Mean Amplitude Deviation of a vector-magnitude signal, per epoch.

    MAD is the mean absolute deviation from the epoch's own mean::

        MAD = mean(|VM - mean(VM)|)

    Subtracting the epoch mean removes the constant 1 g of gravity, which is
    why no separate detrending step is needed - and why adding one would be
    wrong. A high-pass filter ahead of this would strip out slow movement that
    the published cut-points were derived to include.

    Returns one value per epoch, in the units of ``vm``. A trailing partial
    epoch is measured on whatever samples it has rather than discarded, so a
    recording that is not a whole number of epochs long keeps its final
    minutes.
    """
    vm = np.asarray(vm, dtype="float64")
    if epoch_samples < 1:
        raise ValueError(f"epoch_samples must be at least 1, got {epoch_samples}")

    n_full = len(vm) // epoch_samples
    values = []

    if n_full:
        epochs = vm[: n_full * epoch_samples].reshape(n_full, epoch_samples)
        values.append(np.mean(np.abs(epochs - epochs.mean(axis=1, keepdims=True)), axis=1))

    remainder = vm[n_full * epoch_samples:]
    if remainder.size:
        values.append(np.array([np.mean(np.abs(remainder - remainder.mean()))]))

    return np.concatenate(values) if values else np.array([])


def readACC(edfFile, tstamp, clip_val=4000, T=10, do_cal=True, calib_cube=0.2,
            cal_stdtol=0.015, cal_win='10s', epoch_seconds=60):
    log.debug("Reading accelerometer data")
    f = pyedflib.EdfReader(edfFile)
    
    colnames = f.getSignalLabels()
    dat = list()
    for i in range(len(colnames)):
        if colnames[i].startswith('Accelerometer'):
            # print(colnames[i])
            dat.append(f.readSignal(i))
            units, fs = f.getSignalHeader(i)["dimension"], f.getSignalHeader(i)["sample_frequency"]
    f._close()

    dat_info =  pd.DataFrame({
        'fs_acc': [fs],
        'units_acc': [units],
        'N_acc': [len(dat[0])],
    })

    dat = np.vstack(dat).astype('float32')#[:,:int(3*3600*fs)]
    # dat = np.vstack(dat).astype('float16')#[:,:int(3*3600*fs)]
    
    
    # remove >14 days?
    # Tlim = Tlim*24*3600*fs
    # dat = dat[:,:Tlim]
    
    # clipped
    dat_c = np.abs(np.max(dat,axis=0))>=clip_val # flag clipped values

    if do_cal:
        
        dat = dat / 1000 # to g
        time_intervals = np.arange(dat.shape[1]) / fs  # Time in seconds
        t = pd.to_datetime(tstamp) + pd.to_timedelta(time_intervals, unit='s')
        # print(dat.shape)
        dat = pd.DataFrame({"time": t, "x": dat[0],"y": dat[1],"z": dat[2] })
        dat = dat.set_index("time")
        
        dat = calibrate_gravity(dat,window=cal_win,stdtol=cal_stdtol,calib_cube=calib_cube)

        dat_info = pd.concat([dat_info, pd.DataFrame([dat[1]])], axis=1)
        dat = dat[0][['x','y','z']].to_numpy()
        # Vector magnitude in milli-g, gravity included. MAD removes it below.
        dat = 1000 * np.linalg.norm(dat, axis=-1)

    else:
        dat = np.linalg.norm(dat, axis=0)

    # Movement is Mean Amplitude Deviation, the quantity the activity
    # cut-points in Etzkorn et al. (2024) were derived from. They were
    # published at minute level, so the epoch defaults to 60 seconds; a
    # shorter epoch gives MAD a wider distribution and would bias the time
    # spent in each intensity band.
    epoch_samples = max(1, int(round(epoch_seconds * fs)))
    mad = mean_amplitude_deviation(dat, epoch_samples)

    # One MAD value covers a whole epoch, so every sample inside that epoch
    # carries it. Binning below then reduces it to the analysis resolution
    # without changing the value.
    dat = np.repeat(mad, epoch_samples)[:len(dat_c)]
    if len(dat) < len(dat_c):  # trailing partial epoch
        dat = np.concatenate([dat, np.full(len(dat_c) - len(dat), mad[-1])])

    dat = pd.DataFrame({"bin":((np.arange(len(dat))/fs) // T).astype(int) * T, "acc": dat, "acc_clipped": dat_c})
    dat = dat.set_index('bin')
 
    return dat.groupby(dat.index)[['acc', 'acc_clipped']].mean(), dat_info, dat
    
def prepSig(ecg,nseg=2500,fs=250, clip_val=4,var_range=[0.0001,2], min_ptp=0.025, fs_filt=[2,40],
            mains_hz=50.0, mains_q=30.0):
    
    if (len(ecg) % nseg)>0: # pad if needed
        pad_size = nseg - len(ecg) % nseg # padding size
        ecg = np.pad(ecg, (0, pad_size), mode='edge')
    
    ecg = ecg.reshape(-1,nseg)
    # Non-wear is a perfectly flat trace. Peak-to-peak rather than standard
    # deviation, because np.std of a constant array can return a value around
    # 1e-24 from rounding in the mean, whereas max-minus-min of identical
    # values is exactly zero. With a > 0 test the former reports non-wear as
    # worn, for some constant values but not others.
    i_device_worn = np.ptp(ecg, axis=-1) > 0

    
    # clip, only accept ECGs with <5% clipped values
    ix_non_clipped = np.mean(np.abs(ecg)>clip_val,axis=-1)<.05

    ecg = np.clip(ecg.flatten(), -clip_val, clip_val)
    # Mains notch filter, yes - I have seen extreme noise in this band despite wearable device.
    # 50 Hz across most of the world, 60 Hz in North America and parts of Asia.
    # Q is the quality factor (higher = narrower notch).
    b, a = iirnotch(mains_hz, mains_q, fs)
    ecg = filtfilt(b,a,ecg).astype("float32")
    # ecg = filtfilt(b,a,ecg).astype("float16")

    # filter other bands
    w = np.array(fs_filt) / (fs / 2) # Normalize the frequency
    b, a = butter(4, w, 'bandpass')    
    ecg = filtfilt(b,a,ecg).astype("float32")

    ecg = ecg.reshape(-1,nseg)

    # noise assessment
    var = np.var(ecg, axis=1)
    ptp = np.ptp(ecg, axis=1)
    
    ix_qc = (var >= var_range[0]) & (var <= var_range[1]) & (ptp >= min_ptp)
    
    
    return ecg, i_device_worn, ix_non_clipped, ix_qc

    