import os
from datetime import datetime

import numpy as np
import pandas as pd
import pyedflib
from actipy.processing import calibrate_gravity
from scipy.ndimage import median_filter
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

def readACC(edfFile, tstamp, clip_val=4000,T=10, do_cal=True, calib_cube=0.2, cal_stdtol=0.015, cal_win='10s',m_filt_size=120):
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
        dat = 1000 * (np.linalg.norm(dat,axis=-1) - 1)
        
    else:
        dat = np.linalg.norm(dat,axis=0) - 1000


    # do median filter to get these step functions out
    dat = dat - median_filter(dat, size=m_filt_size,axes=1)
    
    dat[dat<0] = 0 # remove negative values

    # if len(acc) % NSEG_A>0: # pad if needed
    # pad_size = NSEG_A - len(acc) % NSEG_A # padding size
    # acc = np.pad(acc, (0, pad_size))
    
    dat = pd.DataFrame({"bin":((np.arange(len(dat))/fs) // T).astype(int) * T, "acc": dat, "acc_clipped": dat_c})
    dat = dat.set_index('bin')
 
    return dat.groupby(dat.index)[['acc', 'acc_clipped']].mean(), dat_info, dat
    
def prepSig(ecg,nseg=2500,fs=250, clip_val=4,var_range=[0.0001,2], min_ptp=0.025, fs_filt=[2,40]):
    
    if (len(ecg) % nseg)>0: # pad if needed
        pad_size = nseg - len(ecg) % nseg # padding size
        ecg = np.pad(ecg, (0, pad_size), mode='edge')
    
    ecg = ecg.reshape(-1,nseg)
    i_device_worn = np.std(ecg,axis=-1)>0

    
    # clip, only accept ECGs with <5% clipped values
    ix_non_clipped = np.mean(np.abs(ecg)>clip_val,axis=-1)<.05

    ecg = np.clip(ecg.flatten(), -clip_val, clip_val)
    # 50Hz notch filter, yes - I have seen extreme noise in this band despite wearable device 
    # Notch filter design
    f0 = 50.0  # Frequency to remove (Hz)
    Q = 30.0   # Quality factor (higher = narrower notch)

    # Design notch filter
    b, a = iirnotch(f0, Q, fs)
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

    