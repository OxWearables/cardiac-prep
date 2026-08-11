__author__ = "Stefan van Duijvenboden"


import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing
from scipy.signal import decimate, detrend
from sklearn.preprocessing import StandardScaler


# noise dectection
def downsampleECG(ecg,fs_org, fs=125,thrs_mar=1.0):
    ecg_dc = decimate(ecg,int(fs_org/fs))
    # clip
    amp_val = np.max(ecg_dc,axis=-1)
    amp_val = amp_val[amp_val>0]
    if amp_val.size > 0:
        val_up = np.median(amp_val) + thrs_mar
    else:
        val_up = thrs_mar  # fallback upper clip
    
    amp_val = np.min(ecg_dc,axis=-1)
    amp_val = amp_val[amp_val<0]
    if amp_val.size > 0:
        val_low = np.median(amp_val) - thrs_mar
    else:
        val_low = -thrs_mar  # fallback lower clip
    

    ecg_dc = np.clip(ecg_dc,a_min=val_low,a_max=val_up)

    # scale
    scaler = StandardScaler()
    ecg_dc = np.expand_dims(ecg_dc, -1)
    for i in range(ecg_dc.shape[0]):
        ecg_dc[i] = scaler.fit_transform(ecg_dc[i])

    return ecg_dc


def getSNR(X):
    X = detrend(X,type='constant',axis=-1)
    Xm = np.median(X, axis=0)                    # Median QRS (signal template)
    signal_power = np.mean(Xm ** 2)              # Power of the median template
    noise_power = np.mean((X - Xm) ** 2)         # Mean squared deviation from template
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR, np.ptp(Xm)


def getQRSmask(ecg_dc, ix_qc, m_qrs, output_size=250, structure_size=2, threshold=0.5):
    # derive QRS label mask, only analyse sections that passed QC

    y_hat = np.zeros((len(ix_qc),output_size)).astype("bool")
    y_hat[ix_qc] = m_qrs.predict(ecg_dc, verbose=0).squeeze() > threshold
    y_hat = y_hat.flatten()

    # filtering
    pad_w = structure_size // 2
    y_hat = np.pad(y_hat, pad_width=pad_w, mode='edge')
    # Apply closing
    y_hat = binary_closing(y_hat, structure=np.ones(structure_size))
    # Remove padding
    y_hat = y_hat[pad_w:-pad_w]
    # y_hat = binary_closing(y_hat, structure=filt_struct)

    return y_hat
    

def getQRS(ecg, mask, fs=250, mask_output_size=250, T=10):

    if not np.any(mask):
        return pd.DataFrame({'t_rw': []})

    N = int((fs*T)/mask_output_size)
    
    mask = np.pad(mask, (1,1))
    diff = np.diff(mask.astype(int))

    ix0, ix1 = np.where(diff == 1)[0], np.where(diff == -1)[0]
    ix0, ix1 = (ix0 * N).astype(int), (ix1 * N).astype(int)

    rw_max = np.zeros(len(ix0)).astype(int)
    rw_min = np.zeros(len(ix0)).astype(int)

    for i, (ix0_, ix1_) in enumerate(zip(ix0, ix1)):
        w = ecg[ix0_:ix1_] # this is not a copy but a slice, total size of x doesn't affect slicing speed directly
        rw_max[i] = np.argmax(w) + ix0_
        rw_min[i] = np.argmin(w) + ix0_


    # check on T basis QRS polarity to determine R-peak
    df_rr = pd.DataFrame({'rw_t_max': rw_max, 'rw_t_min': rw_min,
                          'rw_amp_max': np.abs(ecg[rw_max]), 'rw_amp_min': np.abs(ecg[rw_min])})

    del rw_max, rw_min
    
    df_rr['rw_time'] = df_rr[['rw_t_max', 'rw_t_min']].max(axis=1)
    df_rr['bin'] = (df_rr['rw_time']/fs // T).astype(int) * T
    
    # we are now going to calculate the Rw  (either max or min), and QCs per bin (=10s)
    median_rr = df_rr.groupby('bin')[['rw_amp_max','rw_amp_min']].median() # min or max
    median_rr['rw_selector'] = median_rr['rw_amp_max']>median_rr['rw_amp_min']
    median_rr = median_rr[['rw_selector']]  # false means take rw_min
    
    df_rr = df_rr.set_index('bin').join(median_rr, how='left')
    del median_rr
    
    df_rr['t_rw'] = df_rr['rw_t_max'].where(df_rr['rw_selector'], df_rr['rw_t_min'])
    df_rr = df_rr[['t_rw']] # bin and Rw is sample

    return df_rr    


def getQCmetrics(ecg, rw, wl_qrs = 15, nseg=2500 ,rr_lim=[50, 1250], fs=250, rr_outlier_factor=1.8):

    # w = fc_hp / (fs / 2) # Normalise the frequency
    # b, a = butter(4, w, 'high')
    # ecg = filtfilt(b,a,ecg)
    
    rr_unfiltered = np.diff(rw)

    # Filter out non-physiological RR intervals FIRST
    rr = rr_unfiltered[(rr_unfiltered < rr_lim[1]) & (rr_unfiltered > rr_lim[0])]

    # Now, calculate RMSSD on the CLEANED rr intervals
    if len(rr) > 1:
        # Calculate successive differences, square them, get the mean, then the square root
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    else:
        rmssd = np.nan

    # Calculate other metrics on the cleaned rr intervals
    rrM, rrC, rrsd = np.median(rr), np.sum(rr)/nseg, np.std(rr)
    rr_outliers = np.sum(rr > rr_outlier_factor * rrM)
    
    win_qrs = rw[:, np.newaxis] + np.arange(-wl_qrs, wl_qrs)
    win_qrs = ecg[np.clip(win_qrs, 0, len(ecg) - 1)] # maybe not?
    snr, amp = getSNR(win_qrs)

    return len(rr),rrM,rrC,rrsd,rr_outliers, snr, amp, rmssd  # => qrs_amp, qrs_snr 


def doImp(df_qc,mrk_name,gap_lim=600, tseg=10):
    """
    Impute missing values in a time-indexed pandas DataFrame.
    
    - Linearly interpolate gaps ≤ gap_lim (in seconds).
    - Fill longer gaps using average values from same time-of-day bin across minutes.
    
    Parameters:
    - df_qc: DataFrame with time index in seconds.
    - mrk_name: column name to impute.
    - gap_lim: maximum gap length (in seconds) to interpolate (default: 10 min = 600).
    - tseg: sampling interval in seconds (default: 10s).
    
    Returns:
    - df_qc with two new columns: `{mrk_name}_imputed` and `{mrk_name}_was_imputed`
    """
    mrk_imp_name = mrk_name + "_imputed"
    mrk_was_imp = mrk_name + "_isImputed"
    

    is_valid = ~df_qc[mrk_name].isna()

    # determine what we can interpolate based on temporal information:
    # inear interpolation across a short series of NaN values only makes sense 
    # if enough valid data exists on both sides of the gap.
    # only interpolate if 50% of data

    win_size = int(gap_lim/tseg)
    valid_ratio = (df_qc[mrk_name].rolling(window=win_size, min_periods=1).apply(lambda x: (~np.isnan(x)).mean(), raw=True)) > 0.5
    
    unable_to_interp = (~is_valid) & (~valid_ratio)

    # interpolation based on temporal info
    df_qc[mrk_imp_name] = df_qc[mrk_name].interpolate(method='linear', limit_direction='both')
    df_qc.loc[unable_to_interp, mrk_imp_name] = np.nan  # not needed, as we will overwrite the, but just in case preserve missing where not interpolating

    # if any values missing >1hr? should be checked in dat_info before accepting this signal

    if np.any(unable_to_interp):
        # Add time-of-day helper column
        df_qc['tod_bin'] = df_qc['time'].dt.strftime('%H:%M')
        mean_by_tod = df_qc[~unable_to_interp].groupby('tod_bin')[mrk_imp_name].mean()
        # missing value simply remains NaN
        df_qc.loc[unable_to_interp, mrk_imp_name] = df_qc.loc[unable_to_interp, 'tod_bin'].map(mean_by_tod)
        df_qc.drop(columns=['tod_bin'], inplace=True)
        
    # Optional: flag which values were imputed
    df_qc[mrk_was_imp] = ~is_valid
    
    # rename to RAW so it's clear that we have raw data and imputed
    raw_col_name = f"{mrk_name}_raw"
    df_qc.rename(columns={mrk_name: raw_col_name}, inplace=True)

    return df_qc
