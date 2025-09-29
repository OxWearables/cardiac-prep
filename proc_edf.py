#!/usr/bin/env python3
"""
Main script for processing multi-day EDF files containing ECG and Accelerometer data.

This script reads EDF files, processes the signals in daily chunks to manage memory,
performs quality control, calculates heart rate, HRV, and activity metrics,
imputes missing data, and generates a comprehensive summary report and plots for
each participant.
"""
__author__ = "Awa Bator"
__credits__ = "Stefan van Duijvenboden"
## This script is based on the original work by Stefan van Duijvenboden.

import time
import glob
import os
import pyedflib
import numpy as np
import pandas as pd
import multiprocessing as mp
import traceback
from read_utils import readEDFECG_info, readACC, prepSig
from proc_utils import doImp, getQRSmask, getQCmetrics, getQRS, downsampleECG
from plot_utils import (plotFunc, plotECG_failedQC, create_pdf_report, 
                        plot_hr_distribution, plot_activity_pie_chart, 
                        plot_daily_activity_bars, plot_24hr_profile_for_report)

# Global Settings
NSEG = 2500  # 10s segment length in samples at 250Hz
Nchunk = int(3600 * 24)  # Process data in 24-hour chunks

# ECG QC settings
RRCOVER_LIM = 0.75
NBEATS_LIM = 5
RRmin = 250  # 0.25 seconds in ms
RRmax = 2500  # 2.5 seconds in ms
N_RR_outliers_max = 1
acc_sedentary_thrs = 4  # in mg

# Activity Thresholds in milli-g (mg)
ACTIVITY_THRESHOLDS = {
    'light': 15,  # Start of light activity
    'moderate': 50, # Start of moderate activity (e.g., brisk walking)
    'vigorous': 120 # Start of vigorous activity (e.g., running, cycling)
}

# New, lower threshold to define resting/sleep periods (hypothetical value for now)
SLEEP_THRS = 5 # in mg

# Multiprocessing and ECG Processing Functions 
def init_worker():
    """Initialiser function for the multiprocessing pool to load the ML model."""
    global m_qrs
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    m_qrs = load_model("./models/QRS_detector_125Hz_080525.keras")


def procECG(f, i, chunk_samples, signal_label='ECG', fs=250):
    """Processes a single chunk of ECG data."""
    # This function is correct and remains unchanged.
    start = i * chunk_samples
    iECG = f.getSignalLabels().index(signal_label)
    ecg = f.readSignal(iECG, start=start, n=chunk_samples)
    ecg = ecg / 1000
    ecg, i_device_worn, ix_non_clipped, ix_pre_qc = prepSig(ecg=ecg, fs=fs, nseg=NSEG)
    ix_qc = i_device_worn & ix_non_clipped & ix_pre_qc
    df_qc = pd.DataFrame({'device_worn': i_device_worn, 'clipped_5perc_thrs': ~ix_non_clipped, 'passed_initialQC': ix_qc})
    df_qc = df_qc.set_index(df_qc.index * 10)
    if not np.any(ix_qc):
        df_qc['passed_finalQC'] = False
        return df_qc
    ecg_dc = downsampleECG(ecg[ix_qc], fs_org=fs)
    ecg = ecg.flatten()
    qrs_mask = getQRSmask(ecg_dc, ix_qc, m_qrs)
    df_rw = getQRS(mask=qrs_mask, ecg=ecg)
    df_snr = pd.DataFrame(df_rw.groupby(df_rw.index).size(), columns=['N_beats'])
    idx_u = df_snr[df_snr['N_beats'] >= NBEATS_LIM].index.unique()
    grouped_rw = df_rw.groupby(df_rw.index)
    t_rw_cache = {x: group['t_rw'].to_numpy() for x, group in grouped_rw}
    rr_lim_samples = [int(RRmin / 1000 * fs), int(RRmax / 1000 * fs)]
    results = [getQCmetrics(ecg, t_rw_cache[x], rr_lim=rr_lim_samples) for x in idx_u]
    df_metrics = pd.DataFrame(results, index=idx_u, columns=['N_RR', 'RRm', 'rr_Cover', 'rr_sd', 'rr_outliers', 'qrs_snr', 'qrs_amp', 'rmssd'])
    df_snr = df_snr.join(df_metrics, how='left')
    df_qc = df_qc.join(df_snr, how='left')
    df_qc['RRm'] = df_qc['RRm'] / fs * 1000 
    df_qc['rr_sd'] = df_qc['rr_sd'] / fs * 1000 
    df_qc['rmssd'] = df_qc['rmssd'] / fs * 1000 
    # Summarise QC
    c1 = (df_qc['rr_outliers'] <= N_RR_outliers_max)
    c2 = (df_qc['rr_Cover'] >= RRCOVER_LIM)
    c3 = (df_qc['N_beats'] >= NBEATS_LIM)
    # Fill any NaN values with False. This is the crucial fix.
    # Segments that didn't have enough beats to calculate these metrics will now correctly fail.
    c1.fillna(False, inplace=True)
    c2.fillna(False, inplace=True)
    c3.fillna(False, inplace=True)

    df_qc['passed_finalQC'] = (c1 & c2 & c3 & df_qc['passed_initialQC'])
    df_qc.loc[~df_qc['passed_finalQC'], 'RRm'] = np.nan
    print(f"--> processed day: {i+1}")
    return df_qc


def calculate_daily_hrv_summary(df_qc, acc_sedentary_thrs):
    """Calculates the daily median of 5-minute median RMSSD values during sedentary periods."""
    # This function is correct.
    df_hrv = df_qc[(df_qc['passed_finalQC']) & (df_qc['acc_imputed'] < acc_sedentary_thrs)].copy()
    if df_hrv.empty: return None
    df_hrv['date'] = df_hrv['time'].dt.date
    df_hrv['bin_5min'] = df_hrv.index // 300
    rmssd_5min = df_hrv.groupby(['date', 'bin_5min'])['rmssd'].median()
    daily_hrv = rmssd_5min.groupby('date').median()
    if daily_hrv.empty: return None
    daily_hrv_summary = daily_hrv.to_frame(name='rmssd')
    daily_hrv_summary['norm_hrv'] = np.log(daily_hrv_summary['rmssd'].replace(0, np.nan))
    return daily_hrv_summary


def calculate_summary_metrics(df_qc, sleep_thrs):
    """
    Resamples data to 1-minute windows and calculates robust summary metrics.
    """
    if df_qc.empty:
        return {} # Return an empty dictionary if there's no data

    # 1. Resample 10-second data to 1-minute averages of RRm, rmssd, and acc
    df_1min = df_qc.resample('1min', on='time').mean()

    # 2. Calculate the 1-minute average heart rate (just translate RR to HR)
    df_1min['HR_1min'] = 60 * 1000 / df_1min['RRm_imputed']
    
    # 3. Pick the 1-min segments with min, max, and mean avg HR
    summary = {
        'HR_min': df_1min['HR_1min'].min(),
        'HR_max': df_1min['HR_1min'].max(),
        'HR_mean': df_1min['HR_1min'].mean()
    }

    # 4. Isolate resting (sleep) periods using the low acceleration threshold
    resting_periods = df_1min[df_1min['acc_imputed'] < sleep_thrs]

    if not resting_periods.empty:
        # 5. Calculate Resting HR and Resting HRV from these quiet periods
        summary['HR_rest_robust'] = resting_periods['HR_1min'].median() # use median instead of mean for robustness
        summary['median_daily_rmssd'] = resting_periods['rmssd'].median() # use median instead of mean for robustness
    else:
        # Provide fallback values if no resting periods are found
        summary['HR_rest_robust'] = np.nan
        summary['median_daily_rmssd'] = np.nan
        
    return summary

# Main Processing Function
def procEDF(edf_file, m_qrs):
    """Main processing pipeline for a single EDF file."""
    Ts = [['start', time.time()]]
    base_filename = os.path.basename(edf_file)
    output_dirname = os.path.splitext(base_filename)[0]
    
    subject_output_path = os.path.join("./output/", output_dirname)
    plots_path = os.path.join(subject_output_path, "plots")
    data_path = os.path.join(subject_output_path, "processed_data")

    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    fs, start_time, dat_info = readEDFECG_info(edf_file)

    if dat_info.empty or dat_info['N_ecg'].iloc[0] == 0:
        print(f"Warning: no data in: {edf_file}")
        dat_info['failed'] = 1
        return dat_info

    dat_info['failed'] = 0
    chunk_samples = int(fs * Nchunk)
    n_chunks = int(np.ceil(dat_info['N_ecg'].iloc[0] / chunk_samples))

    try:
        with pyedflib.EdfReader(edf_file) as f:
            df_qc_list = [procECG(f, i, chunk_samples) for i in range(n_chunks)]
        
        df_qc = pd.concat([df for df in df_qc_list if df is not None and not df.empty], ignore_index=True)

        if df_qc.empty:
            raise ValueError("No valid ECG chunks found after processing.")

        df_qc.index = df_qc.index * int(NSEG / fs)
        df_qc['time'] = pd.to_datetime(start_time) + pd.to_timedelta(df_qc.index, unit='s')
        
        mean_qc = df_qc.loc[df_qc['device_worn'], 'passed_finalQC'].mean()
        if mean_qc < 0.9:
            print(f"Warning: Low data quality. Only {mean_qc:.1%} of ECG passed final QC.")
            df_f = df_qc[(~df_qc['passed_finalQC']) & (df_qc['device_worn'])].sample(n=min(25, len(df_qc)))
            with pyedflib.EdfReader(edf_file) as f:
                plot_save_path = os.path.join(plots_path, base_filename + '_ECGs_failedQC.pdf')
                plotECG_failedQC(f, df_f, plot_save_path)

        Ts.append(['proc_ecg', time.time()])
        
        df_acc, dat_info_acc, _ = readACC(edf_file, start_time)
        dat_info = pd.concat([dat_info, dat_info_acc], axis=1)
        df_qc = df_qc.join(df_acc, how='left')
        df_qc.loc[~df_qc['device_worn'], 'acc'] = np.nan

        high_quality_rrm = df_qc.loc[df_qc['passed_finalQC'], 'RRm']
        if not high_quality_rrm.empty:
            dat_info['HR_min'] = 60 * 1000 / high_quality_rrm.max()
            dat_info['HR_max'] = 60 * 1000 / high_quality_rrm.min()
        else:
            dat_info['HR_min'], dat_info['HR_max'] = np.nan, np.nan

        df_qc = doImp(df_qc, 'RRm')
        df_qc = doImp(df_qc, 'acc')
        
        # Calculate all summary metrics using the new 1-minute window method
        summary_metrics = calculate_summary_metrics(df_qc, SLEEP_THRS)
        
        # Update the main dat_info DataFrame with these new, robust values
        for key, value in summary_metrics.items():
            dat_info[key] = value

        # The daily HRV summary for the report table is still useful
        daily_hrv_summary = calculate_daily_hrv_summary(df_qc, SLEEP_THRS)

        # Final HR column for plotting
        df_qc['HRm_imputed'] = 60 * 1000 / df_qc['RRm_imputed']
        
        # Time in activity zones calculation (uses the new chest thresholds)
        acc_series = df_qc.loc[df_qc['device_worn'], 'acc_imputed']
        dat_info['hours_sedentary'] = (acc_series < ACTIVITY_THRESHOLDS['light']).sum() * 10 / 3600
        dat_info['hours_light_activity'] = ((acc_series >= ACTIVITY_THRESHOLDS['light']) & (acc_series < ACTIVITY_THRESHOLDS['moderate'])).sum() * 10 / 3600
        dat_info['hours_moderate_activity'] = ((acc_series >= ACTIVITY_THRESHOLDS['moderate']) & (acc_series < ACTIVITY_THRESHOLDS['vigorous'])).sum() * 10 / 3600
        dat_info['hours_vigorous_activity'] = (acc_series >= ACTIVITY_THRESHOLDS['vigorous']).sum() * 10 / 3600
        
        # Final wrap-up stats
        dat_info['wear_time_ECG_10s'] = df_qc["device_worn"].mean()
        dat_info['prop_ECG_passed_finalQC'] = df_qc['passed_finalQC'].mean()
        dat_info['frac_RR_imp'] = df_qc['RRm_isImputed'].mean()
        
        df_qc.to_csv(os.path.join(data_path, base_filename + "_df_qc.csv.gz"), compression='gzip')
        plotFunc(df_qc=df_qc.copy(), outpath=plots_path, edf_file=edf_file, mrk_hr='HRm_imputed', mrk_acc='acc_imputed')

        profile_plot_path = None
        # Check if the recording is long and complete enough
        if (dat_info.loc[0,'N_ecg']/fs/3600/24 > 3) and (df_qc['time'].dt.hour.nunique() == 24):
            # Create named Series for grouping to avoid column name conflicts
            hour_of_day = df_qc['time'].dt.hour.rename('hour')
            minute_of_hour = df_qc['time'].dt.minute.rename('minute')
            
            # Perform the groupby and reset the index
            df_24hr = df_qc.groupby([hour_of_day, minute_of_hour])[['RRm_imputed']].median().reset_index()
            
            # Rename the data column to be consistent
            df_24hr.rename(columns={'RRm_imputed': 'RRm_median'}, inplace=True)
            
            # Now pass the clean DataFrame with columns ['hour', 'minute', 'RRm_median']
            profile_plot_path = plot_24hr_profile_for_report(
                df_24hr,
                save_path=os.path.join(plots_path, base_filename + "_24hr_profile.png")
            )
        
        num_days = df_qc['time'].dt.date.nunique()
        pie_chart_path = plot_activity_pie_chart(dat_info, save_path=os.path.join(plots_path, base_filename + "_activity_pie.png"))
        hr_dist_path = plot_hr_distribution(df_qc, save_path=os.path.join(plots_path, base_filename + "_hr_distribution.png"))
        daily_bars_path = plot_daily_activity_bars(df_qc.copy(), ACTIVITY_THRESHOLDS, save_path=os.path.join(plots_path, base_filename + "_daily_bars.png"))
        
        create_pdf_report(dat_info, subject_output_path, edf_file, ACTIVITY_THRESHOLDS, num_days, daily_bars_path, profile_plot_path, daily_hrv_summary, pie_chart_path, hr_dist_path)
        
        Ts.append(['create_report', time.time()])
        df_time = pd.DataFrame(Ts, columns=['task', 't'])
        df_time['dt'] = df_time['t'].diff().fillna(0)
        print(df_time[['task', 'dt']])

    except Exception as e:
        print(f"[ERROR] procEDF failed in {edf_file}: {e}")
        traceback.print_exc()
        dat_info['failed'] = 1
        
    return dat_info


def procEDF_wrapper(edf_filename):
    """A simple wrapper for use with multiprocessing.Pool."""
    return procEDF(edf_filename, m_qrs)


if __name__ == "__main__":
    # This block is for DNAnexus execution and can be ignored for local runs
    import dxpy
    @dxpy.entry_point('main')
    def main(input_names, i_job):
        pass
    dxpy.run()