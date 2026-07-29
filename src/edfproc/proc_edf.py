#!/usr/bin/env python3
"""
Main script for processing multi-day EDF files containing ECG and Accelerometer data.

This script reads EDF files, processes the signals in daily chunks to manage memory,
performs quality control, calculates heart rate, HRV, and activity metrics,
imputes missing data, and generates a comprehensive summary report and plots for
each participant.
"""
__author__ = "Anna Bator"
__credits__ = "Stefan van Duijvenboden"
## This script is based on the original work by Stefan van Duijvenboden.

import os
import time
import traceback

import numpy as np
import pandas as pd
import pyedflib

from .logging_utils import configure_logging, get_logger
from .model_utils import find_model, model_fingerprint
from .plot_utils import (
    create_pdf_report,
    plot_24hr_profile_for_report,
    plot_activity_pie_chart,
    plot_daily_activity_bars,
    plot_hr_distribution,
    plotECG_failedQC,
    plotFunc,
)
from .proc_utils import doImp, downsampleECG, getQCmetrics, getQRS, getQRSmask
from .read_utils import prepSig, readACC, readEDFECG_info

log = get_logger("proc")

# Per-worker state. Populated by init_worker() in each pool process; on macOS
# and Windows the pool uses 'spawn', so nothing set in the parent is inherited.
m_qrs = None
_CONFIG = None
_MODEL_INFO = None


def init_worker(config, verbose=False):
    """Load the QRS model once per worker process.

    Loading in the initialiser rather than per file means the model is read
    from disk once per core instead of once per recording. Logging is
    reconfigured here because a spawned worker inherits no handlers.
    """
    global m_qrs, _CONFIG, _MODEL_INFO
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    configure_logging(verbose=verbose, include_process=True)

    # One TensorFlow thread per worker: the pool already provides parallelism,
    # and letting TF spawn its own threads on top oversubscribes the CPU.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    _CONFIG = config
    model_path = find_model(config.model_dir, config.model_path)
    _MODEL_INFO = model_fingerprint(model_path)
    m_qrs = load_model(str(model_path))
    log.debug("Worker ready, model %s loaded", model_path.name)


def compute_mad(ax, ay, az, epoch_samples):
    """
    Compute Mean Amplitude Deviation over fixed-length epochs.
    
    ax, ay, az: raw accelerometer arrays (in mg or g, consistent units)
    epoch_samples: number of samples per epoch (e.g. 500 for 5s at 100Hz)
    
    Returns array of MAD values, one per epoch, in same units as input.
    """
    # Vector magnitude at each sample
    vm = np.sqrt(ax**2 + ay**2 + az**2)
    
    n_epochs = len(vm) // epoch_samples
    mad_values = []
    
    for i in range(n_epochs):
        epoch = vm[i * epoch_samples : (i + 1) * epoch_samples]
        mad = np.mean(np.abs(epoch - np.mean(epoch)))
        mad_values.append(mad)
    
    return np.array(mad_values)


def procECG(f, i, chunk_samples, fname, cfg, model, signal_label='ECG', fs=250):
    """Processes a single chunk of ECG data."""
    nseg = cfg.segment_samples
    start = i * chunk_samples
    iECG = f.getSignalLabels().index(signal_label)
    ecg = f.readSignal(iECG, start=start, n=chunk_samples)
    ecg = ecg / 1000
    ecg, i_device_worn, ix_non_clipped, ix_pre_qc = prepSig(ecg=ecg, fs=fs, nseg=nseg)
    ix_qc = i_device_worn & ix_non_clipped & ix_pre_qc
    df_qc = pd.DataFrame({'device_worn': i_device_worn, 'clipped_5perc_thrs': ~ix_non_clipped, 'passed_initialQC': ix_qc})
    df_qc = df_qc.set_index(df_qc.index * cfg.segment_seconds)
    if not np.any(ix_qc):
        df_qc['passed_finalQC'] = False
        return df_qc
    ecg_dc = downsampleECG(ecg[ix_qc], fs_org=fs)
    ecg = ecg.flatten()
    qrs_mask = getQRSmask(ecg_dc, ix_qc, model)
    df_rw = getQRS(mask=qrs_mask, ecg=ecg)
    df_snr = pd.DataFrame(df_rw.groupby(df_rw.index).size(), columns=['N_beats'])
    idx_u = df_snr[df_snr['N_beats'] >= cfg.n_beats_min].index.unique()
    grouped_rw = df_rw.groupby(df_rw.index)
    t_rw_cache = {x: group['t_rw'].to_numpy() for x, group in grouped_rw}
    rr_lim_samples = [int(cfg.rr_min_ms / 1000 * fs), int(cfg.rr_max_ms / 1000 * fs)]
    results = [getQCmetrics(ecg, t_rw_cache[x], rr_lim=rr_lim_samples, nseg=nseg) for x in idx_u]
    df_metrics = pd.DataFrame(results, index=idx_u, columns=['N_RR', 'RRm', 'rr_Cover', 'rr_sd', 'rr_outliers', 'qrs_snr', 'qrs_amp', 'rmssd'])
    df_snr = df_snr.join(df_metrics, how='left')
    df_qc = df_qc.join(df_snr, how='left')
    df_qc['RRm'] = df_qc['RRm'] / fs * 1000
    df_qc['rr_sd'] = df_qc['rr_sd'] / fs * 1000
    df_qc['rmssd'] = df_qc['rmssd'] / fs * 1000
    # Summarise QC
    c1 = (df_qc['rr_outliers'] <= cfg.max_rr_outliers)
    c2 = (df_qc['rr_Cover'] >= cfg.rr_cover_min)
    c3 = (df_qc['N_beats'] >= cfg.n_beats_min)
    # Fill any NaN values with False. This is the crucial fix.
    # Segments that didn't have enough beats to calculate these metrics will now correctly fail.
    c1.fillna(False, inplace=True)
    c2.fillna(False, inplace=True)
    c3.fillna(False, inplace=True)

    df_qc['passed_finalQC'] = (c1 & c2 & c3 & df_qc['passed_initialQC'])
    df_qc.loc[~df_qc['passed_finalQC'], 'RRm'] = np.nan
    log.info("Processed day %d for %s", i + 1, fname)
    return df_qc


def _is_within_rest_window(index, cfg):
    """Boolean mask selecting timestamps inside the configured rest window.

    Handles windows that wrap past midnight (the usual case, e.g. 21:00-09:00)
    as well as same-day windows, so a shifted schedule can be configured
    without code changes.

    NOTE: this is a fixed clock window, not a per-participant estimate. Someone
    who habitually sleeps outside it has their resting HR and HRV computed from
    the wrong hours. Detecting the main rest period from accelerometer data
    directly would remove that assumption.
    """
    hours = index.hour
    if cfg.night_start_hour > cfg.night_end_hour:
        # Wraps past midnight: late evening OR early morning.
        return (hours >= cfg.night_start_hour) | (hours < cfg.night_end_hour)
    # Does not wrap: a single contiguous block within one day.
    return (hours >= cfg.night_start_hour) & (hours < cfg.night_end_hour)


def _rest_periods(df_10min, cfg):
    """10-minute windows that are both inside the rest window and low-movement."""
    in_window = _is_within_rest_window(df_10min.index, cfg)
    low_movement = df_10min['acc_imputed'] < cfg.sleep_threshold_mg
    return df_10min[in_window & low_movement].copy()


def calculate_summary_metrics(df_qc, cfg):
    """
    Resamples data to 10-minute windows and calculates robust summary metrics.
    """
    if df_qc.empty:
        return {} # Return an empty dictionary if there's no data

    # 1. Resample 10-second data to 10-minute averages of RRm, rmssd, and acc
    df_10min = df_qc.resample('10min', on='time').mean()

    # 2. Calculate the 10-minute average heart rate (just translate RR to HR)
    df_10min['HR_10min'] = 60 * 1000 / df_10min['RRm_imputed']

    # 3. Pick the 10-min segments with min, max, and mean avg HR
    summary = {
        'HR_min': df_10min['HR_10min'].min(),
        'HR_max': df_10min['HR_10min'].max(),
        'HR_mean': df_10min['HR_10min'].mean()
    }

    # 4. Isolate resting periods: inside the rest window and barely moving
    sleep_periods = _rest_periods(df_10min, cfg)
    log.debug("Found %d rest periods (10-min segments with acc < %s mg)",
              len(sleep_periods), cfg.sleep_threshold_mg)

    if not sleep_periods.empty:
        # 5. Calculate Resting HR and Resting HRV from these quiet periods
        summary['HR_rest_robust'] = sleep_periods['HR_10min'].median() # use median instead of mean for robustness
        summary['median_daily_rmssd'] = sleep_periods['rmssd'].median() # use median instead of mean for robustness
    else:
        # Provide fallback values if no resting periods are found
        summary['HR_rest_robust'] = np.nan
        summary['median_daily_rmssd'] = np.nan

    return summary


def calculate_daily_hrv_for_report(df_qc, cfg):
    """Calculates the daily median RMSSD from 10-minute sleep periods for the report table."""
    if df_qc.empty: return None

    df_10min = df_qc.resample('10min', on='time').mean()
    sleep_periods = _rest_periods(df_10min, cfg)

    if sleep_periods.empty:
        log.warning("No rest periods found within the %02d:00-%02d:00 window.",
                    cfg.night_start_hour, cfg.night_end_hour)
        return None

    sleep_periods['date'] = sleep_periods.index.date
    # Use .agg() to ensure the output is always a Pandas Series, even if only one night is present
    daily_hrv = sleep_periods.groupby('date')['rmssd'].agg('median')
    
    if daily_hrv.empty: 
        return None
        
    daily_hrv_summary = daily_hrv.to_frame(name='rmssd')
    try:
        daily_hrv_summary['norm_hrv'] = np.log(daily_hrv_summary['rmssd'].replace(0, np.nan))
    except TypeError:
        log.warning("Could not calculate normalised HRV (np.log failed). Skipping.")
        daily_hrv_summary['norm_hrv'] = np.nan
        
    return daily_hrv_summary


# Main Processing Function
def procEDF(edf_file, cfg, model, model_info=None):
    """Main processing pipeline for a single EDF file."""
    Ts = [['start', time.time()]]
    base_filename = os.path.basename(edf_file)
    output_dirname = os.path.splitext(base_filename)[0]

    subject_output_path = os.path.join(str(cfg.output_dir), output_dirname)
    plots_path = os.path.join(subject_output_path, "plots")
    data_path = os.path.join(subject_output_path, "processed_data")

    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    fs, start_time, dat_info = readEDFECG_info(edf_file)

    if dat_info.empty or dat_info['N_ecg'].iloc[0] == 0:
        log.warning("No data in: %s", edf_file)
        dat_info['failed'] = 1
        return dat_info

    # The QRS detector was trained at a fixed sample rate, and several helpers
    # assume it too. Processing at a different rate would produce plausible
    # but wrong beat timings, so make the mismatch impossible to miss.
    if int(fs) != int(cfg.fs_expected):
        log.warning(
            "%s is sampled at %s Hz but the QRS detector expects %s Hz. "
            "Results for this file are unlikely to be valid.",
            base_filename, fs, cfg.fs_expected)

    # Record which model produced these results, for provenance.
    if model_info:
        for key, value in model_info.items():
            dat_info[key] = value

    dat_info['failed'] = 0
    chunk_samples = int(fs * cfg.chunk_seconds)
    n_chunks = int(np.ceil(dat_info['N_ecg'].iloc[0] / chunk_samples))

    try:
        with pyedflib.EdfReader(edf_file) as f:
            df_qc_list = [procECG(f, i, chunk_samples, base_filename, cfg, model)
                          for i in range(n_chunks)]

        df_qc = pd.concat(
            [df.dropna(axis=1, how='all') for df in df_qc_list if df is not None and not df.empty],
            ignore_index=True
        )

        if df_qc.empty:
            raise ValueError("No valid ECG chunks found after processing.")

        df_qc.index = df_qc.index * cfg.segment_seconds
        df_qc['time'] = pd.to_datetime(start_time) + pd.to_timedelta(df_qc.index, unit='s')

        mean_qc = df_qc.loc[df_qc['device_worn'], 'passed_finalQC'].mean()
        if mean_qc < cfg.qc_warn_below:
            log.warning("Low data quality in %s: only %.1f%% of worn ECG passed QC.",
                        base_filename, mean_qc * 100)
            df_f = df_qc[(~df_qc['passed_finalQC']) & (df_qc['device_worn'])].sample(n=min(25, len(df_qc)))
            with pyedflib.EdfReader(edf_file) as f:
                plot_save_path = os.path.join(plots_path, base_filename + '_ECGs_failedQC.pdf')
                plotECG_failedQC(f, df_f, plot_save_path)

        Ts.append(['proc_ecg', time.time()])
        
        df_acc, dat_info_acc, _ = readACC(edf_file, start_time)
        dat_info = pd.concat([dat_info, dat_info_acc], axis=1)
        df_qc = df_qc.join(df_acc, how='left')
        df_qc.loc[~df_qc['device_worn'], 'acc'] = np.nan

        df_qc = doImp(df_qc, 'RRm')
        df_qc = doImp(df_qc, 'acc')
        
        # Calculate all summary metrics using the new 10-minute window method
        summary_metrics = calculate_summary_metrics(df_qc, cfg)

        # Update the main dat_info DataFrame with these new, robust values
        for key, value in summary_metrics.items():
            dat_info[key] = value

         # Calculate the daily HRV summary specifically for the report table
        daily_hrv_summary_for_report = calculate_daily_hrv_for_report(df_qc, cfg)

        # Final HR column for plotting
        df_qc['HRm_imputed'] = 60 * 1000 / df_qc['RRm_imputed']

        # Time in activity zones (Etzkorn et al. 2024 chest-worn MAD thresholds)
        thresholds = cfg.activity_thresholds
        hours_per_segment = cfg.segment_seconds / 3600
        acc_series = df_qc.loc[df_qc['device_worn'], 'acc_imputed']
        dat_info['hours_sleep_sedentary'] = (acc_series < thresholds['very_light']).sum() * hours_per_segment
        dat_info['hours_very_light']      = ((acc_series >= thresholds['very_light']) & (acc_series < thresholds['light'])).sum() * hours_per_segment
        dat_info['hours_light_activity']  = ((acc_series >= thresholds['light'])      & (acc_series < thresholds['moderate'])).sum() * hours_per_segment
        dat_info['hours_mvpa']            = (acc_series >= thresholds['moderate']).sum() * hours_per_segment

        # Final wrap-up stats
        dat_info['wear_time_ECG_10s'] = df_qc["device_worn"].mean()
        dat_info['prop_ECG_passed_finalQC'] = df_qc['passed_finalQC'].mean()
        dat_info['prop_ECG_worn_passed_finalQC'] = mean_qc # Taken from mean_qc calculation above
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
        daily_bars_path = plot_daily_activity_bars(df_qc.copy(), thresholds, save_path=os.path.join(plots_path, base_filename + "_daily_bars.png"))
        create_pdf_report(dat_info, subject_output_path, edf_file, thresholds, num_days, daily_bars_path, profile_plot_path, daily_hrv_summary_for_report, pie_chart_path, hr_dist_path)

        Ts.append(['create_report', time.time()])
        df_time = pd.DataFrame(Ts, columns=['task', 't'])
        df_time['dt'] = df_time['t'].diff().fillna(0)
        log.debug("Timings for %s:\n%s", base_filename, df_time[['task', 'dt']])

    except Exception as e:
        log.error("Processing failed for %s: %s", edf_file, e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        dat_info['failed'] = 1
        
    return dat_info


def procEDF_wrapper(edf_filename):
    """Entry point for multiprocessing.Pool.

    Reads the model and settings from the per-worker globals populated by
    init_worker(), because a loaded Keras model cannot be pickled and sent to
    a worker as an argument.
    """
    if m_qrs is None or _CONFIG is None:
        raise RuntimeError(
            "Worker was not initialised. multiprocessing.Pool must be created "
            "with initializer=init_worker and initargs=(config,)."
        )
    return procEDF(edf_filename, _CONFIG, m_qrs, _MODEL_INFO)