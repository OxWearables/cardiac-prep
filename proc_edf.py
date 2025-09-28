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
#!/usr/bin/env python3

import time
import glob
import os
import pyedflib
import numpy as np
import pandas as pd
from read_utils import readEDFECG_info, readACC, prepSig
from proc_utils import doImp, getQRSmask, getQCmetrics, getQRS, downsampleECG
from plot_utils import plotFunc, plotECG_failedQC, create_pdf_report, plot_hr_distribution, plot_activity_pie_chart, plot_daily_activity_bars, plot_24hr_profile_for_report
import multiprocessing as mp

os.makedirs("./plots/", exist_ok=True)

Tlim = 3600*24*3
NSEG = 2500 # 10s segment length in samples at 250Hz
fs_algorithm = 125 
Nchunk = int(3600*24) # Process data in 24-hour chunks
# ECG final QC settings
RRCOVER_LIM = 0.75
NBEATS_LIM = 5
RRmin = 0.25 # in seconds
RRmax = 2.5  # in seconds
N_RR_outliers_max = 1
acc_sedentary_thrs = 4 # in mg
# Activity Thresholds in milli-g (mg)
ACTIVITY_THRESHOLDS = {
    'light': 20,
    'moderate': 80,
    'vigorous': 150
}


def init_worker():
    """Initialiser function for the multiprocessing pool to load the ML model."""
    global m_qrs
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    m_qrs = load_model("./models/QRS_detector_125Hz_080525.keras")

def procECG(f, i, chunk_samples, signal_label='ECG', fs=250):
    """
    Processes a single chunk of ECG data.
    """
    start = i * chunk_samples

    iECG = f.getSignalLabels().index(signal_label)
    ecg = f.readSignal(iECG, start=start, n=chunk_samples)
    ecg = ecg / 1000 # to mV

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
    idx_u = df_snr[df_snr['N_beats']>=NBEATS_LIM].index.unique()
    
    grouped_rw = df_rw.groupby(df_rw.index) 
    t_rw_cache = {x: group['t_rw'].to_numpy() for x, group in grouped_rw}

    results = [getQCmetrics(ecg, t_rw_cache[x], rr_lim=[int(RRmin*fs), int(RRmax*fs)]) for x in idx_u]

    df_metrics = pd.DataFrame(results, index=idx_u, columns=['N_RR', 'RRm', 'rr_Cover','rr_sd', 'rr_outliers' ,'qrs_snr', 'qrs_amp', 'rmssd'])
    df_snr = df_snr.join(df_metrics, how='left')
    df_qc = df_qc.join(df_snr, how='left')
    
    df_qc['RRm'] = df_qc['RRm'] / fs * 1000 
    df_qc['rr_sd'] = df_qc['rr_sd'] / fs * 1000 
    df_qc['rmssd'] = df_qc['rmssd'] / fs * 1000 

    c1 = (df_qc['rr_outliers'] <= N_RR_outliers_max)
    c2 = (df_qc['rr_Cover'] >= RRCOVER_LIM)
    c3 = (df_qc['N_beats'] >= NBEATS_LIM)

    df_qc['passed_finalQC'] = (c1 & c2 & c3 & df_qc['passed_initialQC'])
    df_qc.loc[~df_qc['passed_finalQC'], 'RRm'] = np.nan
    
    print(f"--> processed day: {i+1}")
    
    return df_qc


def procEDF(edf_file, m_qrs):
    """
    Main processing pipeline for a single EDF file.
    """
    Ts = [['start',time.time()]]
    base_filename = os.path.basename(edf_file)
    output_dirname = os.path.splitext(base_filename)[0]
    
    subject_output_path = os.path.join("./output/", output_dirname)
    plots_path = os.path.join(subject_output_path, "plots")
    data_path = os.path.join(subject_output_path, "processed_data")

    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    fs, start_time, dat_info = readEDFECG_info(edf_file)

    if dat_info['N_ecg'][0] == 0:
        print(f"Warning: no data in: {edf_file}")
        dat_info['failed'] = 1
        return dat_info

    dat_info['failed'] = 0
    daily_hrv_summary = None

    chunk_samples = int(fs * Nchunk)
    n_chunks = int(np.ceil(dat_info['N_ecg'][0] / chunk_samples))

    with pyedflib.EdfReader(edf_file) as f:
        df_qc = []
        for i in range(n_chunks):
            try:
                df = procECG(f, i, chunk_samples)
                if df is not None and not df.empty:
                    df_qc.append(df)
            except Exception as e:
                print(f"[ERROR] procECG failed on chunk {i} in {edf_file}: {e}")

    if not df_qc:
        print(f"[ERROR] No valid ECG chunks found in: {edf_file}")
        dat_info['failed'] = 1
        return dat_info

    df_qc = pd.concat(df_qc, ignore_index=True)
        
    try:
        df_qc.index = df_qc.index * int(NSEG/fs)
        df_qc['time'] = pd.to_datetime(start_time) + pd.to_timedelta(df_qc.index, unit='s')
        
        mean_qc = df_qc.loc[df_qc['device_worn'], 'passed_finalQC'].mean()
        if mean_qc < 0.9:
            print(f"Warning: Low data quality. Only {mean_qc:.1%} of ECG passed final QC.")
            df_f = df_qc[(~df_qc['passed_finalQC']) & (df_qc['device_worn'])]
            df_f = df_f.sample(n=min(25, len(df_f)))
            with pyedflib.EdfReader(edf_file) as f:
                plot_save_path = os.path.join(plots_path, base_filename + '_ECGs_failedQC.pdf')
                plotECG_failedQC(f, df_f, plot_save_path)
            
        Ts.append(['proc_ecg',time.time()])
        df_acc, dat_info_acc, _ = readACC(edf_file, start_time)
        
        dat_info = pd.concat([dat_info, dat_info_acc], axis=1)
        df_qc = df_qc.join(df_acc, how='left')
        df_qc.loc[~df_qc['device_worn'], 'acc'] = np.nan
        Ts.append(['get_ACCmetrics',time.time()])

        high_quality_rrm = df_qc.loc[df_qc['passed_finalQC'], 'RRm']
        if not high_quality_rrm.empty:
            dat_info['HR_min'] = 60 * 1000 / high_quality_rrm.max()
            dat_info['HR_max'] = 60 * 1000 / high_quality_rrm.min()
        else:
            dat_info['HR_min'] = np.nan
            dat_info['HR_max'] = np.nan

        df_5min = df_qc.resample('5T', on='time').agg({
            'acc': 'mean', 'rmssd': 'mean', 'RRm': 'mean',
            'passed_finalQC': lambda x: x.mean() > 0.8
        })
        sedentary_windows = df_5min[(df_5min['acc'] < acc_sedentary_thrs) & (df_5min['passed_finalQC'])]

        if not sedentary_windows.empty:
            sedentary_windows['norm_hrv'] = sedentary_windows.apply(
                lambda row: np.log(row['rmssd']) if row['rmssd'] > 0 else np.nan, axis=1
            )
            daily_hrv_summary = sedentary_windows.resample('D').agg({
                'rmssd': 'median', 'norm_hrv': 'median'
            }).dropna()

            for i, (idx, row) in enumerate(daily_hrv_summary.iterrows()):
                day_num = i + 1
                dat_info[f'RMSSD_Day_{day_num}'] = row['rmssd']
                dat_info[f'NormHRV_Day_{day_num}'] = row['norm_hrv']
        
        Ts.append(['calc_raw_metrics', time.time()])

        df_qc = doImp(df_qc, 'RRm')
        df_qc = doImp(df_qc, 'acc')
        Ts.append(['do_imputation',time.time()])

        df_qc['HRm_imputed'] = 60 * 1000 / df_qc['RRm_imputed']
        dat_info['HR_mean'] = df_qc['HRm_imputed'].mean()
        
        sedentary_hr = df_qc.loc[df_qc['acc_imputed'] < acc_sedentary_thrs, 'HRm_imputed']
        if not sedentary_hr.empty:
            dat_info['HR_rest_robust'] = sedentary_hr.rolling(window=180, min_periods=60).mean().min()
        else:
            dat_info['HR_rest_robust'] = np.nan
        
        total_wear_seconds = df_qc['device_worn'].sum() * 10
        if total_wear_seconds > 0:
            acc_series = df_qc.loc[df_qc['device_worn'], 'acc_imputed']
            
            sedentary_seconds = (acc_series < ACTIVITY_THRESHOLDS['light']).sum() * 10
            light_seconds = ((acc_series >= ACTIVITY_THRESHOLDS['light']) & (acc_series < ACTIVITY_THRESHOLDS['moderate'])).sum() * 10
            moderate_seconds = ((acc_series >= ACTIVITY_THRESHOLDS['moderate']) & (acc_series < ACTIVITY_THRESHOLDS['vigorous'])).sum() * 10
            vigorous_seconds = (acc_series >= ACTIVITY_THRESHOLDS['vigorous']).sum() * 10
            
            dat_info['hours_sedentary'] = sedentary_seconds / 3600
            dat_info['hours_light_activity'] = light_seconds / 3600
            dat_info['hours_moderate_activity'] = moderate_seconds / 3600
            dat_info['hours_vigorous_activity'] = vigorous_seconds / 3600
        else:
            dat_info['hours_sedentary'] = 0
            dat_info['hours_light_activity'] = 0
            dat_info['hours_moderate_activity'] = 0
            dat_info['hours_vigorous_activity'] = 0
        
        # --- FIX: Re-added the missing block for final summary statistics ---
        dat_info['wear_time_ECG_10s'] = df_qc["device_worn"].mean() 
        dat_info['ECG_passed_initialQC'] = df_qc["passed_initialQC"].mean() 
        dat_info['prop_ECG_passed_finalQC'] = df_qc['passed_finalQC'].mean()
        dat_info['prop_acc_passed_QC'] = 1-df_qc['acc_clipped'].mean()
        dat_info['meanRR_raw'] = df_qc.loc[df_qc['passed_finalQC'], 'RRm_raw'].mean()
        dat_info['meanACC_raw'] = df_qc.loc[df_qc['device_worn'] & (df_qc['acc_clipped']<0.1), 'acc_raw'].mean()
        
        # Imputation stats
        dat_info['meanRR_imp'] = df_qc['RRm_imputed'].mean()
        dat_info['meanACC_imp'] = df_qc['acc_imputed'].mean()
        dat_info['frac_RR_imp'] = df_qc['RRm_isImputed'].mean() # The crucial missing column
        dat_info['frac_acc_imp'] = df_qc['acc_isImputed'].mean()
        
        df_qc.to_csv(os.path.join(data_path, base_filename + "_df_qc.csv.gz"), compression='gzip')
        plotFunc(df_qc=df_qc, outpath=plots_path, edf_file=edf_file, mrk_hr='HRm_imputed', mrk_acc='acc_imputed')
        
        c1 = (dat_info.loc[0,'N_ecg']/fs/3600/24 > 3)
        c2 = (len(df_qc.groupby(df_qc['time'].dt.hour)['RRm_imputed'].mean()) == 24)

        profile_plot_path = None
        if c1 & c2:
            rrm_median = df_qc.groupby([df_qc['time'].dt.hour, df_qc['time'].dt.minute])['RRm_imputed'].median()
            df_24hr = pd.DataFrame({'RRm_median': rrm_median})
            profile_plot_path = plot_24hr_profile_for_report(
                df_24hr,
                save_path=os.path.join(plots_path, base_filename + "_24hr_profile.png")
            )
        
        Ts.append(['export_results',time.time()])
        
        daily_bars_path = plot_daily_activity_bars(
            df_qc.copy(),
            thresholds=ACTIVITY_THRESHOLDS,
            save_path=os.path.join(plots_path, base_filename + "_daily_bars.png")
        )
        
        pie_chart_path = plot_activity_pie_chart(
            dat_info,
            save_path=os.path.join(plots_path, base_filename + "_activity_pie.png")
        )
        
        hr_dist_path = plot_hr_distribution(
            df_qc,
            save_path=os.path.join(plots_path, base_filename + "_hr_distribution.png")
        )

        create_pdf_report(dat_info, subject_output_path, edf_file, ACTIVITY_THRESHOLDS, daily_bars_path, profile_plot_path, daily_hrv_summary, pie_chart_path, hr_dist_path)

        Ts.append(['create_report',time.time()])
        df = pd.DataFrame(Ts, columns=['task','t'])
        df['dt'] = df['t'] - df.loc[0, 't']
        print(df[['task','dt']])

    except Exception as e:
        print(f"[ERROR] procEDF failed in {edf_file}: {e}")
        import traceback
        traceback.print_exc()
        dat_info['failed'] = 1

    return dat_info


def procEDF_wrapper(edf_filename):
    """A simple wrapper for use with multiprocessing.Pool."""
    return procEDF(edf_filename, m_qrs)


if __name__ == "__main__":
    import dxpy
    @dxpy.entry_point('main')
    
    def main(input_names, i_job):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"

        input_names = [dxpy.DXFile(item) for item in input_names]
        edf_filenames = []
        df_info = []
        
        n_processes = os.cpu_count()
        print(f"N cpu: {n_processes}")
        
        for f in input_names:
            edf_filename = f.describe()["name"]
            dxpy.download_dxfile(f.get_id(), edf_filename)
            edf_filenames.append(edf_filename)
        print("Downloading complete...")

        with mp.Pool(processes=n_processes, initializer=init_worker) as pool:
            df_info = pool.map(procEDF_wrapper, edf_filenames)

        df_info = pd.concat(df_info)

        print(df_info.head())
        outputfile = f"df_info{i_job}.csv.gz"
        df_info.to_csv(outputfile, compression='gzip')

        output_files = glob.glob("*.zip")
        if os.path.exists(outputfile):
            output_files.append(outputfile)

        uploaded_files = []
        project_destination = "project-GVj7k4jJ7YkPP225Y8pB7Q7g"
        folder_destination = "/out"
        for file in output_files:
            uploaded_file = dxpy.upload_local_file(file, project=project_destination, folder=folder_destination)
            uploaded_files.append(uploaded_file)

        output = {}
        output["output_files"] = [dxpy.dxlink(item) for item in uploaded_files]

        return output

    dxpy.run()
