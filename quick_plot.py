"""
A simple utility script for quick, manual visualisation of a single participant's data.

This script loads the processed time-series, daily, and 24-hour profile data
for a specified subject and generates several plots for immediate inspection.
It is intended for ad-hoc analysis and debugging.
"""
__author__ = "Awa Bator"

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# TODO CHANGE THIS to the base name of the subject you want to plot (e.g., "subject_01")
subject_id = "REDACTED_ID" 
base_output_path = "./output/"

def load_data(subject_path):
    """Loads the three main csv.gz files for a given subject."""
    
    qc_path = glob.glob(os.path.join(subject_path, "processed_data", "*_df_qc.csv.gz"))
    daily_path = glob.glob(os.path.join(subject_path, "processed_data", "*_df_14d.csv.gz"))
    profile_path = glob.glob(os.path.join(subject_path, "processed_data", "*_df_24hr.csv.gz"))
    
    df_qc = pd.read_csv(qc_path[0], parse_dates=['time'], index_col='time') if qc_path else None
    df_daily = pd.read_csv(daily_path[0]) if daily_path else None
    
    df_24hr = None
    if profile_path:
        df_24hr = pd.read_csv(profile_path[0])
        # The first two columns from the CSV are the hour and minute.
        # We rename them here for clarity and robust access.
        df_24hr.rename(columns={
            df_24hr.columns[0]: 'hour', 
            df_24hr.columns[1]: 'minute'
        }, inplace=True)
        
    return df_qc, df_daily, df_24hr


def plot_detailed_timeseries(df_qc, subject_id):
    """Plots the detailed 10-second HR and Activity data."""
    if df_qc is None:
        print(f"Skipping detailed plot for {subject_id}: _df_qc.csv.gz not found.")
        return
        
    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'Detailed 10-Second Data for {subject_id}', fontsize=16)
    
    ax1.plot(df_qc.index, df_qc['HRm_imputed'], color='tab:blue', label='Heart Rate')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Heart Rate (BPM)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(df_qc.index, df_qc['acc_imputed'], color='tab:orange', alpha=0.7, label='Activity')
    ax2.set_ylabel('Acceleration (milli-g)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_daily_summary(df_daily, subject_id):
    """Plots the average daily heart rate."""
    if df_daily is None:
        print(f"Skipping daily summary for {subject_id}: _df_14d.csv.gz not found.")
        return

    df_daily['HRm_imputed'] = 60 / df_daily['RRm_imputed']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_daily.index, df_daily['HRm_imputed'], color='steelblue')
    
    ax.set_title(f'Average Daily Heart Rate for {subject_id}')
    ax.set_xlabel('Day Number')
    ax.set_ylabel('Average Heart Rate (BPM)')
    ax.set_xticks(df_daily.index)
    ax.grid(axis='y', linestyle='--')
    
    fig.tight_layout()


def plot_24hr_profile(df_24hr, subject_id):
    """Plots the typical 24-hour heart rate rhythm."""
    if df_24hr is None:
        print(f"Skipping 24hr profile for {subject_id}: _df_24hr.csv.gz not found.")
        return
        
    # Create the time-of-day string using the now-reliable 'hour' and 'minute' columns
    df_24hr['time_str'] = df_24hr['hour'].astype(str).str.zfill(2) + ':' + df_24hr['minute'].astype(str).str.zfill(2)

    df_24hr['HRm_median'] = 60 / df_24hr['RRm_median']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_24hr['time_str'], df_24hr['HRm_median'], color='purple')
    
    ax.set_xticks(df_24hr['time_str'][::120]) # Show a tick every 2 hours
    ax.tick_params(axis='x', rotation=45)
    
    ax.set_title(f'Typical 24-Hour Heart Rate Profile for {subject_id}')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Median Heart Rate (BPM)')
    ax.grid(linestyle='--')
    
    fig.tight_layout()


if __name__ == '__main__':
    subject_path = os.path.join(base_output_path, subject_id)
    
    if not os.path.isdir(subject_path):
        print(f"Error: Output directory not found for subject '{subject_id}' at '{subject_path}'")
    else:
        print(f"--- Loading data for {subject_id} ---")
        df_qc, df_daily, df_24hr = load_data(subject_path)
        
        plot_detailed_timeseries(df_qc, subject_id)
        plot_daily_summary(df_daily, subject_id)
        plot_24hr_profile(df_24hr, subject_id)
        
        print("\nDisplaying plots...")
        plt.show()
