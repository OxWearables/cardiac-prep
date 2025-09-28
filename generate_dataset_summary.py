"""
Generates high-level summary visualisations for the entire processed dataset.

This script reads the aggregated summary CSV file produced by `run_local.py`
and creates two summary plots:
1. A grid of histograms showing the distribution of key metrics across all participants.
2. A scatter plot showing the relationship between resting heart rate and HRV.
"""
__author__ = "Awa Bator"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from plot_utils import SEDENTARY, LIGHT_GREEN, MODERATE_YELLOW, VIGOROUS_RED, NEUTRAL_GRAY
from matplotlib.ticker import MaxNLocator

SUMMARY_FILE_PATH = "./output/df_info_summary.csv.gz"
OUTPUT_PLOT_PATH = "./output/dataset_summary_histograms.png"

def plot_hrv_vs_rhr(df, save_path):
    """
    Generates a scatter plot of Resting HR vs. HRV (RMSSD) with a regression line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.regplot(
        x='HR_rest_robust', 
        y='mean_daily_rmssd', 
        data=df, 
        ax=ax, 
        scatter_kws={'alpha': 0.6, 'color': SEDENTARY},
        line_kws={'color': VIGOROUS_RED}
    )
    
    ax.set_title('Resting Heart Rate vs. Heart Rate Variability (RMSSD)', fontsize=16)
    ax.set_xlabel('Robust Resting HR (BPM)', fontsize=12)
    ax.set_ylabel('Mean Daily RMSSD (ms)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Dataset RHR vs. HRV plot saved to: {save_path}")
    plt.close(fig)


def plot_dataset_histograms(df):
    """
    Generates a 3x2 grid of histograms for key dataset metrics.
    """
    if df.empty:
        print("The summary DataFrame is empty. No plots will be generated.")
        return

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Summary of Processed Dataset Metrics', fontsize=18, y=1.02)

    ## Plotting Health Metrics
    # Robust Resting Heart Rate
    sns.histplot(df['HR_rest_robust'].dropna(), kde=True, ax=axes[0, 0], color=SEDENTARY)
    axes[0, 0].set_title('Distribution of Resting Heart Rate', fontsize=14)
    axes[0, 0].set_xlabel('Robust Resting HR (BPM)')
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Mean Daily RMSSD (HRV)
    sns.histplot(df['mean_daily_rmssd'].dropna(), kde=True, ax=axes[0, 1], color=LIGHT_GREEN)
    axes[0, 1].set_title('Distribution of Heart Rate Variability (RMSSD)', fontsize=14)
    axes[0, 1].set_xlabel('Mean Daily RMSSD (ms)')
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Hours of MVPA (Moderate-to-Vigorous Physical Activity)
    if 'hours_mvpa' not in df.columns:
        df['hours_mvpa'] = df['hours_moderate_activity'] + df['hours_vigorous_activity']
    
    data_to_plot = df['hours_mvpa'].dropna()
    upper_limit = data_to_plot.quantile(0.995)
    sns.histplot(data_to_plot[data_to_plot <= upper_limit], kde=True, ax=axes[1, 0], color=MODERATE_YELLOW)
    axes[1, 0].set_title('Distribution of Daily MVPA', fontsize=14)
    axes[1, 0].set_xlabel('Average Hours per Day')
    
    # Hours of Vigorous Activity
    sns.histplot(df['hours_vigorous_activity'].dropna(), kde=True, ax=axes[1, 1], color=VIGOROUS_RED)
    axes[1, 1].set_title('Distribution of Daily Vigorous Activity', fontsize=14)
    axes[1, 1].set_xlabel('Average Hours per Day')

    ## Plotting Data Quality Metrics
    #  Proportion of Good Quality ECG
    sns.histplot(df['prop_ECG_passed_finalQC'].dropna(), kde=True, ax=axes[2, 0], color=NEUTRAL_GRAY)
    axes[2, 0].set_title('Distribution of Usable ECG Data', fontsize=14)
    axes[2, 0].set_xlabel('Proportion of High-Quality ECG Segments')
    axes[2, 0].set_xlim(0, 1)

    # Fraction of Imputed Data
    sns.histplot(df['frac_RR_imp'].dropna(), kde=True, ax=axes[2, 1], color=NEUTRAL_GRAY)
    axes[2, 1].set_title('Distribution of Imputed Data', fontsize=14)
    axes[2, 1].set_xlabel('Fraction of Heart Rate Data Imputed')
    axes[2, 1].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"Dataset summary plot saved to: {OUTPUT_PLOT_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    if not os.path.exists(SUMMARY_FILE_PATH):
        print(f"Error: Summary file not found at '{SUMMARY_FILE_PATH}'")
        print("Please ensure you have run 'run_local.py' to process your dataset first.")
    else:
        summary_df = pd.read_csv(SUMMARY_FILE_PATH)
        
        # Find all columns that match the 'RMSSD_Day_X' pattern
        rmssd_day_cols = [col for col in summary_df.columns if 'RMSSD_Day_' in col]
        if rmssd_day_cols:
            # Calculate the mean across these columns for each participant
            summary_df['mean_daily_rmssd'] = summary_df[rmssd_day_cols].mean(axis=1)
        else:
            # Fallback if no daily columns are found (for older summary files)
            print("Warning: No 'RMSSD_Day_X' columns found. Falling back to 'mean_rmssd_ms_overall'.")
            summary_df['mean_daily_rmssd'] = summary_df['mean_rmssd_ms_overall']
        
        # Generate histogram plots
        plot_dataset_histograms(summary_df.copy())

        # Generate RHR vs HRV plot
        plot_hrv_vs_rhr(
            summary_df.copy(), 
            save_path="./output/dataset_rhr_vs_hrv.png"
        )
