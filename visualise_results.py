__author__ = "Awa Bator"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# TODO CHANGE THIS to the base name of the subject you want to plot 
subject_id = "001" 
base_output_path = "./output/"

def load_detailed_data(subject_path):
    """Loads the detailed _df_qc.csv.gz file for a given subject."""
    qc_file_path = glob.glob(os.path.join(subject_path, "processed_data", "*_df_qc.csv.gz"))
    if not qc_file_path:
        return None
    return pd.read_csv(qc_file_path[0], parse_dates=['time'])


def plot_weekly_heatmap(df, column, title, cbar_label, save_path):
    """Generates a heatmap of a given metric by day of week and hour."""
    df['day_of_week'] = df['time'].dt.day_name()
    df['hour_of_day'] = df['time'].dt.hour
    
    pivot_df = df.pivot_table(
        index='day_of_week', 
        columns='hour_of_day', 
        values=column, 
        aggfunc='mean'
    )
    
    # Ensure days are in the correct order
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_df = pivot_df.reindex(days_order)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot_df, cmap='viridis', ax=ax, cbar_kws={'label': cbar_label})
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Generated heatmap: {os.path.basename(save_path)}")


if __name__ == '__main__':
    subject_path = os.path.join(base_output_path, subject_id)
    
    if not os.path.isdir(subject_path):
        print(f"Error: Output directory not found for subject '{subject_id}' at '{subject_path}'")
    else:
        df_qc = load_detailed_data(subject_path)
        
        if df_qc is not None:
            # Create a folder for these new plots if it doesn't exist
            visualization_path = os.path.join(subject_path, "visualizations")
            os.makedirs(visualization_path, exist_ok=True)
            
            print(f"\n--- Generating advanced plots for {subject_id} ---")
            
            plot_weekly_heatmap(
                df_qc.copy(), # Use a copy to avoid modifying the original df
                column='HRm_imputed', 
                title=f'Average Heart Rate by Hour and Day ({subject_id})',
                cbar_label='Heart Rate (BPM)',
                save_path=os.path.join(visualization_path, f"{subject_id}_hr_heatmap.png")
            )
            
            plot_weekly_heatmap(
                df_qc.copy(),
                column='acc_imputed',
                title=f'Average Activity by Hour and Day ({subject_id})',
                cbar_label='Acceleration (milli-g)',
                save_path=os.path.join(visualization_path, f"{subject_id}_acc_heatmap.png")
            )
            
            thresholds = {'sedentary': 25, 'light': 100, 'moderate': 200}
            plot_daily_activity_bars(
                df_qc.copy(),
                thresholds=thresholds,
                save_path=os.path.join(visualization_path, f"{subject_id}_daily_activity_bars.png")
            )
        else:
            print(f"Could not load detailed data for {subject_id}. No plots generated.")