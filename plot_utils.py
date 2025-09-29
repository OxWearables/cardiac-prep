"""
Utility functions for plotting and generating PDF reports from processed ECG and ACC data.

This module contains functions to:
- Generate various plots (time series, histograms, heatmaps, pie charts).
- Assemble the generated plots and summary data into a final PDF report.
"""
__author__ = "Awa Bator"
__credits__ = "Stefan van Duijvenboden"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from pdf2image import convert_from_path
import matplotlib.dates as mdates

# A centralised, cohesive colour palette for all plots
PRIMARY_BLUE = '#31748F'      # Slate Blue for HR plots
SEDENTARY = '#A9D6E5'        # Light Sky Blue
LIGHT_GREEN = '#558B6E'      # Muted Sage Green
MODERATE_YELLOW = '#E9C46A'    # Saffron Yellow
VIGOROUS_RED = '#BC4749'      # Muted Brick Red
NEUTRAL_GRAY = '#808080'      # Neutral Gray

def plotECG_failedQC(f, df_f, plotpath, fs=250, chunk_samples=2500, signal_label="ECG"):
    """
    Plots a grid of 25 ECG segments that failed quality control.
    """
    df_f = df_f.sort_index().head(25)
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 6), dpi=72, 
                             gridspec_kw=dict(hspace=0.4, wspace=0.4))
    axes = axes.flatten()
    iECG = f.getSignalLabels().index(signal_label)

    for i, idx in enumerate(df_f.index):
        ecg_f = f.readSignal(iECG, start=int(idx*fs), n=chunk_samples)
        axes[i].plot(ecg_f, color=PRIMARY_BLUE)
        axes[i].grid(True, which='both', linestyle=':', linewidth=0.4)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_title(df_f['time'].loc[idx].strftime('%d-%m %H:%M:%S'), fontsize=8)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Examples of ECG Segments Failing Quality Control", fontsize=12)
    fig.savefig(plotpath)
    plt.close(fig)

def plotFunc(df_qc,edf_file,outpath, days_per_row = 3, mrk_hr='HRm',mrk_acc='acc'):
    # create plot plotting x no of days per row (heart rate and accelerometer)
    groups = df_qc.groupby(df_qc.index // int(days_per_row*3600*24))[[mrk_hr,mrk_acc,'time','passed_finalQC']]
    
    n = len(groups)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2.5 * n),dpi=72,rasterized=True)
    # fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 1.9 * n),dpi=300,rasterized=True)
    
    # Make sure axes is always iterable
    if n == 1:
        axes = [axes]
    
    for ax, (name, group) in zip(axes, groups):
        group['time'] = group['time'].dt.strftime('%d/%m %H:%M')
    
        # Separate DataFrames for HRm and acc
        df_hr = group[['time', mrk_hr]].set_index('time')
        df_acc = group[['time', mrk_acc]].set_index('time')
        df_qc_ = group[['time', 'passed_finalQC']].set_index('time')
    
        # Create second y-axis, accelerometer
        ax2 = ax.twinx()
        df_acc.plot(ax=ax2, color='tab:orange', label=mrk_acc, alpha=.5)
        ax2.set_ylabel('Acc', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
    
        # Create first y-axis, heart rate
        df_hr.plot(ax=ax, color='tab:blue', label=mrk_hr)
    
    
        # create QC bar on top
        ylim = ax.get_ylim()
        df_qc_['passed'] = ylim[1]+10
        df_qc_['failed'] = ylim[1]+10
        df_qc_.loc[~df_qc_['passed_finalQC'], 'passed'] = np.nan
        df_qc_.loc[df_qc_['passed_finalQC'], 'failed'] = np.nan
        mQC = df_qc_['passed_finalQC'].mean()
        df_qc_.drop(columns=['passed_finalQC'],inplace=True)
        df_qc_.plot(ax=ax, color=['green', 'red'],linewidth=2)
        
        xticks = np.linspace(0, len(df_hr.index) - 1, 6, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(df_hr.index[xticks])
        ax.set_xlabel('')
        ax.set_ylabel(f'HRm ({np.round(100 * mQC, 1)}%)', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
    
        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)
        ax.grid()
    
        fig.suptitle(os.path.basename(edf_file), fontsize=8)
        plt.tight_layout() #rect=[0, 0, 1, 0.96])  # leave space for title

         # Save the entire figure AFTER the loop is complete

        fig.savefig(os.path.join(outpath, os.path.basename(edf_file) + '_HRm_Acc_Plot.pdf'))
        plt.close(fig)


def plot_hr_distribution(df, save_path):
    """Generates a histogram for the overall heart rate distribution."""
    if 'RRm_raw' not in df.columns or not df['passed_finalQC'].any():
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    hr_data = 60 * 1000 / df.loc[df['passed_finalQC'], 'RRm_raw'].dropna()
    if hr_data.empty: plt.close(fig); return None
    hr_upper_limit = hr_data.quantile(0.995)
    sns.histplot(hr_data[hr_data <= hr_upper_limit], kde=True, ax=ax, bins=50, color=PRIMARY_BLUE)
    ax.set_title('Heart Rate Distribution (from High-Quality Data)', fontsize=14)
    ax.set_xlabel('Heart Rate (BPM)'); ax.set_ylabel('Count (10s Segments)')
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    return save_path

def plot_activity_pie_chart(dat_info, save_path):
    """Generates a donut chart of the average daily activity distribution."""
    labels = ['Sedentary', 'Light', 'Moderate', 'Vigorous']
    sizes = [
        dat_info['hours_sedentary'].iloc[0], dat_info['hours_light_activity'].iloc[0],
        dat_info['hours_moderate_activity'].iloc[0], dat_info['hours_vigorous_activity'].iloc[0]
    ]
    colours = [SEDENTARY, LIGHT_GREEN, MODERATE_YELLOW, VIGOROUS_RED]
    non_zero_elements = [(s, l, c) for s, l, c in zip(sizes, labels, colours) if s > 0.01]
    if not non_zero_elements: return None
    sizes, labels, colours = zip(*non_zero_elements)
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, _, _ = ax.pie(sizes, colors=colours, autopct=lambda p: f'{p:.1f}%' if p > 3 else '', startangle=90)
    ax.add_artist(plt.Circle((0, 0), 0.70, fc='white'))
    ax.axis('equal'); ax.set_title("Average Daily Activity", y=1.08, fontsize=16)
    ax.legend(wedges, labels, title="Activity Zone", loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))
    plt.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return save_path

def plot_daily_activity_bars(df, thresholds, save_path):
    """Generates a stacked bar chart of time in activity zones for each day."""
    labels = ['Sedentary', 'Light', 'Moderate', 'Vigorous']
    bins = [-np.inf, thresholds['light'], thresholds['moderate'], thresholds['vigorous'], np.inf]
    df['activity_zone'] = pd.cut(df['acc_imputed'], bins=bins, labels=labels, right=False)
    daily_counts = df.groupby([df['time'].dt.date, 'activity_zone']).size().unstack(fill_value=0)
    daily_hours = daily_counts * 10 / 3600
    daily_hours.index = [d.strftime('%b %d') for d in daily_hours.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_hours.plot(kind='bar', stacked=True, ax=ax, color=[SEDENTARY, LIGHT_GREEN, MODERATE_YELLOW, VIGOROUS_RED], width=0.8)
    ax.set_title("Time in Activity Zones per Day", fontsize=16); ax.set_ylabel("Hours"); ax.set_xlabel("Date")
    ax.legend(title="Activity Zone"); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    return save_path

def plot_24hr_profile_for_report(df_24hr, save_path):
    """Generates a smoothed 24-hour heart rate profile plot."""
    if df_24hr is None or df_24hr.empty: return None
    df_24hr['HRm_median'] = 60 * 1000 / df_24hr['RRm_median']
    df_24hr['HRm_smoothed'] = df_24hr['HRm_median'].rolling(window=30, center=True, min_periods=1).median()
    df_24hr['time_str'] = df_24hr['hour'].astype(str).str.zfill(2) + ':' + df_24hr['minute'].astype(str).str.zfill(2)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_24hr['time_str'], df_24hr['HRm_smoothed'], color=PRIMARY_BLUE)
    ax.set_xticks(df_24hr['time_str'][::120]); ax.tick_params(axis='x', rotation=45)
    ax.set_title('Typical 24-Hour Heart Rate Profile', fontsize=16)
    ax.set_xlabel('Time of Day'); ax.set_ylabel('Median Heart Rate (BPM)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    return save_path

def create_pdf_report(dat_info, subject_output_path, edf_file, thresholds, num_days, daily_bars_path, profile_plot_path, daily_hrv_summary, pie_chart_path, hr_dist_path):
    """Generates the final summary PDF report, assembling all plots and metrics."""
    base_filename = os.path.basename(edf_file)
    pdf_path = os.path.join(subject_output_path, "plots", base_filename + "_Summary_Report.pdf")
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()

    # --- PAGE 1: Overall Summary ---
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2.0, height - 0.75*inch, "Your Activity & Heart Report")
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2.0, height - 1.05*inch, "Summary Metrics")
    
    # --- Main Plots ---
    if profile_plot_path and os.path.exists(profile_plot_path):
        c.drawImage(profile_plot_path, 0.5*inch, height - 4.25*inch, width=width-1*inch, height=3*inch, preserveAspectRatio=True)
    if pie_chart_path and os.path.exists(pie_chart_path):
        c.drawImage(pie_chart_path, 0.7*inch, height - 6.5*inch, width=3*inch, height=2.2*inch, preserveAspectRatio=True)
    
    # Heart Rate Table
    hr_data = [
        ['Resting', f"{dat_info['HR_rest_robust'].iloc[0]:.1f}"], 
        ['Lowest', f"{dat_info['HR_min'].iloc[0]:.1f}"],
        ['Average', f"{dat_info['HR_mean'].iloc[0]:.1f}"], 
        ['Highest', f"{dat_info['HR_max'].iloc[0]:.1f}"]
    ]
    hr_table = Table(hr_data, colWidths=[1.2*inch]*2, rowHeights=0.4*inch)
    hr_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 1, colors.black),
                                  ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold')]))
    hr_table.wrapOn(c, 2.4*inch, 2*inch)
    hr_table.drawOn(c, 4.75*inch, height - 6.0*inch)

    if hr_dist_path and os.path.exists(hr_dist_path):
        c.drawImage(hr_dist_path, 0.5*inch, height - 9.5*inch, width=width-1*inch, height=2.8*inch, preserveAspectRatio=True)
    
    # --- PAGE 2: Day-by-Day Details ---
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2.0, height - 0.75*inch, "Daily Results")
    
    y_cursor = height - 1.25*inch
    if daily_hrv_summary is not None and not daily_hrv_summary.empty:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y_cursor, "Heart Rate Variability")
        y_cursor -= 0.15*inch

        p_style = styles['Italic']
        p_style.fontSize = 9
        explanation_text = """This table shows your median Heart Rate Variability (RMSSD) calculated 
        during sleep. We also show the natural log of RMSSD in brackets, which 'normalises' the value. 
        This is useful because factors like alcohol or stress can raise your heart rate, 
        which in turn lowers your HRV. The normalised value helps to reduce this effect, 
        giving a clearer picture of your nervous system's recovery."""
        p = Paragraph(explanation_text, p_style)
        p.wrapOn(c, width - 2*inch, 1*inch)
        p.drawOn(c, 1*inch, y_cursor - p.height)
        y_cursor -= (p.height + 0.2*inch)

        dates = [d.strftime('%b %d') for d in daily_hrv_summary.index]
        num_nights = len(dates) - 1
        headers = [''] + [f"Night {i+1}" for i in range(num_nights)]
        # Correctly create date pairs for n-1 nights, with "Jul 10/11" format
        date_row = ['Date'] + [f"{dates[i]}/{dates[i+1].split()[-1]}" for i in range(num_nights)]
        # Extract ALL formatted values first
        formatted_values = daily_hrv_summary.apply(
            lambda row: f"{row['rmssd']:.1f} ({row['norm_hrv']:.2f})" if pd.notna(row['rmssd']) else "No Data", 
            axis=1
        ).tolist()
        # Correctly slice data for n-1 nights.
        # We use formatted_values[1:] because the HRV for the night of "Day 1 / Day 2" is typically
        # calculated from the sleep data recorded on the morning of "Day 2".
        data_row = ['RMSSD (Norm)'] + formatted_values[1:]
        table_data = [headers, date_row, data_row]
      
        # The number of data columns is now num_nights
        t = Table(table_data, colWidths=[1.2*inch] + [1.0*inch] * num_nights)
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                               ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold')]))
        
        t.wrapOn(c, width - 2*inch, height)
        t.drawOn(c, (width - t._width) / 2, y_cursor - t._height)
        y_cursor -= (t._height + 0.5 * inch)

    if daily_bars_path and os.path.exists(daily_bars_path):
        c.drawImage(daily_bars_path, 0.5*inch, y_cursor - 4.5*inch, width=width-1*inch, height=4*inch, preserveAspectRatio=True)


    c.save()
    print(f"Generated PDF report, saved to: {pdf_path}")