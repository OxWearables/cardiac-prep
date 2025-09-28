"""
Utility functions for plotting and generating PDF reports from processed ECG and ACC data.

This module contains functions to:
- Generate various plots (time series, histograms, heatmaps, pie charts).
- Assemble the generated plots and summary data into a final PDF report.
"""
__author__ = "Awa Bator"
__credits__ = "Stefan van Duijvenboden"
## This script is based on the original work by Stefan van Duijvenboden.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- ReportLab Imports for PDF Generation ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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

    Args:
        f (pyedflib.EdfReader): An open EDF reader object.
        df_f (pd.DataFrame): DataFrame containing the indices and times of failed segments.
        plotpath (str): The file path to save the output PDF plot.
        fs (int): Sampling frequency of the ECG signal.
        chunk_samples (int): Number of samples per segment to plot.
        signal_label (str): The label of the ECG signal in the EDF file.
    """
    df_f = df_f.sort_index()
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 6), dpi=72, 
                             gridspec_kw=dict(hspace=0, wspace=0))
    axes = axes.flatten()
    iECG = f.getSignalLabels().index(signal_label)

    for i, idx in enumerate(df_f.index):
        ecg_f = f.readSignal(iECG, start=int(idx*fs), n=chunk_samples)
        axes[i].plot(ecg_f)
        axes[i].grid(True, which='both', linestyle=':', linewidth=0.4)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xticklabels([])
        axes[i].set_title(df_f['time'].loc[idx].strftime('%d-%m %H:%M:%S'), fontsize=8)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("ECGs failed QC: " + os.path.basename(plotpath), fontsize=10)
    fig.savefig(plotpath)
    plt.close()


def plotFunc(df_qc, edf_file, outpath, days_per_row=3, mrk_hr='HRm', mrk_acc='acc'):
    """
    Creates multi-day overview plots of heart rate and activity.

    Args:
        df_qc (pd.DataFrame): The main processed DataFrame.
        edf_file (str): Original EDF filename, used for the title.
        outpath (str): Directory path to save the output plot.
        days_per_row (int): Number of days to display per subplot row.
        mrk_hr (str): Column name for heart rate data.
        mrk_acc (str): Column name for accelerometer data.
    """
    groups = df_qc.groupby(df_qc.index // int(days_per_row*3600*24))[[mrk_hr,mrk_acc,'time','passed_finalQC']]
    
    n = len(groups)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2.5 * n), dpi=72, rasterized=True)
    
    if n == 1:
        axes = [axes]
    
    for ax, (name, group) in zip(axes, groups):
        group['time'] = group['time'].dt.strftime('%d/%m %H:%M')
    
        df_hr = group[['time', mrk_hr]].set_index('time')
        df_acc = group[['time', mrk_acc]].set_index('time')
        df_qc_ = group[['time', 'passed_finalQC']].set_index('time')
    
        ax2 = ax.twinx()
        df_acc.plot(ax=ax2, color='tab:orange', label=mrk_acc, alpha=.5)
        ax2.set_ylabel('Acc', color='tab:orange', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=10)
    
        df_hr.plot(ax=ax, color='tab:blue', label=mrk_hr)
    
        ylim = ax.get_ylim()
        df_qc_['passed'] = ylim[1] + (ylim[1] * 0.05)
        df_qc_['failed'] = ylim[1] + (ylim[1] * 0.05)
        df_qc_.loc[~df_qc_['passed_finalQC'], 'passed'] = np.nan
        df_qc_.loc[df_qc_['passed_finalQC'], 'failed'] = np.nan
        mQC = df_qc_['passed_finalQC'].mean()
        df_qc_.drop(columns=['passed_finalQC'], inplace=True)
        df_qc_.plot(ax=ax, color=['green', 'red'], linewidth=2)
        
        xticks = np.linspace(0, len(df_hr.index) - 1, 6, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(df_hr.index[xticks], fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(f'HRm ({mQC:.1%})', color='tab:blue', fontsize=12)
        ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
    
        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)
        ax.grid()
    
        fig.suptitle(os.path.basename(edf_file), fontsize=12)
        plt.tight_layout()
        
        fig.savefig(os.path.join(outpath, os.path.basename(edf_file) + '_HRm_Acc_Plot.pdf'))
        plt.close()


def plot_hr_distribution(df, save_path):
    """
    Generates and saves a histogram for the overall heart rate distribution.

    Args:
        df (pd.DataFrame): The main processed DataFrame.
        save_path (str): The file path to save the output plot.

    Returns:
        str or None: The path to the saved plot, or None if no plot was generated.
    """
    if 'RRm_raw' not in df.columns or df['passed_finalQC'].sum() == 0:
        return None
        
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    
    hr_data = 60 * 1000 / df.loc[df['passed_finalQC'], 'RRm_raw'].dropna()
    
    if hr_data.empty:
        plt.close(fig)
        return None

    hr_upper_limit = hr_data.quantile(0.99)
    
    sns.histplot(
        hr_data[hr_data <= hr_upper_limit],
        kde=True, ax=ax, bins=50,
        hist_kws={'color': PRIMARY_BLUE, 'alpha': 0.6},
        kde_kws={'color': PRIMARY_BLUE, 'linewidth': 2.5}
    )
    
    ax.set_title('Overall Heart Rate Distribution', fontsize=16)
    ax.set_xlabel('Heart Rate (BPM)', fontsize=12)
    ax.set_ylabel('Count (10s Segments)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_activity_pie_chart(dat_info, save_path):
    """
    Generates a donut chart of the average daily activity distribution.

    Args:
        dat_info (pd.DataFrame): The summary DataFrame containing activity hours.
        save_path (str): The file path to save the output plot.

    Returns:
        str or None: The path to the saved plot, or None if no plot was generated.
    """
    labels = ['Sedentary', 'Light', 'Moderate', 'Vigorous']
    sizes = [
        dat_info['hours_sedentary'].iloc[0],
        dat_info['hours_light_activity'].iloc[0],
        dat_info['hours_moderate_activity'].iloc[0],
        dat_info['hours_vigorous_activity'].iloc[0]
    ]
    colours = [SEDENTARY, LIGHT_GREEN, MODERATE_YELLOW, VIGOROUS_RED]
    
    non_zero_elements = [(size, label, colour) for size, label, colour in zip(sizes, labels, colours) if size > 0.01]
    if not non_zero_elements:
        return None
    
    sizes, labels, colours = zip(*non_zero_elements)

    fig, ax = plt.subplots(figsize=(6, 4))
    
    def autopct_generator(limit):
        def inner_autopct(pct):
            return f'{pct:.1f}%' if pct > limit else ''
        return inner_autopct

    wedges, _, _ = ax.pie(sizes, colors=colours, 
                                      autopct=autopct_generator(3),
                                      startangle=90, 
                                      textprops={'fontsize': 12})
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')
    ax.set_title("Average Daily Activity", y=1.08, fontsize=16)

    ax.legend(wedges, labels,
              title="Activity Zone",
              loc="centre left",
              bbox_to_anchor=(0.9, 0, 0.5, 1),
              fontsize=12,
              title_fontsize=12)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_daily_activity_bars(df, thresholds, save_path):
    """
    Generates a stacked bar chart of time in activity zones for each day.

    Args:
        df (pd.DataFrame): The main processed DataFrame.
        thresholds (dict): Dictionary of activity thresholds.
        save_path (str): The file path to save the output plot.

    Returns:
        str: The path to the saved plot.
    """
    labels = ['Sedentary', 'Light', 'Moderate', 'Vigorous']
    bins = [-np.inf, thresholds['light'], thresholds['moderate'], thresholds['vigorous'], np.inf]
    df['activity_zone'] = pd.cut(df['acc_imputed'], bins=bins, labels=labels, right=False)
    
    daily_counts = df.groupby([df['time'].dt.date, 'activity_zone']).size().unstack(fill_value=0)
    daily_hours = daily_counts * 10 / 3600
    daily_hours.index = [d.strftime('%b %d') for d in daily_hours.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_hours.plot(kind='bar', stacked=True, ax=ax, 
                     color=[SEDENTARY, LIGHT_GREEN, MODERATE_YELLOW, VIGOROUS_RED], 
                     width=0.8, edgecolor='none')
    
    ax.set_title("Time in Activity Zones per Day", fontsize=16)
    ax.set_ylabel("Hours", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(title="Activity Zone", fontsize=11, title_fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_24hr_profile_for_report(df_24hr, save_path):
    """
    Generates and saves a smoothed 24-hour heart rate profile plot.

    Args:
        df_24hr (pd.DataFrame): DataFrame of minute-by-minute median HR data.
        save_path (str): The file path to save the output plot.

    Returns:
        str or None: The path to the saved plot, or None if no plot was generated.
    """
    if df_24hr is None or df_24hr.empty:
        return None
        
    hour = df_24hr.index.get_level_values(0).astype(int).astype(str).str.zfill(2)
    minute = df_24hr.index.get_level_values(1).astype(int).astype(str).str.zfill(2)
    df_24hr['time_str'] = hour + ':' + minute
    
    df_24hr['HRm_median'] = 60 * 1000 / df_24hr['RRm_median']
    df_24hr['HRm_smoothed'] = df_24hr['HRm_median'].rolling(window=30, center=True, min_periods=1).median()
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df_24hr['time_str'], df_24hr['HRm_smoothed'], color=PRIMARY_BLUE)
    
    ax.set_xticks(df_24hr['time_str'][::120])
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    ax.set_title('Typical 24-Hour Heart Rate Profile', fontsize=16)
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Median Heart Rate (BPM)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

def create_pdf_report(dat_info, subject_output_path, edf_file, thresholds, daily_bars_path, profile_plot_path, daily_hrv_summary, pie_chart_path, hr_dist_path):
    """
    Generates the final summary PDF report, assembling all plots and metrics.
    """
    base_filename = os.path.basename(edf_file)
    pdf_path = os.path.join(subject_output_path, "plots", base_filename + "_Summary_Report.pdf")
    
    # --- FIX: Added the necessary PDF and style initialisation ---
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()

    # Page 1: Overall Summary
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2.0, height - 0.75*inch, "Your Activity & Heart Summary")
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2.0, height - 1.0*inch, f"Participant File: {base_filename}")

    y_cursor = height - 1.25*inch
    if profile_plot_path and os.path.exists(profile_plot_path):
        c.drawImage(profile_plot_path, 0.5*inch, y_cursor - 3.0*inch, width=width-1*inch, height=3*inch, preserveAspectRatio=True)
        y_cursor -= 3.25*inch
    else:
        c.setFont("Helvetica-Oblique", 10)
        c.drawCentredString(width/2.0, y_cursor - 1.5*inch, "(24-hour profile plot not generated)")
        y_cursor -= 3.25*inch
    
    pie_height = 2.2 * inch
    if pie_chart_path and os.path.exists(pie_chart_path):
        c.drawImage(pie_chart_path, 0.7*inch, y_cursor - pie_height, width=3*inch, height=pie_height, preserveAspectRatio=True)
    
    y_def_start = y_cursor - 0.8*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(4.5*inch, y_def_start, "How We Define Activity (milli-g)")
    y_def_start -= 0.25*inch
    c.setFont("Helvetica", 11)
    c.drawString(4.7*inch, y_def_start, f"- Sedentary: < {thresholds['light']}")
    y_def_start -= 0.2*inch
    c.drawString(4.7*inch, y_def_start, f"- Light: {thresholds['light']} to {thresholds['moderate'] - 1}")
    y_def_start -= 0.2*inch
    c.drawString(4.7*inch, y_def_start, f"- Moderate: {thresholds['moderate']} to {thresholds['vigorous'] - 1}")
    y_def_start -= 0.2*inch
    c.drawString(4.7*inch, y_def_start, f"- Vigorous: >= {thresholds['vigorous']}")
    
    y_cursor -= (pie_height + 0.25*inch)
    
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2.0, y_cursor, "Average Daily Heart Rate Values (BPM)")
    y_cursor -= 0.25*inch

    hr_data = [
        ['Resting', f"{dat_info['HR_rest_robust'].iloc[0]:.1f}"],
        ['Lowest',  f"{dat_info['HR_min'].iloc[0]:.1f}"],
        ['Average', f"{dat_info['HR_mean'].iloc[0]:.1f}"],
        ['Highest', f"{dat_info['HR_max'].iloc[0]:.1f}"]
    ]
    hr_table = Table(hr_data, colWidths=[1.2*inch, 1.2*inch])
    hr_table.setStyle(TableStyle([
       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
       ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
       ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    hr_table.wrapOn(c, 2.4*inch, 2*inch)
    hr_table.drawOn(c, (width - 2.4*inch)/2, y_cursor - hr_table._height)
    y_cursor -= (hr_table._height + 0.2*inch)

    if hr_dist_path and os.path.exists(hr_dist_path):
        c.drawImage(hr_dist_path, 0.5*inch, y_cursor - 2.5*inch, width=width-1*inch, height=2.2*inch, preserveAspectRatio=True)
    
    # Page 2: Day-by-Day Details
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2.0, height - 0.75*inch, "Day-by-Day Details")
    
    y_cursor = height - 1.25*inch

    if daily_hrv_summary is not None and not daily_hrv_summary.empty:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y_cursor, "Daily Heart Rate Variability")
        y_cursor -= 0.15*inch

        p_style = styles['Italic']
        p_style.fontSize = 10
        explanation_text = """
        HRV (RMSSD) measures beat-to-beat changes in heart rate, reflecting autonomic nervous system activity.
        The normalised value (in brackets) accounts for average heart rate, allowing a fairer comparison across days.
        """
        p = Paragraph(explanation_text, p_style)
        p.wrapOn(c, width - 2*inch, 1*inch)
        p.drawOn(c, 1*inch, y_cursor - p.height)
        y_cursor -= (p.height + 0.15*inch)

        dates = [d.strftime('%b %d') for d in daily_hrv_summary.index]
        headers = [''] + [f"Day {i+1}" for i in range(len(daily_hrv_summary))]
        date_row = ['Date'] + dates
        
        formatted_values = daily_hrv_summary.apply(
            lambda row: f"{row['rmssd']:.0f} ({row['norm_hrv']:.2f})", axis=1
        ).tolist()
        data_row = ['RMSSD (Norm HRV)'] + formatted_values
        
        table_data = [headers, date_row, data_row]
        
        t = Table(table_data, colWidths=[1.5*inch] + [1.0*inch] * len(daily_hrv_summary))
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        t.wrapOn(c, width - 2*inch, height)
        t.drawOn(c, (width - t._width) / 2, y_cursor - t._height)
        y_cursor -= (t._height + 0.5 * inch)

    if daily_bars_path and os.path.exists(daily_bars_path):
        c.drawImage(daily_bars_path, 0.5*inch, y_cursor - 4.5*inch, width=width-1*inch, height=4*inch, preserveAspectRatio=True)

    c.save()
    print(f"Generated PDF report, saved to: {pdf_path}")
