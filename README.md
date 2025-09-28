```markdown
# ECG & Accelerometry (EDF) Preprocessing Pipeline

This repository contains an end-to-end Python pipeline for processing multi-day wearable sensor data from `.EDF` files. It extracts key physiological metrics related to heart rate, heart rate variability (HRV), and physical activity, performs quality control and data imputation, and generates detailed summary reports and visualisations.

## Features ✨
* **Parallel Processing**: Efficiently processes large batches of files using all available **CPU** cores (no need to run with GPU).
* **Preprocessing**: Implements QRS detection, artifact removal, and signal quality assessment.
* **Data Imputation**: Fills gaps in summary data (e.g. median R-R interval) by using linear interpolation for short gaps and a time-of-day averaging for longer gaps.
* **Comprehensive Metrics**: Calculates a wide range of metrics, including:
    * Heart Rate (Resting, Min, Max, Average)
    * Heart Rate Variability (RMSSD and log-normalised RMSSD to account for HR)
    * Physical Activity (Time spent in Sedentary, Light, Moderate, and Vigorous zones)
* **Automated Reporting**: Generates a detailed, 2-page PDF summary report for each participant.
* **Dataset Summarisation**: Creates summary plots for the entire dataset to visualise population-level trends.

## How to Use 🚀

### 1. Clone the Repository
First, clone this repository to your local machine using git.
```bash
# TODO UPDATE WHEN TRANSFERRED TO OxWearables
git clone https://github.com/annabator/preprocessing-edf.git
cd your-repository-folder
```

### 2. Installation
Install all the necessary Python dependencies from the `requirements.txt` file. It is highly recommended to do this within a virtual environment.
```bash
pip install -r requirements.txt
```

### 3.  **Add Data**: Place your raw `.EDF` files in the `./input_data/` directory.

### 4.  **Run Pipeline**: Execute the scripts in order from your terminal.
    ```bash
    # Step 1: Process Individual Files
    # This command processes every .edf file and generates a detailed output folder for each participant inside ./output/.
    # Average time per file: ~30 seconds
    python run_local.py

    # Step 2: Generate Dataset-Level Summary Plots
    # This command reads the aggregated summary file created in the first step and generates plots for your entire dataset.
    python generate_dataset_summary.py

    # Step 3 (Optional): Generate Additional Patient-Level Heatmaps
    # HR and accelerometry heatmap - runs one participant at a time,
    # not integrated with the rest of the pipeline - just leaving
    # the code here in case it is useful in the future.
    python visualise_results.py
    ```

## Outputs
The pipeline generates an `./output/` directory containing:
* **Participant Folders**: Each contains a `plots` directory with a PDF summary report and a `processed_data` directory with detailed CSVs.
* **Top-Level Files**: An aggregated `df_info_summary.csv.gz` for all participants and summary plots (`.png`) for the entire dataset.

## A Note on HRV Normalisation
This pipeline uses the natural logarithm of RMSSD ($ln(RMSSD)$) to calculate a normalised HRV value. This is a standard statistical method that removes the mathematical influence of a person's average heart rate, allowing for fairer comparisons. For a detailed explanation, see this article by Marco Altini:
* [**Should we normalize HRV by heart rate?**](https://marcoaltini.substack.com/p/should-we-normalize-hrv-by-heart)

## Authors
* **Stefan van Duijvenboden**
* **Awa Bator**
