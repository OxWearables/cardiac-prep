# ECG & Accelerometry (EDF) Preprocessing Pipeline

This repository contains an end-to-end Python pipeline for processing multi-day wearable sensor data from `.EDF` files. It extracts key physiological metrics related to heart rate, heart rate variability (HRV), and physical activity, performs quality control and data imputation, and generates detailed summary reports and visualisations.

**You do not need to be a programmer to use this.** Follow the steps below in order, copying each command exactly. Every command is a single line — copy the whole line.

---

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

---

## Before you start 📝

You need **Python 3.9, 3.10, 3.11 or 3.12**. Python 3.13 is not yet supported.

To check what you have, open a terminal and type:

```
python --version
```

If that says "command not found", or shows a version outside the range above, install **Miniforge** (a small, free Python distribution) from <https://conda-forge.org/download/> and then reopen your terminal.

> **Opening a terminal:** on **macOS** press `Cmd + Space`, type "Terminal", press Enter. On **Windows** press the Start button, type "Miniforge Prompt", press Enter. On **Linux** press `Ctrl + Alt + T`.

---

## Setup 🚀

You only do this once.

### Step 1 — Download the code

```
git clone https://github.com/OxWearables/preprocessing-edf.git
```

```
cd preprocessing-edf
```

Stay in this folder for every command that follows.

### Step 2 — Create a separate environment

An "environment" is a private space for this project's software, so it cannot clash with anything else on your computer. **Pick ONE of the two options below.**

**Option A — conda (recommended)**

```
conda create -n edfproc python=3.11 -y
```

```
conda activate edfproc
```

**Option B — venv (built into Python)**

```
python -m venv .venv
```

Then activate it. On **macOS or Linux**:

```
source .venv/bin/activate
```

On **Windows**:

```
.venv\Scripts\activate
```

✅ **How to tell it worked:** your terminal prompt now starts with `(edfproc)` or `(.venv)`.

> ⚠️ You must activate the environment **every time** you open a new terminal to use this pipeline. If your prompt does not show the name in brackets, run the `conda activate edfproc` (or `source .venv/bin/activate`) line again.

### Step 3 — Install the required software

```
pip install -r requirements.txt
```

This downloads a lot of packages and can take several minutes. It is finished when your prompt reappears.

### Step 4 — Download the heart-beat detector

The pipeline uses a machine-learning model to find heartbeats. This file is **not** included in this repository and must be downloaded separately.

1. Download the model file from: **[TODO: ADD DOWNLOAD LINK]**
2. Put the downloaded `.keras` file into the `models` folder inside `preprocessing-edf`.

That folder should then contain a file ending in `.keras`.

### Step 5 — Check everything works

```
python run_local.py --help
```

If you see a list of options, setup is complete. 🎉

---

## Running the pipeline 🏃

### Step 1 — Add your data

Put your `.edf` files into the `input_data` folder.

You can check what the pipeline has found without processing anything:

```
python run_local.py --dry-run
```

### Step 2 — Process the recordings

```
python run_local.py
```

This creates one results folder per recording inside `output`, and a combined spreadsheet.

**Roughly 30 seconds per file.** Progress is printed as it goes. You can leave it running and come back later.

### Step 3 — Make dataset-wide summary plots

Once processing has finished:

```
python generate_dataset_summary.py
```

### Step 4 — Look at one participant in detail (optional)

To see which participants have results:

```
python plot_subject.py --list
```

Then, using one of the names it printed:

```
python plot_subject.py --subject NAME_FROM_THE_LIST
```

Add `--show` to open the plots in a window instead of only saving them.

### Using folders elsewhere on your computer

If your data is not in `input_data`, point at it directly:

```
python run_local.py --input /path/to/my/edfs --output /path/to/my/results
```

---

## Changing settings ⚙️

All adjustable settings live in **`config.yaml`**. Open it in any text editor. Every setting is explained in comments, and every one is optional — delete a line and the pipeline uses its default.

You should **never need to edit a `.py` file** to change how the pipeline behaves.

Common things you might change:

| Setting | What it does |
|---|---|
| `input_dir`, `output_dir` | Where recordings are read from and results written to |
| `n_processes` | How many recordings to process at once. Set to `1` if you want clearer error messages |
| `night_start_hour`, `night_end_hour` | The hours treated as overnight rest (see the warning below) |
| `activity_*_mg` | The movement cut-points separating activity intensity levels |

If you make a mistake, the pipeline tells you exactly what is wrong and stops immediately, before processing anything.

---

## Outputs 📂

The pipeline generates an `./output/` directory containing:

* **Participant Folders**: Each contains a `plots` directory with a PDF summary report and a `processed_data` directory with detailed CSVs.
* **Top-Level Files**: An aggregated `df_info_summary.csv.gz` for all participants and summary plots (`.png`) for the entire dataset.

Every row of `df_info_summary.csv.gz` also records `model_file` and `model_sha256`, identifying exactly which detector produced those results.

---

## Things to be aware of ⚠️

**Resting heart rate and HRV are measured between 21:00 and 09:00 by default.** Participants who sleep outside those hours — shift workers, or someone who regularly sleeps 02:00–11:00 — will have these metrics computed from the wrong hours, or not at all. You can move the window in `config.yaml`, but it is a single fixed window applied to everybody. Automatic per-participant detection of the main sleep period is planned.

**Activity cut-points come from older adults.** The thresholds are from Etzkorn et al. (2024), derived in 381 adults with a median age of 78. Interpret them cautiously in younger populations.

---

## A Note on HRV Normalisation

This pipeline uses the natural logarithm of RMSSD to calculate a normalised HRV value. This is a standard statistical method that removes the mathematical influence of a person's average heart rate, allowing for fairer comparisons. For a detailed explanation, see this article by Marco Altini:
* [**Should we normalize HRV by heart rate?**](https://marcoaltini.substack.com/p/should-we-normalize-hrv-by-heart)

---

## If something goes wrong 🔧

| What you see | What to do |
|---|---|
| `command not found: conda` | Install Miniforge (see **Before you start**), then close and reopen your terminal |
| `No module named ...` | Your environment is not active. Run `conda activate edfproc` and try again |
| `No '*.keras' file found` | The detector model is missing. Redo **Setup Step 4** |
| `Found 2 model files` | Remove the spare `.keras` file from `models`, keeping only the one you want |
| `No .edf files found` | Your recordings are not in `input_data`, or do not end in `.edf` |
| `Configuration problem: ...` | There is a mistake in `config.yaml`. The message names the setting — fix it and rerun |
| Something failed for one file | The other files still process. Rerun with `--jobs 1` to see the error clearly |

---

## For developers 🛠️

Install with the development tools and run the test suite:

```
pip install -e ".[dev]"
```

```
pytest
```

```
ruff check .
```

Tests use only synthetic data — no recordings are needed. GitHub Actions runs the tests and linter on Python 3.9–3.12 for every push.

Installing the package also provides `edfproc`, `edfproc-plot` and `edfproc-summary` as commands you can run from any folder, equivalent to the `python ...` scripts above.

---

## Licence

MIT — see [LICENSE](LICENSE).

## Authors
* **Stefan van Duijvenboden**
* **Anna Bator**
