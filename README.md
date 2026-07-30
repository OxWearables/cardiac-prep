# ECG & Accelerometry (EDF) Preprocessing Pipeline

An end-to-end Python pipeline for processing multi-day wearable sensor data from `.EDF` files. It extracts heart rate, heart rate variability (HRV) and physical activity metrics, performs quality control and imputation, and generates summary reports and visualisations.

**You do not need to be a programmer to use this.** Follow the steps in order, copying each command exactly.

---

## 🚦 The three commands

Once set up, this is the whole pipeline. Run them in this order:

### 1. Process your recordings

```
python process.py
```

Reads every `.edf` file in `input_data` and writes results to `output`. About 30 seconds per file.

### 2. Summarise the whole dataset

```
python summarise_dataset.py
```

Creates population-level plots across all participants.

### 3. Inspect one participant (optional)

```
python plot_subject.py --list
```

```
python plot_subject.py --subject NAME_FROM_THE_LIST
```

Add `--show` to open the plots in a window as well as saving them.

> **First time here?** Do the [Setup](#setup-) below first. Add `--help` to any command to see its options.

---

## Features ✨
* **Parallel Processing**: Uses all **CPU** cores except one, so your machine stays usable (no GPU needed). Configurable via `n_processes`.
* **Preprocessing**: QRS detection, artifact removal, signal quality assessment.
* **Data Imputation**: Linear interpolation for short gaps, time-of-day averaging for longer ones.
* **Comprehensive Metrics**:
    * Heart Rate (Resting, Min, Max, Average)
    * Heart Rate Variability (RMSSD and log-normalised RMSSD to account for HR)
    * Physical Activity (time in Sedentary, Light, Moderate and Vigorous zones)
* **Automated Reporting**: A 2-page PDF summary report per participant.
* **Dataset Summarisation**: Population-level trend plots.

---

## Before you start 📝

You need **Python 3.9, 3.10, 3.11 or 3.12**. Python 3.13 is not yet supported.

To check what you have, open a terminal and type:

```
python --version
```

If that says "command not found", or shows a version outside the range, install **Miniforge** from <https://conda-forge.org/download/> and reopen your terminal.

> **Opening a terminal:** on **macOS** press `Cmd + Space`, type "Terminal", press Enter. On **Windows** press Start, type "Miniforge Prompt", press Enter. On **Linux** press `Ctrl + Alt + T`.

---

## Setup 🚀

You only do this once.

### Step 1 – Download the code

```
git clone https://github.com/OxWearables/preprocessing-edf.git
```

```
cd preprocessing-edf
```

Stay in this folder for every command that follows.

### Step 2 – Create a separate environment

An environment is a private space for this project's software, so it cannot clash with anything else on your computer. **Pick ONE option.**

**Option A – conda (recommended)**

```
conda create -n edfproc python=3.11 -y
```

```
conda activate edfproc
```

**Option B – venv (built into Python)**

```
python -m venv .venv
```

Activate it. On **macOS or Linux**:

```
source .venv/bin/activate
```

On **Windows**:

```
.venv\Scripts\activate
```

✅ **How to tell it worked:** your prompt now starts with `(edfproc)` or `(.venv)`.

> ⚠️ Activate the environment **every time** you open a new terminal. If your prompt does not show the name in brackets, run the activate line again.

### Step 3 – Install the required software

```
pip install -r requirements.txt
```

This takes several minutes. It is finished when your prompt reappears.

### Step 4 – Download the heart-beat detector

The pipeline uses a machine-learning model to find heartbeats. This file is **not** included in this repository.

1. Download it from: **[TODO: ADD DOWNLOAD LINK]**
2. Put the `.keras` file into the `models` folder.

### Step 5 – Check it works

```
python process.py --help
```

If you see a list of options, setup is complete. 🎉

---

## Running it 🏃

Put your `.edf` files into the `input_data` folder, then use [the three commands](#-the-three-commands) above.

To preview what will be processed without processing anything:

```
python process.py --dry-run
```

To use folders elsewhere on your computer:

```
python process.py --input /path/to/my/edfs --output /path/to/my/results
```

---

## Changing settings ⚙️

All settings live in **`config.yaml`**. Open it in any text editor. Every setting is explained in comments and every one is optional, so deleting a line restores its default.

You should **never need to edit a `.py` file**.

| Setting | What it does |
|---|---|
| `input_dir`, `output_dir` | Where recordings are read from and results written to |
| `n_processes` | How many recordings to process at once. Set to `1` for clearer error messages |
| `night_start_hour`, `night_end_hour` | The hours treated as overnight rest (see below) |
| `activity_*_mg` | Movement cut-points separating activity intensity levels |

Mistakes are reported clearly and stop the run immediately, before anything is processed.

---

## Outputs 📂

An `./output/` directory containing:

* **Participant folders**: a `plots` directory with the PDF report, and a `processed_data` directory with detailed CSVs.
* **Top-level files**: an aggregated `df_info_summary.csv.gz` plus dataset-wide summary plots.

Every row of `df_info_summary.csv.gz` records `model_file` and `model_sha256`, identifying exactly which detector produced those results.

---

## Things to be aware of ⚠️

**Resting heart rate and HRV use a fixed 21:00–09:00 window.** Participants who sleep outside those hours, such as shift workers or someone who regularly sleeps 02:00–11:00, will have these metrics computed from the wrong hours or not at all. You can move the window in `config.yaml`, but it is one fixed window applied to everybody.

*Planned improvement:* detect each participant's main sleep period from their own accelerometer data instead of assuming a clock window.

**Activity cut-points come from older adults.** The thresholds are from Etzkorn et al. (2024), derived in 381 adults with a median age of 78. Interpret cautiously in younger populations.

---

## A note on HRV normalisation

This pipeline uses the natural logarithm of RMSSD to normalise HRV. This removes the mathematical influence of a person's average heart rate, allowing fairer comparisons. For a detailed explanation, see [Should we normalize HRV by heart rate?](https://marcoaltini.substack.com/p/should-we-normalize-hrv-by-heart) by Marco Altini.

---

## If something goes wrong 🔧

| What you see | What to do |
|---|---|
| `command not found: conda` | Install Miniforge, then close and reopen your terminal |
| `No module named ...` | Your environment is not active. Run `conda activate edfproc` |
| `No '*.keras' file found` | The detector model is missing. Redo **Setup Step 4** |
| `Found 2 model files` | Remove the spare `.keras` file from `models` |
| `No .edf files found` | Your recordings are not in `input_data`, or do not end in `.edf` |
| `Configuration problem: ...` | A mistake in `config.yaml`. The message names the setting |
| One file failed | The others still process. Rerun with `--jobs 1` to see the error clearly |

---

## For developers 🛠️

```
pip install -e ".[dev]"
```

```
pytest
```

Tests use synthetic data only, so no recordings or model weights are needed. GitHub Actions runs the tests and `ruff check .` on Python 3.9–3.12 for every push.

Installing also provides `edfproc`, `edfproc-plot` and `edfproc-summary` as commands runnable from any folder.

---

## Licence

MIT, see [LICENSE](LICENSE).

## Authors
* **Stefan van Duijvenboden**
* **Anna Bator**
