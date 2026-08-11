# Installation

You only do this once.

## Before you start

You need **Python 3.9, 3.10, 3.11 or 3.12**. Python 3.13 is not yet supported.

To check what you have, open a terminal and type:

```text
python --version
```

If that says "command not found", or shows a version outside the range, install
[Miniforge](https://conda-forge.org/download/) and reopen your terminal.

:::{admonition} Opening a terminal
:class: tip

On **macOS** press `Cmd + Space`, type "Terminal", press Enter. On **Windows**
press Start, type "Miniforge Prompt", press Enter. On **Linux** press
`Ctrl + Alt + T`.
:::

## Step 1 - Download the code

```text
git clone https://github.com/OxWearables/cardiac-prep.git
```

```text
cd cardiac-prep
```

Stay in this folder for every command that follows.

## Step 2 - Create a separate environment

An environment is a private space for this project's software, so it cannot
clash with anything else on your computer. **Pick one option.**

### Option A - conda (recommended)

```text
conda create -n edfproc python=3.11 -y
```

```text
conda activate edfproc
```

### Option B - venv (built into Python)

```text
python -m venv .venv
```

Activate it. On **macOS or Linux**:

```text
source .venv/bin/activate
```

On **Windows**:

```text
.venv\Scripts\activate
```

**How to tell it worked:** your prompt now starts with `(edfproc)` or
`(.venv)`.

:::{warning}
Activate the environment **every time** you open a new terminal. If your prompt
does not show the name in brackets, run the activate line again.
:::

## Step 3 - Install the required software

```text
pip install -r requirements.txt
```

This takes several minutes. It is finished when your prompt reappears.

## Step 4 - Download the heart-beat detector

The pipeline uses a machine-learning model to find heartbeats. This file is
**not** included in the repository.

1. Download it from: **[TODO: ADD DOWNLOAD LINK]**
2. Put the `.keras` file into the `models` folder.

The pipeline finds the file by extension, so the exact name does not matter -
but there must be exactly one `.keras` file in that folder. Every output row
records the file name and its SHA-256, so you can always tell which detector
produced a given result.

## Step 5 - Check it works

```text
python process.py --help
```

If you see a list of options, setup is complete.
