# cardiac-prep

An end-to-end Python pipeline for processing multi-day wearable sensor data
from `.EDF` files. It extracts heart rate, heart rate variability (HRV) and
physical activity metrics, performs quality control and imputation, and
generates summary reports and visualisations.

**You do not need to be a programmer to use this.** Follow
[Installation](installation.md), then [Quickstart](quickstart.md), copying each
command exactly.

## What it does

- **Preprocessing** – QRS detection, artefact removal, signal quality
  assessment.
- **Quality control** — per-segment screening for non-wear, saturation and
  noise, with every decision recorded in the output.
- **Imputation** — linear interpolation for short gaps, time-of-day averaging
  for longer ones, with a flag marking every filled value.
- **Metrics** — resting, minimum, maximum and mean heart rate; RMSSD and
  log-normalised RMSSD; time spent at each physical activity intensity.
- **Reporting** — a two-page PDF per participant, plus population-level plots
  across a whole dataset.
- **Parallel processing** — uses every CPU core but one, so your machine stays
  usable. No GPU needed.

## Where to start

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Page
  - Read it when
* - [Installation](installation.md)
  - Setting the pipeline up for the first time.
* - [Quickstart](quickstart.md)
  - Pipeline is installed and you want to process recordings.
* - [Configuration reference](configuration.md)
  - You need to change a threshold, a folder, or the night-time window.
* - [Outputs](outputs.md)
  - You have results and want to know what the columns mean.
* - [Interpreting results](interpreting.md)
  - Before drawing conclusions - the assumptions worth knowing about.
* - [Troubleshooting](troubleshooting.md)
  - Something failed and you want the fix.
* - [Development](development.md)
  - You want to run the tests or contribute a change.
* - [API reference](api.md)
  - You are calling the pipeline from your own Python code.
:::

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
configuration
outputs
interpreting
troubleshooting
development
api
```

## Licence

Academic Use Licence. Free for academic, non-commercial research and teaching;
commercial use requires a separate licence from Oxford University Innovation
(<enquiries@innovation.ox.ac.uk>).

Copyright © 2026, University of Oxford.
