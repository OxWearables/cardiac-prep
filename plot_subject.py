#!/usr/bin/env python3
"""
Plot one participant's results, from a clone of this repository.

    python plot_subject.py --list
    python plot_subject.py --subject 001_recording
    python plot_subject.py --help

Replaces the older quick_plot.py and visualise_results.py, which each needed a
participant id edited into the source.

If you have installed the package (pip install .), the 'edfproc-plot' command
does exactly the same thing from any folder.
"""
__author__ = "Anna Bator"

import sys
from pathlib import Path

# Make the package importable straight from the source tree, so this script
# works in a fresh clone with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from edfproc.subject_plots import main  # noqa: E402  (import must follow the path setup)

if __name__ == "__main__":
    sys.exit(main())
