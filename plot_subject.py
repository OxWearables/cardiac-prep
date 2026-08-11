#!/usr/bin/env python3
"""
Plot one participant's results, from a clone of this repository.

    python plot_subject.py --list
    python plot_subject.py --subject 001_recording
    python plot_subject.py --help

This is a shortcut for the 'inspect' subcommand. If you have installed the
package (pip install .), the same thing from any folder is:

    cardiac-prep inspect
"""
__author__ = "Anna Bator"

import sys
from pathlib import Path

# Make the package importable straight from the source tree, so this script
# works in a fresh clone with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from edfproc.entry import main  # noqa: E402  (import must follow the path setup)

if __name__ == "__main__":
    sys.exit(main(["inspect", *sys.argv[1:]]))
