#!/usr/bin/env python3
"""
Run the pipeline from a clone of this repository, without installing anything.

    python process.py
    python process.py --input /path/to/edfs --output /path/to/results
    python process.py --help

If you have installed the package (pip install .), the 'edfproc' command does
exactly the same thing from any folder.
"""
__author__ = "Anna Bator"

import sys
from pathlib import Path

# Make the package importable straight from the source tree, so this script
# works in a fresh clone with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from edfproc.cli import main  # noqa: E402  (import must follow the path setup)

if __name__ == "__main__":
    sys.exit(main())
