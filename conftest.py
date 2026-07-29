"""Pytest configuration.

Puts ``src`` on sys.path so the tests import the package straight from the
source tree, with or without an editable install, and forces a non-interactive
matplotlib backend so importing plot_utils never tries to open a window on a
headless machine or in CI.
"""

import os
import sys
from pathlib import Path

# Must be set before matplotlib.pyplot is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
