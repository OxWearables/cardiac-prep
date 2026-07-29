"""Pytest configuration.

Puts the repository root on sys.path so the flat top-level modules
(proc_utils, read_utils, proc_edf, ...) are importable from tests, and forces a
non-interactive matplotlib backend so importing plot_utils never tries to open
a window on a headless machine or in CI.
"""

import os
import sys

# Must be set before matplotlib.pyplot is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
