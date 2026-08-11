"""Loading and reshaping one participant's processed output.

Shared by the subject-level plotting commands so they agree on where files
live and how derived views are computed.

The pipeline writes a single per-subject file, ``*_df_qc.csv.gz``, holding
10-second resolution data. The daily and 24-hour views are derived from it
here rather than read from disk, because no stage of the pipeline writes them.
"""

from pathlib import Path

import pandas as pd

from .logging_utils import get_logger

QC_GLOB = "*_df_qc.csv.gz"
PROCESSED_SUBDIR = "processed_data"

log = get_logger("subject")


class SubjectNotFoundError(Exception):
    """Raised when a participant has no processed output to plot."""


def list_subjects(output_dir):
    """Return the ids of every participant with processed output, sorted.

    A subject counts as processed only if the QC file is actually present, so
    a half-finished or failed run does not show up as available.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []

    subjects = []
    for entry in sorted(output_dir.iterdir()):
        if entry.is_dir() and any((entry / PROCESSED_SUBDIR).glob(QC_GLOB)):
            subjects.append(entry.name)
    return subjects


def _describe_available(output_dir):
    """Build the 'here is what you can plot instead' half of an error message."""
    subjects = list_subjects(output_dir)
    if not subjects:
        return (
            f"No processed participants found in '{output_dir}'.\n"
            "Run the pipeline first:  python process.py"
        )
    listing = "\n  ".join(subjects)
    return f"Available participants in '{output_dir}':\n  {listing}"


def load_subject_qc(output_dir, subject_id):
    """Load a participant's 10-second QC table, indexed by time.

    Raises:
        SubjectNotFoundError: The participant folder or its QC file is absent.
            The message lists what is available, since a mistyped id is the
            most likely cause.
    """
    output_dir = Path(output_dir)
    subject_path = output_dir / subject_id

    if not subject_path.is_dir():
        raise SubjectNotFoundError(
            f"No results folder for '{subject_id}' in '{output_dir}'.\n\n"
            + _describe_available(output_dir)
        )

    matches = sorted((subject_path / PROCESSED_SUBDIR).glob(QC_GLOB))
    if not matches:
        raise SubjectNotFoundError(
            f"'{subject_id}' has no {QC_GLOB} file in its "
            f"{PROCESSED_SUBDIR} folder, so it cannot be plotted.\n"
            "This usually means processing failed for that recording."
        )

    if len(matches) > 1:
        log.warning(
            "%s has %d QC files; using %s", subject_id, len(matches), matches[0].name
        )

    df = pd.read_csv(matches[0], parse_dates=["time"])
    return df.set_index("time")


def daily_summary(df_qc):
    """Average heart rate per calendar day.

    Derived from the 10-second data because the pipeline writes no daily file.
    """
    if "RRm_imputed" not in df_qc.columns:
        raise KeyError("df_qc has no 'RRm_imputed' column, so heart rate cannot be derived.")

    daily = df_qc.groupby(df_qc.index.date)["RRm_imputed"].mean().to_frame("RRm_imputed")
    daily.index.name = "date"
    daily["HRm_imputed"] = 60 * 1000 / daily["RRm_imputed"]
    return daily


def daily_profile(df_qc):
    """Median heart rate by time of day, averaged across all recorded days.

    Uses the median rather than the mean, matching how the PDF report builds
    its 24-hour profile, so the two cannot disagree.
    """
    if "RRm_imputed" not in df_qc.columns:
        raise KeyError("df_qc has no 'RRm_imputed' column, so heart rate cannot be derived.")

    hour = df_qc.index.hour
    minute = df_qc.index.minute
    profile = (
        df_qc.groupby([hour, minute])["RRm_imputed"].median().to_frame("RRm_median")
    )
    profile.index.names = ["hour", "minute"]
    profile = profile.reset_index()
    profile["time_of_day"] = (
        profile["hour"].astype(str).str.zfill(2)
        + ":"
        + profile["minute"].astype(str).str.zfill(2)
    )
    profile["HRm_median"] = 60 * 1000 / profile["RRm_median"]
    return profile
