"""Dataset-level summary plots across every processed participant.

Reads the aggregated df_info_summary.csv.gz written by the pipeline and
produces distribution and relationship plots for the cohort as a whole.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from . import __version__
from .config import ConfigError, load_config
from .logging_utils import configure_logging, get_logger
from .plot_utils import (
    LIGHT_GREEN,
    MODERATE_YELLOW,
    NEUTRAL_GRAY,
    SEDENTARY,
    VIGOROUS_RED,
)

SUMMARY_FILENAME = "df_info_summary.csv.gz"
HIST_FILENAME = "dataset_summary_histograms.png"
SCATTER_FILENAME = "dataset_rhr_vs_hrv.png"

# Without these the summary file predates the current pipeline.
REQUIRED_COLUMNS = ("HR_rest_robust", "median_daily_rmssd")

log = get_logger("summary")


def plot_hrv_vs_rhr(df, save_path):
    """Resting heart rate against HRV, with a regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(
        x="HR_rest_robust",
        y="median_daily_rmssd",
        data=df,
        ax=ax,
        scatter_kws={"alpha": 0.6, "color": SEDENTARY},
        line_kws={"color": VIGOROUS_RED},
    )

    ax.set_title("Resting heart rate vs heart rate variability (RMSSD)", fontsize=16)
    ax.set_xlabel("Robust resting HR (bpm)", fontsize=12)
    ax.set_ylabel("Median daily RMSSD (ms)", fontsize=12)

    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_dataset_histograms(df, save_path):
    """A 3x2 grid showing how key metrics are distributed across participants."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Summary of processed dataset metrics", fontsize=18, y=1.02)

    panels = [
        (axes[0, 0], "HR_rest_robust", "Distribution of resting heart rate",
         "Robust resting HR (bpm)", SEDENTARY, None),
        (axes[0, 1], "median_daily_rmssd", "Distribution of heart rate variability (RMSSD)",
         "Median daily RMSSD (ms)", LIGHT_GREEN, None),
        # MVPA has a long right tail; trimming the top 0.5% keeps the bulk legible.
        (axes[1, 0], "hours_mvpa", "Distribution of daily MVPA",
         "Average hours per day", MODERATE_YELLOW, 0.995),
        (axes[1, 1], "hours_light_activity", "Distribution of daily light activity",
         "Average hours per day", LIGHT_GREEN, None),
        (axes[2, 0], "prop_ECG_passed_finalQC", "Distribution of usable ECG data",
         "Proportion of high-quality ECG segments", NEUTRAL_GRAY, None),
        (axes[2, 1], "frac_RR_imp", "Distribution of imputed data",
         "Fraction of heart rate data imputed", NEUTRAL_GRAY, None),
    ]

    for ax, column, title, xlabel, colour, upper_quantile in panels:
        if column not in df.columns:
            ax.set_visible(False)
            log.warning("Column '%s' missing from the summary file; panel skipped.", column)
            continue

        values = df[column].dropna()
        if upper_quantile is not None and not values.empty:
            values = values[values <= values.quantile(upper_quantile)]

        sns.histplot(values, kde=True, ax=ax, color=colour)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if column.startswith(("prop_", "frac_")):
            ax.set_xlim(0, 1)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="edfproc-summary",
        description=(
            "Generate dataset-level summary plots from the aggregated results "
            "written by the pipeline."
        ),
        epilog=(
            "Examples:\n"
            "  edfproc-summary\n"
            "  edfproc-summary --output /path/to/results\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output", metavar="FOLDER",
        help="Results folder to read from and write into. Overrides output_dir in config.yaml.",
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE",
        help="Path to a settings file. Defaults to config.yaml if present.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed progress messages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        config = load_config(
            path=Path(args.config) if args.config else None,
            overrides={"output_dir": args.output},
        )
    except ConfigError as exc:
        print(f"\nConfiguration problem:\n\n{exc}\n", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    summary_path = output_dir / SUMMARY_FILENAME

    if not summary_path.is_file():
        print(
            f"\nSummary file not found: '{summary_path}'\n"
            "Process your recordings first:  python run_local.py\n",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(summary_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(
            f"\n'{summary_path}' is missing required column(s): {', '.join(missing)}.\n"
            "It was probably written by an older version of the pipeline. "
            "Re-run the pipeline to regenerate it.\n",
            file=sys.stderr,
        )
        return 1

    if df.empty:
        print(f"\n'{summary_path}' contains no participants.\n", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} participants from {summary_path}")

    hist_path = plot_dataset_histograms(df.copy(), output_dir / HIST_FILENAME)
    print(f"Saved {hist_path}")

    scatter_path = plot_hrv_vs_rhr(df.copy(), output_dir / SCATTER_FILENAME)
    print(f"Saved {scatter_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
