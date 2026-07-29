"""Plots for a single participant, for inspection after a pipeline run.

Replaces the earlier quick_plot.py and visualise_results.py, which duplicated
the same loading logic and each required editing a subject id into the source.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from . import __version__
from .config import ConfigError, load_config
from .logging_utils import configure_logging, get_logger
from .plot_utils import SEDENTARY
from .subject_data import (
    SubjectNotFoundError,
    daily_profile,
    daily_summary,
    list_subjects,
    load_subject_qc,
)

log = get_logger("plots")

PLOT_KINDS = ("timeseries", "daily", "profile", "heatmap")

DAYS_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def plot_timeseries(df_qc, subject_id):
    """Heart rate and movement at full 10-second resolution."""
    fig, ax_hr = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"10-second heart rate and activity: {subject_id}", fontsize=15)

    ax_hr.plot(df_qc.index, df_qc["HRm_imputed"], color="tab:blue", linewidth=0.8)
    ax_hr.set_xlabel("Time")
    ax_hr.set_ylabel("Heart rate (bpm)", color="tab:blue")
    ax_hr.tick_params(axis="y", labelcolor="tab:blue")

    ax_acc = ax_hr.twinx()
    ax_acc.plot(df_qc.index, df_qc["acc_imputed"], color="tab:orange", alpha=0.6, linewidth=0.8)
    ax_acc.set_ylabel("Acceleration (milli-g)", color="tab:orange")
    ax_acc.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_daily(df_qc, subject_id):
    """Average heart rate for each day of the recording."""
    daily = daily_summary(df_qc)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(daily)), daily["HRm_imputed"], color=SEDENTARY)

    ax.set_title(f"Average daily heart rate: {subject_id}", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average heart rate (bpm)")
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels([d.strftime("%d %b") for d in daily.index], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_profile(df_qc, subject_id):
    """Typical 24-hour heart rate rhythm, pooled across all recorded days."""
    profile = daily_profile(df_qc)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(profile)), profile["HRm_median"], color="purple", linewidth=1.2)

    # One tick every two hours; the profile has one row per minute of the day.
    step = max(1, len(profile) // 12)
    ax.set_xticks(range(0, len(profile), step))
    ax.set_xticklabels(profile["time_of_day"].iloc[::step], rotation=45, ha="right")

    ax.set_title(f"Typical 24-hour heart rate profile: {subject_id}", fontsize=15)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Median heart rate (bpm)")
    ax.grid(linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def _heatmap(df_qc, column, title, cbar_label):
    """Mean of a column by day of week and hour of day."""
    frame = df_qc.copy()
    frame["day_of_week"] = frame.index.day_name()
    frame["hour_of_day"] = frame.index.hour

    pivot = frame.pivot_table(
        index="day_of_week", columns="hour_of_day", values=column, aggfunc="mean"
    ).reindex(DAYS_ORDER)

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(pivot, cmap="viridis", ax=ax, cbar_kws={"label": cbar_label})
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")

    fig.tight_layout()
    return fig


def plot_heatmaps(df_qc, subject_id):
    """Weekly rhythm heatmaps for heart rate and for movement."""
    return [
        ("hr_heatmap", _heatmap(
            df_qc, "HRm_imputed",
            f"Average heart rate by hour and weekday: {subject_id}",
            "Heart rate (bpm)")),
        ("acc_heatmap", _heatmap(
            df_qc, "acc_imputed",
            f"Average activity by hour and weekday: {subject_id}",
            "Acceleration (milli-g)")),
    ]


def parse_args(argv=None):
    # prog is left unset so the help text names the command the user typed.
    parser = argparse.ArgumentParser(
        description="Plot one participant's processed results.",
        epilog=(
            "Examples:\n"
            "  %(prog)s --list\n"
            "  %(prog)s --subject 001_recording\n"
            "  %(prog)s --subject 001_recording --kind heatmap --show\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s", "--subject", metavar="ID",
        help="Participant to plot, as named by its folder inside the output directory.",
    )
    parser.add_argument(
        "-l", "--list", action="store_true",
        help="List the participants that have processed results, then exit.",
    )
    parser.add_argument(
        "-o", "--output", metavar="FOLDER",
        help="Results folder to read from. Overrides output_dir in config.yaml.",
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE",
        help="Path to a settings file. Defaults to config.yaml if present.",
    )
    parser.add_argument(
        "-k", "--kind", nargs="+", choices=PLOT_KINDS + ("all",), default=["all"],
        help="Which plots to make. Default: all.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the plots in a window instead of only saving them.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write PNG files. Useful together with --show.",
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

    if args.list:
        subjects = list_subjects(config.output_dir)
        if not subjects:
            print(
                f"No processed participants found in '{config.output_dir}'.\n"
                "Run the pipeline first:  python process.py",
                file=sys.stderr,
            )
            return 1
        print(f"Participants with results in '{config.output_dir}':")
        for subject in subjects:
            print(f"  {subject}")
        return 0

    if not args.subject:
        print(
            "\nWhich participant? Pass --subject ID, or --list to see what is "
            "available.\n",
            file=sys.stderr,
        )
        return 2

    try:
        df_qc = load_subject_qc(config.output_dir, args.subject)
    except SubjectNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    kinds = PLOT_KINDS if "all" in args.kind else tuple(args.kind)
    plots_dir = Path(config.output_dir) / args.subject / "plots"
    if not args.no_save:
        plots_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    if "timeseries" in kinds:
        figures.append(("timeseries", plot_timeseries(df_qc, args.subject)))
    if "daily" in kinds:
        figures.append(("daily_heart_rate", plot_daily(df_qc, args.subject)))
    if "profile" in kinds:
        figures.append(("24hr_profile", plot_profile(df_qc, args.subject)))
    if "heatmap" in kinds:
        figures.extend(plot_heatmaps(df_qc, args.subject))

    if not args.no_save:
        for name, fig in figures:
            path = plots_dir / f"{args.subject}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")

    if args.show:
        plt.show()
    else:
        for _, fig in figures:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
