"""Command line interface for the preprocessing pipeline.

Reachable either as the installed ``edfproc`` command or by running
``python process.py`` from a clone of the repository.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .config import ConfigError, load_config
from .logging_utils import configure_logging, get_logger
from .model_utils import ModelError, find_model

EDF_SUFFIXES = (".edf",)

log = get_logger("cli")


def parse_args(argv=None):
    # prog is left unset so the help text names the command the user actually
    # typed: "process.py" from a clone, "edfproc" once installed.
    parser = argparse.ArgumentParser(
        description=(
            "Process wearable ECG and accelerometer recordings from EDF files. "
            "Settings come from config.yaml; the options below override it."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --input ~/my_edfs --output ~/results\n"
            "  %(prog)s --jobs 1        (one at a time, clearer errors)\n"
            "  %(prog)s --dry-run       (list what would be processed)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", metavar="FOLDER",
        help="Folder containing .edf files. Overrides input_dir in config.yaml.",
    )
    parser.add_argument(
        "-o", "--output", metavar="FOLDER",
        help="Folder to write results into. Overrides output_dir in config.yaml.",
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE",
        help="Path to a settings file. Defaults to config.yaml if present.",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, metavar="N",
        help="Number of recordings to process at once. Default: all cores but one.",
    )
    parser.add_argument(
        "-m", "--model", metavar="FILE",
        help="QRS detector weights (.keras). Default: the single .keras file in the models folder.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the files that would be processed, then exit without processing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress messages.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    return parser.parse_args(argv)


def find_edf_files(input_dir):
    """Return every EDF file in input_dir, matched case-insensitively.

    Deduplicates by resolved path so a case-insensitive filesystem cannot
    yield the same recording twice.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: '{input_dir}'")

    seen = {}
    for entry in sorted(input_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in EDF_SUFFIXES:
            seen.setdefault(entry.resolve(), entry)
    return [str(path) for path in sorted(seen.values())]


def main(argv=None):
    args = parse_args(argv)
    configure_logging(verbose=args.verbose, include_process=True)

    # Settings: file defaults first, then any command-line overrides.
    try:
        config = load_config(
            path=Path(args.config) if args.config else None,
            overrides={
                "input_dir": args.input,
                "output_dir": args.output,
                "n_processes": args.jobs,
                "model_path": args.model,
            },
        )
    except ConfigError as exc:
        print(f"\nConfiguration problem:\n\n{exc}\n", file=sys.stderr)
        return 2

    # Check the model before starting a long run, not thirty minutes into one.
    try:
        model_path = find_model(config.model_dir, config.model_path)
    except ModelError as exc:
        print(f"\nCould not load the QRS detector:\n\n{exc}\n", file=sys.stderr)
        return 2

    try:
        edf_files = find_edf_files(config.input_dir)
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    if not edf_files:
        print(
            f"\nNo .edf files found in '{config.input_dir}'.\n"
            "Put your recordings there, or point elsewhere with --input.\n",
            file=sys.stderr,
        )
        return 1

    n_processes = min(config.resolved_n_processes, len(edf_files))

    print(f"Input:   {config.input_dir}")
    print(f"Output:  {config.output_dir}")
    print(f"Model:   {model_path.name}")
    print(f"Files:   {len(edf_files)}")
    print(f"Workers: {n_processes}")

    if args.dry_run:
        print("\nDry run - these files would be processed:")
        for path in edf_files:
            print(f"  {os.path.basename(path)}")
        return 0

    os.makedirs(config.output_dir, exist_ok=True)

    # Imported here so --help and configuration errors stay fast: importing
    # proc_edf pulls in the plotting stack, which is slow to load.
    from .proc_edf import init_worker, procEDF_wrapper

    start_time = time.time()
    with mp.Pool(
        processes=n_processes,
        initializer=init_worker,
        initargs=(config, args.verbose),
    ) as pool:
        results = pool.map(procEDF_wrapper, edf_files)

    df_info_all = pd.concat(results, ignore_index=True)
    output_path = os.path.join(str(config.output_dir), "df_info_summary.csv.gz")
    df_info_all.to_csv(output_path, compression="gzip", index=False)

    duration = time.time() - start_time
    n_failed = int(df_info_all["failed"].sum()) if "failed" in df_info_all else 0

    print("\n-----------------------------------------")
    print("Processing complete.")
    print(f"   - Files processed:    {len(edf_files)}")
    if n_failed:
        print(f"   - Files that FAILED:  {n_failed}  (see errors above)")
    print(f"   - Time elapsed:       {duration / 60:.2f} minutes")
    print(f"   - Average per file:   {duration / len(edf_files):.2f} seconds")
    print(f"   - Summary written to: {output_path}")
    print(f"   - Per-file results:   {config.output_dir}")
    print("-----------------------------------------")

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
