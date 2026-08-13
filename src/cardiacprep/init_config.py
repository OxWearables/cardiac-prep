"""The ``init`` subcommand: write a config.yaml into the current folder.

Someone who installs from PyPI has no clone, and therefore no config.yaml to
edit. This copies the annotated template out of the package so they get the
same starting point a clone gives, then names the handful of settings that
genuinely depend on their study rather than leaving them to find those among
the thirty-five in the file.

The values printed are read from the Config dataclass rather than typed out
here, so they cannot drift from the real defaults.
"""

import argparse
import shutil
import sys
from dataclasses import fields
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_FILENAME, Config, load_config
from .logging_utils import configure_logging, get_logger

log = get_logger("init")

TEMPLATE = Path(__file__).parent / "default_config.yaml"

# Settings worth a second look before a first run, grouped as they are read.
# Every name here is checked against the dataclass by the test suite, so a
# renamed setting cannot leave a stale prompt behind.
HIGHLIGHTS = [
    (
        ["fs_expected"],
        "Sampling rate of your ECG in Hz. The bundled detector was trained at\n"
        "250 Hz; a recording at another rate needs a model to match.",
    ),
    (
        ["night_start_hour", "night_end_hour"],
        "Hours treated as overnight rest. Resting heart rate and HRV are\n"
        "computed only from this window, so shift workers or anyone sleeping\n"
        "outside it will get the wrong answer or none at all.",
    ),
    (
        ["sleep_threshold_mg"],
        "Movement below this counts as being at rest, in milli-g.",
    ),
    (
        ["activity_very_light_mg", "activity_light_mg", "activity_moderate_mg"],
        "Activity intensity cut-points in milli-g, from Etzkorn et al. (2024).\n"
        "Derived in adults with a median age of 78 - if your participants are\n"
        "younger, these boundaries may not describe them well.",
    ),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a config.yaml you can edit.",
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --output /path/to/my/study\n"
            "  %(prog)s --force\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH",
        help=(
            "Where to set the study up. A folder gets "
            f"{DEFAULT_CONFIG_FILENAME} and the working folders inside it; a "
            "path ending .yaml names the config file itself. Defaults to the "
            "current folder."
        ),
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite an existing file. Without this, an existing file is left alone.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Write the file without printing the settings worth reviewing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed progress messages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


CONFIG_SUFFIXES = (".yaml", ".yml")


def _target_path(output) -> Path:
    """Where to write the config file.

    A path ending in .yaml or .yml names the file itself. Anything else is a
    folder to set the study up in, whether or not it exists yet - otherwise
    'init --output ./my-study' would create a file called my-study, which is
    not what anyone means by it.
    """
    if output is None:
        return Path.cwd() / DEFAULT_CONFIG_FILENAME
    path = Path(output).expanduser()
    if path.suffix.lower() in CONFIG_SUFFIXES:
        return path
    return path / DEFAULT_CONFIG_FILENAME


def _make_working_folders(config_path: Path):
    """Create the folders the pipeline expects, alongside the config file.

    Returns the paths created, most usefully so the caller can show the user
    where to put their recordings. Folders that already exist are reported
    too, since the point is to show the layout rather than to report news.
    """
    config = load_config(path=config_path)
    made = []
    for label, folder in (
        ("recordings go here", config.input_dir),
        ("results appear here", config.output_dir),
        ("detector weights", config.model_dir),
    ):
        path = Path(folder).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        try:
            path.mkdir(parents=True, exist_ok=True)
            made.append((label, path))
        except OSError as exc:
            log.warning("Could not create %s: %s", path, exc)
    return made


def _shorten(path: Path) -> str:
    """Show a path relative to the current folder when it is inside it."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _describe_folders(created) -> str:
    if not created:
        return ""
    shown = [(label, _shorten(p)) for label, p in created]
    width = max(len(p) for _, p in shown)
    lines = ["", "Folders ready:", ""]
    lines += [f"  {p:<{width}}   {label}" for label, p in shown]
    return "\n".join(lines)


def _describe_highlights(defaults) -> str:
    lines = [
        "",
        "Worth checking before your first run:",
        "",
    ]
    for names, why in HIGHLIGHTS:
        for name in names:
            lines.append(f"  {name}: {defaults[name]}")
        for line in why.splitlines():
            lines.append(f"      {line}")
        lines.append("")
    lines.append(
        "Everything else has a sensible default, and every setting is optional -"
    )
    lines.append("delete a line and the pipeline falls back to the default.")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    target = _target_path(args.output)

    if target.exists() and not args.force:
        print(
            f"\n'{target}' already exists, so it has been left alone.\n"
            "Pass --force to overwrite it, or --output to write somewhere else.\n",
            file=sys.stderr,
        )
        return 1

    if not TEMPLATE.is_file():
        print(
            f"\nThe packaged template is missing from '{TEMPLATE}'.\n"
            "This means the installation is incomplete; reinstalling should fix it.\n",
            file=sys.stderr,
        )
        return 1

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEMPLATE, target)
    except OSError as exc:
        print(f"\nCould not write '{target}': {exc}\n", file=sys.stderr)
        return 1

    # A clone arrives with these folders already present. An install from PyPI
    # does not, and pip cannot create them: it writes into site-packages and
    # has no idea where the study will live. So they are made here, where the
    # user has chosen a working folder, rather than left to fail on first run.
    created = _make_working_folders(target)

    print(f"Created {target}")

    if not args.quiet:
        print(_describe_folders(created))
        defaults = {f.name: getattr(Config(), f.name) for f in fields(Config)}
        print(_describe_highlights(defaults))

    return 0


if __name__ == "__main__":
    sys.exit(main())
