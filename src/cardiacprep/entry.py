"""Single command-line entry point.

The pipeline used to expose three separate commands with no visible
relationship to each other. They are now subcommands of one:

    cardiac-prep process     process .edf recordings
    cardiac-prep summarise   summarise a whole processed dataset
    cardiac-prep inspect     plot one participant's results

Each subcommand keeps its own argument parser, so ``cardiac-prep process
--help`` shows exactly what ``process.py --help`` always showed. This module
only decides which one to call.

Subcommand modules are imported lazily. Importing the processing code pulls in
TensorFlow, which takes seconds; there is no reason for ``cardiac-prep --help``
or ``cardiac-prep inspect`` to pay that cost.
"""

import importlib
import sys
from typing import Dict, List, NamedTuple, Optional

from . import __version__


class Command(NamedTuple):
    module: str
    summary: str


# Order here is the order shown in --help: the order you would run them in.
COMMANDS: Dict[str, Command] = {
    "init": Command(
        "cardiacprep.init_config",
        "Create a config.yaml you can edit.",
    ),
    "process": Command(
        "cardiacprep.cli",
        "Process .edf recordings into per-participant metrics and reports.",
    ),
    "summarise": Command(
        "cardiacprep.dataset_summary",
        "Build population-level plots across a processed dataset.",
    ),
    "inspect": Command(
        "cardiacprep.subject_plots",
        "Plot one participant's processed results.",
    ),
}

# Accepted but not advertised, so American spelling does not fail.
ALIASES = {"summarize": "summarise", "plot": "inspect", "run": "process"}

PROG = "cardiac-prep"


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        f"usage: {PROG} <command> [options]",
        "",
        "Process multi-day wearable ECG and accelerometry from .edf files.",
        "",
        "Commands:",
    ]
    lines += [
        f"  {name:<{width}}  {command.summary}" for name, command in COMMANDS.items()
    ]
    lines += [
        "",
        f"Run '{PROG} <command> --help' for the options of a single command.",
        "",
        "Examples:",
        f"  {PROG} init",
        f"  {PROG} process --input ./input_data",
        f"  {PROG} summarise",
        f"  {PROG} inspect --list",
        f"  {PROG} inspect --subject 001_recording --show",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 2

    if argv[0] in ("-V", "--version"):
        print(f"{PROG} {__version__}")
        return 0

    name = argv[0]
    resolved = ALIASES.get(name, name)

    if resolved not in COMMANDS:
        print(f"{PROG}: unknown command '{name}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    module = importlib.import_module(COMMANDS[resolved].module)
    return module.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
