"""Logging setup shared by the command line and by worker processes.

Worker processes are started with the 'spawn' method on macOS and Windows,
which means they do not inherit logging configuration from the parent. Each
worker therefore calls configure_logging() again in its initialiser.
"""

import logging
import sys

LOGGER_NAME = "edfproc"


def get_logger(name=None):
    """Return the package logger, or a named child of it."""
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(verbose=False, include_process=False):
    """Send package log records to stderr.

    Args:
        verbose: Include DEBUG records as well as INFO and above.
        include_process: Prefix each line with the worker process name. Useful
            when several recordings are processed at once and their output
            would otherwise interleave unattributably.
    """
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Reconfiguring in a worker must not stack duplicate handlers.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    if include_process:
        fmt = "%(asctime)s [%(processName)s] %(levelname)s  %(message)s"
    else:
        fmt = "%(asctime)s %(levelname)s  %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)

    # Records are emitted by our handler only; without this they would also
    # propagate to the root logger and appear twice if anything configures it.
    logger.propagate = False

    return logger
