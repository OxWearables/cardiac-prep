"""Logging setup shared by the command line and by worker processes.

Worker processes are started with the 'spawn' method on macOS and Windows,
which means they do not inherit logging configuration from the parent. Each
worker therefore calls configure_logging() again in its initialiser.
"""

import logging
import sys

LOGGER_NAME = "cardiacprep"


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
    if include_process:
        fmt = "%(asctime)s [%(processName)s] %(levelname)s  %(message)s"
    else:
        fmt = "%(asctime)s %(levelname)s  %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    # Take ownership of the root logger. Some dependencies call
    # logging.basicConfig() at import time, which attaches their own handler
    # and drops the root level to INFO, exposing every library's INFO chatter
    # (actipy does this, which is why matplotlib's notice about plotting
    # date-like strings as categories used to appear once per file).
    #
    # Claiming root serves two purposes: it removes any handler already
    # installed that way, and it makes a later basicConfig() a no-op, since
    # basicConfig does nothing when the root logger already has handlers.
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    # Third-party INFO stays hidden. Their loggers inherit this level, so the
    # records are never even created. Warnings and errors still get through.
    root.setLevel(logging.WARNING)

    # Our own records are gated by our logger's level, not root's, and reach
    # the handler above by propagation. Ancestor levels are not re-checked
    # during propagation, so INFO from us is emitted while INFO from
    # dependencies is not.
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = True

    # Reconfiguring in a worker must not stack duplicate handlers.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    return logger


def quieten_dependency(name, level=logging.WARNING):
    """Raise the log level of a noisy dependency by name.

    Rarely needed, since configure_logging already hides third-party INFO, but
    useful if a dependency logs warnings that are not actionable.
    """
    logging.getLogger(name).setLevel(level)
