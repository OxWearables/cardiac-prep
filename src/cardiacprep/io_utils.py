"""Writing result files so an interrupted run cannot leave partial output.

A plain ``to_csv`` writes straight to its destination, so a Ctrl-C, an
out-of-memory kill or a machine going to sleep mid-write leaves a truncated
file that still looks like a finished result. Weeks later there is nothing to
distinguish it from a complete one except a gzip CRC failure.

Writing to a temporary file and renaming it into place removes that state: the
destination either does not exist or is complete.
"""

import os
from pathlib import Path

from .logging_utils import get_logger

# pandas infers compression from the file extension. The temporary file ends in
# .tmp, so the intended compression has to be stated explicitly or the data
# would be written uncompressed under a .gz name.
_COMPRESSION_BY_SUFFIX = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".zst": "zstd",
    ".zip": "zip",
}

log = get_logger("io")


def atomic_write_csv(frame, path, **to_csv_kwargs):
    """Write a DataFrame to ``path`` as a single all-or-nothing step.

    The data goes to a temporary file alongside the destination, then
    ``os.replace`` moves it into place. That rename is atomic within a
    filesystem, so a reader never sees a half-written file, and an interrupted
    run leaves the previous version untouched rather than a corrupt one.

    Compression is inferred from the destination suffix when not given, since
    the temporary name would otherwise defeat pandas' own inference.

    Args:
        frame: Anything with a ``to_csv(path, **kwargs)`` method.
        path: Final destination. Parent directories are created if needed.
        **to_csv_kwargs: Passed through to ``to_csv``.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "compression" not in to_csv_kwargs:
        inferred = _COMPRESSION_BY_SUFFIX.get(path.suffix.lower())
        if inferred is not None:
            to_csv_kwargs["compression"] = inferred

    # The PID keeps concurrent writers from sharing a temporary name, and the
    # temporary sits in the destination directory so the rename stays within
    # one filesystem. os.replace is only atomic within a filesystem.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")

    try:
        frame.to_csv(tmp, **to_csv_kwargs)
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt is one of the most
        # likely ways a long run gets cut short, and it must not leave the
        # partial file behind either.
        try:
            tmp.unlink(missing_ok=True)
        except OSError as cleanup_error:  # pragma: no cover - unusual
            log.debug("Could not remove partial file %s: %s", tmp, cleanup_error)
        raise

    return path
