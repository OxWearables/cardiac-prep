"""Tests for all-or-nothing result writing.

The property that matters is what happens when a write is interrupted: the
destination must be either absent or complete, never truncated, and a previous
good version must survive. A half-written .csv.gz is indistinguishable from a
finished one except by its gzip checksum, which is a bad thing to discover
weeks later.
"""

import gzip

import pandas as pd
import pytest

from edfproc.io_utils import atomic_write_csv


class ExplodingFrame:
    """Writes some bytes, then fails, imitating an interrupted write."""

    def __init__(self, error=RuntimeError("disk full")):
        self.error = error
        self.attempted_path = None

    def to_csv(self, path, **kwargs):
        self.attempted_path = path
        with open(path, "wb") as handle:
            handle.write(b"partial data\n")
        raise self.error


def _frame():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# Success path

def test_writes_readable_gzip(tmp_path):
    path = atomic_write_csv(_frame(), tmp_path / "out.csv.gz", index=False)

    assert path.is_file()
    # gzip.open validates the CRC on read, so this fails on a truncated file.
    with gzip.open(path, "rt") as handle:
        assert handle.read().splitlines()[0] == "a,b"
    pd.testing.assert_frame_equal(pd.read_csv(path), _frame())


def test_infers_gzip_from_the_suffix(tmp_path):
    """Compression must not be lost just because the temp file ends in .tmp."""
    path = atomic_write_csv(_frame(), tmp_path / "out.csv.gz", index=False)

    with open(path, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b", "not gzip-compressed"


def test_explicit_compression_is_respected(tmp_path):
    path = atomic_write_csv(_frame(), tmp_path / "plain.csv", compression=None, index=False)

    with open(path, "rb") as handle:
        assert handle.read(2) != b"\x1f\x8b"


def test_leaves_no_temporary_file_behind(tmp_path):
    atomic_write_csv(_frame(), tmp_path / "out.csv.gz", index=False)

    assert list(tmp_path.glob("*.tmp")) == []


def test_creates_missing_parent_directories(tmp_path):
    path = atomic_write_csv(_frame(), tmp_path / "a" / "b" / "out.csv.gz", index=False)

    assert path.is_file()


# Interrupted writes

def test_failed_write_leaves_no_destination(tmp_path):
    target = tmp_path / "out.csv.gz"

    with pytest.raises(RuntimeError, match="disk full"):
        atomic_write_csv(ExplodingFrame(), target)

    assert not target.exists(), "a partial file was left at the destination"
    assert list(tmp_path.glob("*.tmp")) == [], "temporary file not cleaned up"


def test_failed_write_preserves_the_previous_version(tmp_path):
    """The whole point: an interrupted rerun must not destroy good results."""
    target = tmp_path / "out.csv.gz"
    atomic_write_csv(_frame(), target, index=False)
    original = target.read_bytes()

    with pytest.raises(RuntimeError):
        atomic_write_csv(ExplodingFrame(), target)

    assert target.read_bytes() == original
    pd.testing.assert_frame_equal(pd.read_csv(target), _frame())


def test_keyboard_interrupt_also_cleans_up(tmp_path):
    """Ctrl-C is the likeliest interruption, and it is not an Exception."""
    target = tmp_path / "out.csv.gz"

    with pytest.raises(KeyboardInterrupt):
        atomic_write_csv(ExplodingFrame(KeyboardInterrupt()), target)

    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_temp_file_sits_beside_the_destination(tmp_path):
    """os.replace is only atomic within a filesystem, so the temp must be local."""
    target = tmp_path / "nested" / "out.csv.gz"
    frame = ExplodingFrame()

    with pytest.raises(RuntimeError):
        atomic_write_csv(frame, target)

    assert frame.attempted_path.parent == target.parent


def test_overwrites_an_existing_file_on_success(tmp_path):
    target = tmp_path / "out.csv.gz"
    atomic_write_csv(pd.DataFrame({"a": [1]}), target, index=False)

    atomic_write_csv(pd.DataFrame({"a": [9, 9]}), target, index=False)

    assert pd.read_csv(target)["a"].tolist() == [9, 9]
