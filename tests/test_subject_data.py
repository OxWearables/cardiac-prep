"""Tests for locating and reshaping one participant's processed output.

The derived daily and 24-hour views matter here: the pipeline writes no such
files, so these are computed from the 10-second QC table rather than read.
All fixtures are synthetic - no participant data is involved.
"""

import numpy as np
import pandas as pd
import pytest

from edfproc.subject_data import (
    SubjectNotFoundError,
    daily_profile,
    daily_summary,
    list_subjects,
    load_subject_qc,
)


def _make_subject(output_dir, subject_id, hours=2, rr_ms=1000.0, start="2025-01-01 00:00:00"):
    """Write a minimal but realistic *_df_qc.csv.gz for one participant."""
    processed = output_dir / subject_id / "processed_data"
    processed.mkdir(parents=True)

    n = int(hours * 3600 / 10)
    frame = pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq="10s"),
            "RRm_imputed": np.full(n, rr_ms),
            "HRm_imputed": np.full(n, 60 * 1000 / rr_ms),
            "acc_imputed": np.full(n, 5.0),
        }
    )
    path = processed / f"{subject_id}.EDF_df_qc.csv.gz"
    frame.to_csv(path, compression="gzip", index=False)
    return path


# list_subjects

def test_lists_subjects_with_results(tmp_path):
    _make_subject(tmp_path, "bbb")
    _make_subject(tmp_path, "aaa")

    assert list_subjects(tmp_path) == ["aaa", "bbb"]  # sorted


def test_ignores_folders_without_a_qc_file(tmp_path):
    """A failed run leaves a folder behind; it must not appear as available."""
    _make_subject(tmp_path, "good")
    (tmp_path / "failed" / "processed_data").mkdir(parents=True)

    assert list_subjects(tmp_path) == ["good"]


def test_missing_output_directory_lists_nothing(tmp_path):
    assert list_subjects(tmp_path / "nope") == []


# load_subject_qc

def test_loads_qc_table_indexed_by_time(tmp_path):
    _make_subject(tmp_path, "s1", hours=1)

    df = load_subject_qc(tmp_path, "s1")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 360  # one hour at 10-second resolution
    assert "HRm_imputed" in df.columns


def test_unknown_subject_lists_the_available_ones(tmp_path):
    """A mistyped id is the likely cause, so the error must show the choices."""
    _make_subject(tmp_path, "real_one")

    with pytest.raises(SubjectNotFoundError) as excinfo:
        load_subject_qc(tmp_path, "typo")

    message = str(excinfo.value)
    assert "typo" in message
    assert "real_one" in message


def test_empty_output_directory_suggests_running_the_pipeline(tmp_path):
    with pytest.raises(SubjectNotFoundError, match="process.py"):
        load_subject_qc(tmp_path, "anything")


def test_subject_folder_without_qc_file_is_reported(tmp_path):
    (tmp_path / "half_done" / "processed_data").mkdir(parents=True)

    with pytest.raises(SubjectNotFoundError, match="processing failed"):
        load_subject_qc(tmp_path, "half_done")


# daily_summary

def test_daily_summary_converts_rr_to_heart_rate(tmp_path):
    """1000 ms RR is 60 bpm, and one row per calendar day."""
    _make_subject(tmp_path, "s1", hours=48, rr_ms=1000.0)
    df = load_subject_qc(tmp_path, "s1")

    daily = daily_summary(df)

    assert len(daily) == 2  # spans two calendar days
    assert daily["HRm_imputed"].iloc[0] == pytest.approx(60.0)


def test_daily_summary_requires_the_rr_column():
    with pytest.raises(KeyError, match="RRm_imputed"):
        daily_summary(pd.DataFrame(index=pd.DatetimeIndex([])))


# daily_profile

def test_daily_profile_has_one_row_per_minute_of_day(tmp_path):
    _make_subject(tmp_path, "s1", hours=24, rr_ms=1200.0)
    df = load_subject_qc(tmp_path, "s1")

    profile = daily_profile(df)

    assert len(profile) == 24 * 60
    assert profile["HRm_median"].iloc[0] == pytest.approx(50.0)  # 60000 / 1200
    assert profile["time_of_day"].iloc[0] == "00:00"
    assert profile["time_of_day"].iloc[-1] == "23:59"


def test_daily_profile_pools_across_days(tmp_path):
    """Two days of recording still collapse to a single 24-hour profile."""
    _make_subject(tmp_path, "s1", hours=48, rr_ms=1000.0)
    df = load_subject_qc(tmp_path, "s1")

    assert len(daily_profile(df)) == 24 * 60


def test_daily_profile_requires_the_rr_column():
    with pytest.raises(KeyError, match="RRm_imputed"):
        daily_profile(pd.DataFrame(index=pd.DatetimeIndex([])))
