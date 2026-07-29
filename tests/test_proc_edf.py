"""Tests for the metric aggregation layer in proc_edf.py.

These cover the summary statistics that end up in the participant PDF reports
and in df_info_summary.csv.gz.
All fixtures are synthetic - no participant data is involved.
"""

import numpy as np
import pandas as pd
import pytest

from proc_edf import ACTIVITY_THRESHOLDS, calculate_summary_metrics, compute_mad


# compute_mad

def test_compute_mad_is_zero_for_static_signal():
    """A device held perfectly still has no amplitude deviation."""
    n = 400
    ax = np.full(n, 3.0)
    ay = np.full(n, 4.0)
    az = np.zeros(n)  # vector magnitude is a constant 5.0

    mad = compute_mad(ax, ay, az, epoch_samples=100)

    assert mad.shape == (4,)
    assert np.allclose(mad, 0.0)


def test_compute_mad_known_value():
    """Hand-computable case: magnitudes [0,0,6,6] deviate from their mean of 3 by 3."""
    ax = np.array([0.0, 0.0, 6.0, 6.0])
    ay = np.zeros(4)
    az = np.zeros(4)

    mad = compute_mad(ax, ay, az, epoch_samples=4)

    assert mad == pytest.approx([3.0])


def test_compute_mad_discards_incomplete_trailing_epoch():
    """Only whole epochs are returned; a partial tail is dropped."""
    ax = np.ones(250)
    ay = np.zeros(250)
    az = np.zeros(250)

    mad = compute_mad(ax, ay, az, epoch_samples=100)

    assert mad.shape == (2,)  # 250 // 100, the final 50 samples are ignored


def test_compute_mad_handles_input_shorter_than_one_epoch():
    ax = np.ones(10)

    mad = compute_mad(ax, np.zeros(10), np.zeros(10), epoch_samples=100)

    assert mad.shape == (0,)


def test_compute_mad_is_invariant_to_axis_permutation():
    """MAD is computed from the vector magnitude, so axis order cannot matter."""
    rng = np.random.default_rng(2)
    ax, ay, az = rng.normal(size=(3, 300))

    assert np.allclose(
        compute_mad(ax, ay, az, 100),
        compute_mad(az, ax, ay, 100),
    )


# calculate_summary_metrics

def _qc_frame(start, hours, rr_ms, acc_mg, rmssd_ms):
    """Build a 10-second-resolution QC frame with constant values."""
    n = int(hours * 3600 / 10)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq="10s"),
            "RRm_imputed": np.full(n, float(rr_ms)),
            "acc_imputed": np.full(n, float(acc_mg)),
            "rmssd": np.full(n, float(rmssd_ms)),
        }
    )


def test_calculate_summary_metrics_empty_input():
    assert calculate_summary_metrics(pd.DataFrame(), sleep_thrs=9.04) == {}


def test_calculate_summary_metrics_converts_rr_to_heart_rate():
    """A 1000 ms RR interval is exactly 60 bpm."""
    df = _qc_frame("2025-01-01 22:00:00", hours=2, rr_ms=1000, acc_mg=5.0, rmssd_ms=30.0)

    summary = calculate_summary_metrics(df, sleep_thrs=9.04)

    assert summary["HR_min"] == pytest.approx(60.0)
    assert summary["HR_max"] == pytest.approx(60.0)
    assert summary["HR_mean"] == pytest.approx(60.0)


def test_calculate_summary_metrics_finds_resting_period_at_night():
    """Night-time samples below the movement threshold define the resting window."""
    df = _qc_frame("2025-01-01 22:00:00", hours=4, rr_ms=1200, acc_mg=2.0, rmssd_ms=45.0)

    summary = calculate_summary_metrics(df, sleep_thrs=9.04)

    assert summary["HR_rest_robust"] == pytest.approx(50.0)  # 60000 / 1200
    assert summary["median_daily_rmssd"] == pytest.approx(45.0)


def test_calculate_summary_metrics_excludes_active_periods_from_rest():
    """Night-time movement above the threshold must not count as rest."""
    df = _qc_frame("2025-01-01 22:00:00", hours=4, rr_ms=1200, acc_mg=50.0, rmssd_ms=45.0)

    summary = calculate_summary_metrics(df, sleep_thrs=9.04)

    assert np.isnan(summary["HR_rest_robust"])
    assert np.isnan(summary["median_daily_rmssd"])


def test_calculate_summary_metrics_excludes_daytime_from_rest():
    """Sitting still at midday is quiet but is not the resting window.

    This is the assumption the current fixed 21:00-09:00 window bakes in, and
    is exactly what automatic sleep-window detection would replace.
    """
    df = _qc_frame("2025-01-01 11:00:00", hours=4, rr_ms=1200, acc_mg=1.0, rmssd_ms=45.0)

    summary = calculate_summary_metrics(df, sleep_thrs=9.04)

    assert np.isnan(summary["HR_rest_robust"])


# Activity thresholds

def test_activity_thresholds_are_strictly_increasing():
    """The zone boundaries must be ordered or the banding logic overlaps."""
    assert (
        ACTIVITY_THRESHOLDS["very_light"]
        < ACTIVITY_THRESHOLDS["light"]
        < ACTIVITY_THRESHOLDS["moderate"]
    )


def test_activity_thresholds_match_published_values():
    """Guards against accidental edits to the Etzkorn et al. (2024) cut-points."""
    assert ACTIVITY_THRESHOLDS == {
        "very_light": 9.04,
        "light": 28.19,
        "moderate": 58.08,
    }
