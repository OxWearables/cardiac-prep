"""Tests for the metric aggregation layer in proc_edf.py.

These cover the summary statistics that end up in the participant PDF reports
and in df_info_summary.csv.gz.
All fixtures are synthetic - no participant data is involved.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from cardiacprep.config import Config
from cardiacprep.proc_edf import _is_within_rest_window, calculate_summary_metrics
from cardiacprep.read_utils import mean_amplitude_deviation

CFG = Config()


# mean_amplitude_deviation
#
# These moved here from a compute_mad() that took three axes and was never
# called. MAD now takes the vector magnitude, because readACC has already
# computed it from the calibrated signal.

def _vm(ax, ay, az):
    return np.sqrt(np.asarray(ax) ** 2 + np.asarray(ay) ** 2 + np.asarray(az) ** 2)


def test_mad_is_zero_for_static_signal():
    """A device held perfectly still has no amplitude deviation."""
    n = 400
    vm = _vm(np.full(n, 3.0), np.full(n, 4.0), np.zeros(n))  # constant 5.0

    mad = mean_amplitude_deviation(vm, epoch_samples=100)

    assert mad.shape == (4,)
    assert np.allclose(mad, 0.0)


def test_mad_known_value():
    """Hand-computable: magnitudes [0,0,6,6] deviate from their mean of 3 by 3."""
    mad = mean_amplitude_deviation(np.array([0.0, 0.0, 6.0, 6.0]), epoch_samples=4)

    assert mad == pytest.approx([3.0])


def test_mad_removes_a_constant_offset():
    """Gravity is a constant, and MAD subtracts the epoch mean, so it cancels.

    This is why no separate detrending step is needed before MAD.
    """
    movement = np.array([0.0, 0.0, 6.0, 6.0])

    assert mean_amplitude_deviation(movement, 4) == pytest.approx(
        mean_amplitude_deviation(movement + 1000.0, 4)
    )


def test_mad_measures_a_trailing_partial_epoch():
    """A recording that is not a whole number of epochs keeps its final minutes."""
    vm = np.ones(250)

    mad = mean_amplitude_deviation(vm, epoch_samples=100)

    assert mad.shape == (3,)  # two whole epochs plus the final 50 samples


def test_mad_handles_input_shorter_than_one_epoch():
    mad = mean_amplitude_deviation(np.ones(10), epoch_samples=100)

    assert mad.shape == (1,)
    assert mad[0] == pytest.approx(0.0)


def test_mad_is_invariant_to_axis_permutation():
    """MAD comes from the vector magnitude, so axis order cannot matter."""
    rng = np.random.default_rng(2)
    ax, ay, az = rng.normal(size=(3, 300))

    assert np.allclose(
        mean_amplitude_deviation(_vm(ax, ay, az), 100),
        mean_amplitude_deviation(_vm(az, ax, ay), 100),
    )


def test_mad_rejects_a_meaningless_epoch():
    with pytest.raises(ValueError, match="at least 1"):
        mean_amplitude_deviation(np.ones(10), epoch_samples=0)


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
    assert calculate_summary_metrics(pd.DataFrame(), CFG) == {}


def test_calculate_summary_metrics_converts_rr_to_heart_rate():
    """A 1000 ms RR interval is exactly 60 bpm."""
    df = _qc_frame("2025-01-01 22:00:00", hours=2, rr_ms=1000, acc_mg=5.0, rmssd_ms=30.0)

    summary = calculate_summary_metrics(df, CFG)

    assert summary["HR_min"] == pytest.approx(60.0)
    assert summary["HR_max"] == pytest.approx(60.0)
    assert summary["HR_mean"] == pytest.approx(60.0)


def test_calculate_summary_metrics_finds_resting_period_at_night():
    """Night-time samples below the movement threshold define the resting window."""
    df = _qc_frame("2025-01-01 22:00:00", hours=4, rr_ms=1200, acc_mg=2.0, rmssd_ms=45.0)

    summary = calculate_summary_metrics(df, CFG)

    assert summary["HR_rest_robust"] == pytest.approx(50.0)  # 60000 / 1200
    assert summary["median_daily_rmssd"] == pytest.approx(45.0)


def test_calculate_summary_metrics_excludes_active_periods_from_rest():
    """Night-time movement above the threshold must not count as rest."""
    df = _qc_frame("2025-01-01 22:00:00", hours=4, rr_ms=1200, acc_mg=50.0, rmssd_ms=45.0)

    summary = calculate_summary_metrics(df, CFG)

    assert np.isnan(summary["HR_rest_robust"])
    assert np.isnan(summary["median_daily_rmssd"])


def test_calculate_summary_metrics_excludes_daytime_from_rest():
    """Sitting still at midday is quiet but is not the resting window.

    This is the assumption the fixed 21:00-09:00 window bakes in, and is
    exactly what automatic sleep-window detection would replace.
    """
    df = _qc_frame("2025-01-01 11:00:00", hours=4, rr_ms=1200, acc_mg=1.0, rmssd_ms=45.0)

    summary = calculate_summary_metrics(df, CFG)

    assert np.isnan(summary["HR_rest_robust"])


def test_rest_window_is_configurable_for_shifted_sleepers():
    """Sleep falling outside the fixed window is missed until the window moves.

    A participant whose only quiet period is 09:30-11:00 is invisible to the
    default 21:00-09:00 window - their resting HR comes back as NaN despite
    the data being present. Widening the window in config recovers it with no
    code change. This is the concrete cost of a fixed clock window.
    """
    df = _qc_frame("2025-01-01 09:30:00", hours=1.5, rr_ms=1200, acc_mg=1.0, rmssd_ms=45.0)

    missed = calculate_summary_metrics(df, CFG)
    assert np.isnan(missed["HR_rest_robust"])

    shifted = dataclasses.replace(CFG, night_start_hour=1, night_end_hour=12)
    found = calculate_summary_metrics(df, shifted)
    assert found["HR_rest_robust"] == pytest.approx(50.0)


def test_default_window_truncates_a_late_sleeper():
    """Sleep spanning the 09:00 boundary is only partly counted by default.

    Someone sleeping 06:00-11:00 has the 09:00-11:00 portion silently dropped,
    so resting metrics are computed from a subset of their actual rest.
    """
    df = _qc_frame("2025-01-01 06:00:00", hours=5, rr_ms=1200, acc_mg=1.0, rmssd_ms=45.0)

    df_10min = df.resample("10min", on="time").mean()

    in_default = _is_within_rest_window(df_10min.index, CFG).sum()
    in_shifted = _is_within_rest_window(
        df_10min.index, dataclasses.replace(CFG, night_start_hour=1, night_end_hour=12)
    ).sum()

    assert in_default == 18   # only 06:00-09:00 of the five hours
    assert in_shifted == 30   # the whole recording


def test_rest_window_supports_non_wrapping_hours():
    """A window that does not cross midnight selects a single daytime block."""
    df = _qc_frame("2025-01-01 13:00:00", hours=2, rr_ms=1200, acc_mg=1.0, rmssd_ms=45.0)

    daytime = dataclasses.replace(CFG, night_start_hour=12, night_end_hour=16)
    summary = calculate_summary_metrics(df, daytime)

    assert summary["HR_rest_robust"] == pytest.approx(50.0)


# Activity thresholds

def test_activity_thresholds_are_strictly_increasing():
    """The zone boundaries must be ordered or the banding logic overlaps."""
    thresholds = CFG.activity_thresholds
    assert thresholds["very_light"] < thresholds["light"] < thresholds["moderate"]


def test_activity_thresholds_match_published_values():
    """Guards against accidental edits to the Etzkorn et al. (2024) cut-points."""
    assert CFG.activity_thresholds == {
        "very_light": 9.04,
        "light": 28.19,
        "moderate": 58.08,
    }
