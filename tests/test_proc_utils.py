"""Tests for the numerical core in proc_utils.py.

These cover the RR-interval / QRS maths that downstream HR and HRV figures are
built on, so a refactor that silently changes a result gets caught here.
All fixtures are synthetic - no participant data is involved.
"""

import numpy as np
import pandas as pd
import pytest

from cardiacprep.proc_utils import doImp, downsampleECG, getQCmetrics, getQRS, getSNR

# getSNR

def test_getSNR_known_values():
    """Hand-computable case: rows [2,-2] and [0,0] give a 0 dB SNR.

    After constant detrending both rows keep their values (each already has
    zero mean). The median template is [1,-1], so signal power is 1 and the
    residual power is also 1, i.e. 10*log10(1) = 0 dB.
    """
    X = np.array([[2.0, -2.0], [0.0, 0.0]])

    snr, amp = getSNR(X)

    assert snr == pytest.approx(0.0)
    assert amp == pytest.approx(2.0)  # peak-to-peak of the [1,-1] template


def test_getSNR_detrends_constant_offset():
    """A DC offset must not change the SNR - detrending should remove it."""
    X = np.array([[2.0, -2.0], [0.0, 0.0]])

    snr_plain, amp_plain = getSNR(X)
    snr_offset, amp_offset = getSNR(X + 100.0)

    assert snr_offset == pytest.approx(snr_plain)
    assert amp_offset == pytest.approx(amp_plain)


# getQCmetrics

def test_getQCmetrics_regular_rhythm():
    """A perfectly regular rhythm has zero RR variability."""
    fs, nseg = 250, 2500
    rw = np.arange(0, 2500, 250)  # 10 R-peaks, 250 samples apart

    rng = np.random.default_rng(0)
    ecg = rng.normal(0.0, 0.01, size=nseg)
    ecg[rw] = 5.0  # QRS spikes so the SNR calculation is well conditioned

    n_rr, rrM, rrC, rrsd, rr_outliers, snr, amp, rmssd = getQCmetrics(
        ecg, rw, nseg=nseg, rr_lim=[50, 1250], fs=fs
    )

    assert n_rr == 9              # 10 peaks -> 9 intervals
    assert rrM == pytest.approx(250.0)
    assert rrC == pytest.approx(9 * 250 / nseg)
    assert rrsd == pytest.approx(0.0)
    assert rmssd == pytest.approx(0.0)   # no beat-to-beat variation
    assert rr_outliers == 0
    assert np.isfinite(snr)
    assert amp > 0


def test_getQCmetrics_filters_non_physiological_intervals():
    """Intervals outside rr_lim are dropped before any metric is computed."""
    # Gaps of 100, 30 and 250 samples; 30 is below the lower limit of 50.
    rw = np.array([0, 100, 130, 380])
    ecg = np.zeros(2500)
    ecg[rw] = 1.0

    n_rr, rrM, *_ = getQCmetrics(ecg, rw, nseg=2500, rr_lim=[50, 1250], fs=250)

    assert n_rr == 2                       # the 30-sample interval is excluded
    assert rrM == pytest.approx(175.0)     # median of [100, 250]


def test_getQCmetrics_rmssd_is_nan_for_single_interval():
    """RMSSD needs at least two intervals to have any meaning."""
    rw = np.array([0, 250])
    ecg = np.zeros(2500)
    ecg[rw] = 1.0

    *_, rmssd = getQCmetrics(ecg, rw, nseg=2500, rr_lim=[50, 1250], fs=250)

    assert np.isnan(rmssd)


def test_getQCmetrics_counts_outliers():
    """Intervals longer than 1.8x the median count as outliers (missed beats)."""
    # Four 200-sample intervals then one 600-sample gap: 600 > 1.8 * 200.
    rw = np.array([0, 200, 400, 600, 800, 1400])
    ecg = np.zeros(2500)
    ecg[rw] = 1.0

    _, rrM, _, _, rr_outliers, *_ = getQCmetrics(
        ecg, rw, nseg=2500, rr_lim=[50, 1250], fs=250
    )

    assert rrM == pytest.approx(200.0)
    assert rr_outliers == 1


# getQRS

def test_getQRS_empty_mask_returns_empty_frame():
    """No detected QRS complexes must yield an empty frame, not an exception."""
    ecg = np.zeros(2500)
    mask = np.zeros(250, dtype=bool)

    result = getQRS(ecg=ecg, mask=mask)

    assert result.empty
    assert "t_rw" in result.columns


def test_getQRS_locates_positive_peak():
    """A single positive spike is reported at its exact sample index.

    The mask is at 1/10th the ECG sample rate, so mask indices 10:13 select
    ECG samples 100:130. The spike at 115 dominates, so the positive-polarity
    branch is taken and t_rw is the argmax position.
    """
    ecg = np.zeros(2500)
    ecg[115] = 5.0

    mask = np.zeros(250, dtype=bool)
    mask[10:13] = True

    result = getQRS(ecg=ecg, mask=mask)

    assert len(result) == 1
    assert result["t_rw"].iloc[0] == 115
    assert result.index[0] == 0  # falls in the first 10-second bin


def test_getQRS_locates_negative_peak():
    """Inverted QRS complexes are found via the negative-polarity branch."""
    ecg = np.zeros(2500)
    ecg[115] = -5.0

    mask = np.zeros(250, dtype=bool)
    mask[10:13] = True

    result = getQRS(ecg=ecg, mask=mask)

    assert len(result) == 1
    assert result["t_rw"].iloc[0] == 115


# downsampleECG

def test_downsampleECG_shape_and_standardisation():
    """250 Hz halves to 125 Hz, and each segment is standardised independently."""
    rng = np.random.default_rng(1)
    ecg = rng.normal(0.0, 1.0, size=(4, 2500))

    out = downsampleECG(ecg, fs_org=250, fs=125)

    assert out.shape == (4, 1250, 1)
    # StandardScaler is applied per segment, so each has ~zero mean, ~unit std.
    assert np.allclose(out.mean(axis=1).squeeze(), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=1).squeeze(), 1.0, atol=1e-6)


def test_downsampleECG_handles_all_zero_segments():
    """An all-zero (non-worn) segment must not raise via the clipping fallback."""
    ecg = np.zeros((2, 2500))

    out = downsampleECG(ecg, fs_org=250, fs=125)

    assert out.shape == (2, 1250, 1)
    assert np.all(np.isfinite(out))


# doImp

def _frame_with_gap(n=200, gap_at=None, col="RRm"):
    """Build a 10-second-resolution frame with an optional NaN gap."""
    values = np.arange(n, dtype=float)
    if gap_at is not None:
        values[gap_at] = np.nan
    return pd.DataFrame(
        {
            col: values,
            "time": pd.date_range("2025-01-01 00:00:00", periods=n, freq="10s"),
        }
    )


def test_doImp_renames_raw_column_and_flags_imputation():
    """The original column is preserved as *_raw alongside the imputed one."""
    df = _frame_with_gap(gap_at=50)

    out = doImp(df, "RRm")

    assert "RRm_raw" in out.columns
    assert "RRm_imputed" in out.columns
    assert "RRm_isImputed" in out.columns
    assert "RRm" not in out.columns


def test_doImp_interpolates_short_gap_linearly():
    """A lone missing sample surrounded by valid data is linearly filled."""
    df = _frame_with_gap(gap_at=50)

    out = doImp(df, "RRm")

    assert out["RRm_imputed"].iloc[50] == pytest.approx(50.0)  # mean of 49 and 51
    assert out["RRm_isImputed"].iloc[50]
    assert not out["RRm_isImputed"].iloc[49]


def test_doImp_leaves_complete_series_untouched():
    """With nothing missing, imputed values must equal the raw values exactly."""
    df = _frame_with_gap(gap_at=None)

    out = doImp(df, "RRm")

    pd.testing.assert_series_equal(
        out["RRm_imputed"], out["RRm_raw"], check_names=False
    )
    assert not out["RRm_isImputed"].any()
