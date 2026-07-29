"""Tests for signal preparation in read_utils.py.

prepSig decides which 10-second segments are usable at all, so its wear
detection and clipping logic gates everything downstream.
All fixtures are synthetic - no participant data is involved.
"""

import numpy as np
import pytest

from edfproc.read_utils import prepSig

FS = 250
NSEG = 2500  # 10 seconds at 250 Hz


def _sine(n_samples, freq_hz=1.2, amplitude=1.0, fs=FS):
    """A crude stand-in for a periodic physiological signal."""
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def test_prepSig_reshapes_into_segments():
    ecg = _sine(NSEG * 3)

    out, _, _, _ = prepSig(ecg, nseg=NSEG, fs=FS)

    assert out.shape == (3, NSEG)


def test_prepSig_pads_partial_final_segment():
    """A recording that does not divide evenly is edge-padded up to a full segment."""
    ecg = _sine(NSEG + 500)  # 1.2 segments

    out, i_device_worn, _, _ = prepSig(ecg, nseg=NSEG, fs=FS)

    assert out.shape == (2, NSEG)
    assert len(i_device_worn) == 2


def test_prepSig_flags_flatline_as_not_worn():
    """A constant trace has zero variance, which is how non-wear is detected."""
    ecg = np.zeros(NSEG * 2)

    _, i_device_worn, _, ix_qc = prepSig(ecg, nseg=NSEG, fs=FS)

    assert not i_device_worn.any()
    assert not ix_qc.any()  # flatline cannot pass the variance/amplitude checks


def test_prepSig_flags_real_signal_as_worn():
    ecg = _sine(NSEG * 2)

    _, i_device_worn, _, _ = prepSig(ecg, nseg=NSEG, fs=FS)

    assert i_device_worn.all()


def test_prepSig_detects_clipping():
    """A segment saturated beyond clip_val fails the <5% clipped-sample rule."""
    clip_val = 4.0
    ecg = np.concatenate(
        [
            _sine(NSEG, amplitude=1.0),                 # clean segment
            np.full(NSEG, clip_val + 10.0),             # fully saturated segment
        ]
    )

    _, _, ix_non_clipped, _ = prepSig(ecg, nseg=NSEG, fs=FS, clip_val=clip_val)

    assert ix_non_clipped[0]
    assert not ix_non_clipped[1]


def test_prepSig_returns_one_flag_per_segment():
    """All four return values must stay aligned with the segment axis."""
    ecg = _sine(NSEG * 4)

    out, i_device_worn, ix_non_clipped, ix_qc = prepSig(ecg, nseg=NSEG, fs=FS)

    n_segments = out.shape[0]
    assert len(i_device_worn) == n_segments
    assert len(ix_non_clipped) == n_segments
    assert len(ix_qc) == n_segments


def test_prepSig_attenuates_mains_interference():
    """The 50 Hz notch filter should suppress mains hum far more than the signal band.

    The comparison signal sits at 10 Hz, inside the 2-40 Hz bandpass, so it
    survives filtering while the 50 Hz hum is removed by both the notch and the
    bandpass.
    """
    clean = _sine(NSEG, freq_hz=10.0)
    hum = _sine(NSEG, freq_hz=50.0, amplitude=1.0)

    out_clean, _, _, _ = prepSig(clean.copy(), nseg=NSEG, fs=FS)
    out_hum, _, _, _ = prepSig(hum.copy(), nseg=NSEG, fs=FS)

    # Ignore filter edge transients when comparing residual power.
    power_clean = np.var(out_clean[0][FS:-FS])
    power_hum = np.var(out_hum[0][FS:-FS])

    assert power_hum < power_clean
    assert power_hum == pytest.approx(0.0, abs=1e-3)
