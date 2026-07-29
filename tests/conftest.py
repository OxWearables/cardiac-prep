"""Shared fixtures for the end-to-end pipeline tests.

Builds synthetic EDF recordings with a known heart rate and a known wear
pattern, so the whole pipeline can be exercised without any participant data
and without the real QRS detector weights.

The detector is replaced by a stub. This is possible because procECG takes the
model as an argument, so the orchestration around it can be tested on a
machine (or in CI) that has neither TensorFlow weights nor a GPU.
"""

import warnings
from datetime import datetime

import numpy as np
import pyedflib
import pytest

FS_ECG = 250
FS_ACC = 100

# The QRS mask the detector emits has 250 values per 10-second segment, i.e.
# 25 Hz. At 60 bpm a beat therefore falls every 25 mask samples.
MASK_SAMPLES_PER_SEGMENT = 250


def synth_ecg(n_samples, fs=FS_ECG, bpm=60, seed=0):
    """A spiky periodic trace with R peaks at exact, known sample positions."""
    rng = np.random.default_rng(seed)
    ecg = rng.normal(0.0, 20.0, n_samples)  # microvolt-scale baseline noise
    period = int(round(fs * 60 / bpm))
    peaks = np.arange(0, n_samples, period)
    ecg[peaks] = 1500.0
    trough = peaks + 2
    ecg[trough[trough < n_samples]] = -400.0
    return ecg, peaks


def write_synthetic_edf(
    path,
    hours=1.0,
    bpm=60,
    fs_ecg=FS_ECG,
    fs_acc=FS_ACC,
    start=datetime(2025, 1, 6, 21, 0, 0),
    worn=True,
    acc_mg=0.0,
):
    """Write an EDF with one ECG channel and three accelerometer channels.

    Args:
        hours: Recording duration.
        bpm: Heart rate encoded in the ECG, so tests can assert on recovery.
        start: Recording start time. Defaults to 21:00 so the data falls
            inside the default overnight rest window.
        worn: If False the ECG is flat, which is how non-wear is detected.
        acc_mg: Extra movement amplitude in milli-g on top of gravity.
    """
    n_ecg = int(hours * 3600 * fs_ecg)
    n_acc = int(hours * 3600 * fs_acc)

    if worn:
        ecg, _ = synth_ecg(n_ecg, fs_ecg, bpm)
    else:
        ecg = np.zeros(n_ecg)

    rng = np.random.default_rng(1)
    # A flat, still device reads about 1 g on the z axis.
    ax = rng.normal(0.0, 8.0, n_acc)
    ay = rng.normal(0.0, 8.0, n_acc)
    az = rng.normal(1000.0, 8.0, n_acc)
    if acc_mg:
        wobble = acc_mg * np.sin(2 * np.pi * np.arange(n_acc) / fs_acc)
        ax = ax + wobble
        ay = ay + wobble

    channels = [
        ("ECG", "uV", fs_ecg, ecg),
        ("Accelerometer_X", "mg", fs_acc, ax),
        ("Accelerometer_Y", "mg", fs_acc, ay),
        ("Accelerometer_Z", "mg", fs_acc, az),
    ]

    writer = pyedflib.EdfWriter(
        str(path), len(channels), file_type=pyedflib.FILETYPE_EDFPLUS
    )
    try:
        writer.setStartdatetime(start)
        for i, (label, dim, fs, signal) in enumerate(channels):
            # EDF stores physical limits as 8-character strings, so round them
            # to avoid a precision warning on every write.
            writer.setSignalHeader(i, {
                "label": label,
                "dimension": dim,
                "sample_frequency": fs,
                "physical_min": float(np.floor(signal.min()) - 1),
                "physical_max": float(np.ceil(signal.max()) + 1),
                # A symmetric digital range makes the physical offset zero, so
                # an all-zero signal round-trips to exactly 0.0. With the usual
                # -32768..32767 it comes back as a tiny constant instead, which
                # defeats prepSig's np.std(x) > 0 non-wear test.
                "digital_min": -32767,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            writer.writeSamples([np.asarray(sig) for _, _, _, sig in channels])
    finally:
        writer.close()

    return path


class FakeQRSModel:
    """Stand-in for the trained QRS detector.

    Emits a beat every ``mask_period`` mask samples, which corresponds to a
    fixed heart rate. It ignores the input entirely: the point of these tests
    is the pipeline around the model, not the model itself.
    """

    def __init__(self, bpm=60):
        self.bpm = bpm
        # 250 mask values per 10 s segment => 25 mask samples per second.
        self.mask_period = int(round(25 * 60 / bpm))

    def predict(self, x, verbose=0):
        n_segments = x.shape[0]
        mask = np.zeros((n_segments, MASK_SAMPLES_PER_SEGMENT), dtype=float)
        mask[:, :: self.mask_period] = 1.0
        return mask[..., np.newaxis]


@pytest.fixture
def make_edf(tmp_path):
    """Factory writing synthetic EDF files into the test's temp directory."""
    counter = {"n": 0}

    def _make(name=None, **kwargs):
        if name is None:
            counter["n"] += 1
            name = f"synthetic_{counter['n']:03d}.EDF"
        return write_synthetic_edf(tmp_path / name, **kwargs)

    return _make


@pytest.fixture
def fake_model():
    return FakeQRSModel(bpm=60)


@pytest.fixture
def make_model():
    """Factory for stub detectors at a chosen heart rate."""
    return FakeQRSModel
