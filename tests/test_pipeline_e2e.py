"""End-to-end tests over synthetic EDF recordings.

These cover procECG and procEDF, the two functions that orchestrate a whole
recording. Everything is driven from a synthetic file with a known heart rate
and a stub detector, so a change that silently alters a reported metric shows
up here rather than in a manual diff of df_info_summary.csv.gz.

No participant data and no real model weights are involved.
"""

import dataclasses

import pandas as pd
import pyedflib
import pytest

from edfproc.config import Config
from edfproc.proc_edf import calculate_summary_metrics, procECG, procEDF
from edfproc.read_utils import readACC, readEDFECG_info

# One-hour chunks keep the synthetic files small enough to stay fast.
CFG = dataclasses.replace(Config(), chunk_hours=1)


def _process_one_chunk(edf_path, cfg, model, chunk=0):
    chunk_samples = int(cfg.fs_expected * cfg.chunk_seconds)
    with pyedflib.EdfReader(str(edf_path)) as f:
        return procECG(f, chunk, chunk_samples, edf_path.name, cfg, model)


# Reading

def test_reads_header_of_a_synthetic_recording(make_edf):
    path = make_edf(hours=1.0)

    fs, start_time, info = readEDFECG_info(str(path))

    assert fs == 250
    assert start_time.hour == 21
    assert info["N_ecg"].iloc[0] == 1 * 3600 * 250
    assert info["units_ecg"].iloc[0] == "uV"


def test_reads_accelerometer_into_ten_second_bins(make_edf):
    path = make_edf(hours=1.0)

    df_acc, info, _ = readACC(str(path), pd.Timestamp("2025-01-06 21:00:00"))

    assert len(df_acc) == 360  # one hour at 10-second resolution
    assert list(df_acc.columns) == ["acc", "acc_clipped"]
    assert info["CalibOK"].iloc[0] in (0, 1)


def test_still_device_reads_as_low_movement(make_edf):
    """A stationary recording must fall below the resting threshold."""
    path = make_edf(hours=1.0, acc_mg=0.0)

    df_acc, _, _ = readACC(str(path), pd.Timestamp("2025-01-06 21:00:00"))

    assert df_acc["acc"].median() < CFG.sleep_threshold_mg


# procECG

def test_procECG_recovers_the_encoded_heart_rate(make_edf, fake_model):
    """A 60 bpm recording must come back as ~1000 ms RR intervals."""
    path = make_edf(hours=1.0, bpm=60)

    df = _process_one_chunk(path, CFG, fake_model)

    assert len(df) == 360  # 10-second segments in one hour
    assert df["device_worn"].all()
    assert df["passed_finalQC"].mean() > 0.95

    median_rr = df.loc[df["passed_finalQC"], "RRm"].median()
    assert median_rr == pytest.approx(1000.0, abs=20.0)
    assert 60 * 1000 / median_rr == pytest.approx(60.0, abs=1.5)


def test_procECG_recovers_a_faster_heart_rate(make_edf, make_model):
    """The recovered rate must track the encoded one, not be a fixed constant."""
    path = make_edf(hours=1.0, bpm=75)

    df = _process_one_chunk(path, CFG, make_model(bpm=75))

    median_rr = df.loc[df["passed_finalQC"], "RRm"].median()
    assert 60 * 1000 / median_rr == pytest.approx(75.0, abs=2.0)


def test_procECG_flags_a_flat_recording_as_not_worn(make_edf, fake_model):
    path = make_edf(hours=1.0, worn=False)

    df = _process_one_chunk(path, CFG, fake_model)

    assert not df["device_worn"].any()
    assert not df["passed_finalQC"].any()


def test_procECG_respects_a_stricter_beat_threshold(make_edf, fake_model):
    """Raising n_beats_min above the beats present must fail every segment."""
    path = make_edf(hours=1.0, bpm=60)
    strict = dataclasses.replace(CFG, n_beats_min=50)  # only ~10 beats per segment

    df = _process_one_chunk(path, strict, fake_model)

    assert not df["passed_finalQC"].any()


# procEDF, the full per-recording pipeline

@pytest.fixture
def processed(make_edf, fake_model, tmp_path):
    """Run the full pipeline over a two-hour synthetic recording."""
    path = make_edf(hours=2.0, bpm=60)
    cfg = dataclasses.replace(CFG, output_dir=tmp_path / "output")
    info = procEDF(str(path), cfg, fake_model, {"model_file": "stub.keras",
                                                "model_sha256": "0" * 64})
    return path, cfg, info


def test_procEDF_succeeds_and_reports_no_failure(processed):
    _, _, info = processed
    assert info["failed"].iloc[0] == 0


def test_procEDF_records_model_provenance(processed):
    """Every result must name the weights that produced it."""
    _, _, info = processed
    assert info["model_file"].iloc[0] == "stub.keras"
    assert info["model_sha256"].iloc[0] == "0" * 64


def test_procEDF_recovers_heart_rate_metrics(processed):
    _, _, info = processed

    assert info["HR_mean"].iloc[0] == pytest.approx(60.0, abs=2.0)
    assert info["HR_min"].iloc[0] == pytest.approx(60.0, abs=3.0)
    assert info["HR_max"].iloc[0] == pytest.approx(60.0, abs=3.0)
    # Starts at 21:00 and barely moves, so the whole recording is resting.
    assert info["HR_rest_robust"].iloc[0] == pytest.approx(60.0, abs=2.0)


def test_procEDF_reports_full_wear_time(processed):
    _, _, info = processed
    assert info["wear_time_ECG_10s"].iloc[0] == pytest.approx(1.0)
    assert info["prop_ECG_worn_passed_finalQC"].iloc[0] > 0.95


def test_procEDF_assigns_all_time_to_the_lowest_activity_band(processed):
    """A still recording must land entirely in sleep/sedentary, not MVPA."""
    _, _, info = processed

    assert info["hours_sleep_sedentary"].iloc[0] == pytest.approx(2.0, abs=0.1)
    assert info["hours_mvpa"].iloc[0] == pytest.approx(0.0, abs=0.01)


def test_procEDF_writes_the_expected_output_files(processed):
    path, cfg, _ = processed
    subject_dir = cfg.output_dir / path.stem

    qc_files = list((subject_dir / "processed_data").glob("*_df_qc.csv.gz"))
    assert len(qc_files) == 1

    plots = list((subject_dir / "plots").glob("*"))
    assert any(p.suffix == ".pdf" for p in plots), "no PDF report was written"


def test_procEDF_qc_csv_has_the_columns_downstream_tools_need(processed):
    """The plotting commands read these columns by name."""
    path, cfg, _ = processed
    qc_file = next((cfg.output_dir / path.stem / "processed_data").glob("*_df_qc.csv.gz"))

    df = pd.read_csv(qc_file)

    for column in ("time", "RRm_imputed", "acc_imputed", "HRm_imputed", "device_worn"):
        assert column in df.columns


def test_procEDF_handles_a_non_worn_recording_without_crashing(make_edf, fake_model, tmp_path):
    """A recording with no usable ECG must be marked failed, not raise."""
    path = make_edf(hours=1.0, worn=False)
    cfg = dataclasses.replace(CFG, output_dir=tmp_path / "output")

    info = procEDF(str(path), cfg, fake_model)

    assert info["failed"].iloc[0] == 1


def test_procEDF_records_why_a_recording_failed(make_edf, fake_model, tmp_path):
    """A failure must explain itself in terms a user can act on.

    A recording where nothing passes QC used to die with a bare
    KeyError: 'RRm', because the column is dropped when every segment fails.
    """
    path = make_edf(hours=1.0, worn=False)
    cfg = dataclasses.replace(CFG, output_dir=tmp_path / "output")

    info = procEDF(str(path), cfg, fake_model)

    reason = info["failure_reason"].iloc[0]
    assert "KeyError" not in reason, "bare KeyError leaked to the user"
    assert "no usable heartbeats" in reason
    assert "device worn" in reason          # says how many segments were worn
    assert "_ECGs_failedQC.pdf" in reason   # points at where to look


def test_procEDF_survives_an_unreadable_file(tmp_path, fake_model):
    """A corrupt file must fail alone, not raise out and abort the batch."""
    broken = tmp_path / "corrupt.EDF"
    broken.write_bytes(b"this is definitely not an EDF file")
    cfg = dataclasses.replace(CFG, output_dir=tmp_path / "output")

    info = procEDF(str(broken), cfg, fake_model)

    assert info["failed"].iloc[0] == 1
    assert "unreadable file" in info["failure_reason"].iloc[0]
    assert info["Name"].iloc[0] == "corrupt.EDF"


def test_procEDF_writes_the_failed_qc_plot_with_few_failing_segments(
    make_edf, make_model, tmp_path
):
    """Few worn-but-failing segments must not break the failed-QC sampling.

    The sample size has to be bounded by the filtered subset; bounding it by
    the whole frame raised "Cannot take a larger sample than population".
    """
    path = make_edf(hours=1.0, bpm=60)
    # n_beats_min just above the beats present fails every segment, so the
    # low-quality branch runs; a strict threshold keeps the subset small.
    cfg = dataclasses.replace(CFG, n_beats_min=11, output_dir=tmp_path / "output")

    info = procEDF(str(path), cfg, make_model(bpm=60))

    # Whatever the outcome, it must not be a sampling crash.
    reason = str(info.get("failure_reason", pd.Series([""])).iloc[0])
    assert "larger sample than population" not in reason


def test_summary_metrics_agree_with_the_qc_table(processed):
    """The reported means must match a recomputation from the saved data."""
    path, cfg, info = processed
    qc_file = next((cfg.output_dir / path.stem / "processed_data").glob("*_df_qc.csv.gz"))
    df = pd.read_csv(qc_file, parse_dates=["time"])

    recomputed = calculate_summary_metrics(df, cfg)

    assert recomputed["HR_mean"] == pytest.approx(info["HR_mean"].iloc[0])
    assert recomputed["HR_rest_robust"] == pytest.approx(info["HR_rest_robust"].iloc[0])


# Chunk boundaries
#
# pyedflib.readSignal returns an empty array when n exceeds the total signal
# length, and zero-pads when reading past the end from a valid start. procECG
# clamps the request to the samples that remain, so neither leaks through.

def test_recording_shorter_than_one_chunk_is_processed(make_edf, fake_model, tmp_path):
    """A one-hour recording must process fine under the default 24-hour chunk.

    Before the chunk length was clamped, readSignal returned nothing at all
    here and the recording failed outright.
    """
    path = make_edf(hours=1.0, bpm=60)
    cfg = dataclasses.replace(Config(), output_dir=tmp_path / "output")  # chunk_hours=24

    info = procEDF(str(path), cfg, fake_model)

    assert info["failed"].iloc[0] == 0
    assert info["wear_time_ECG_10s"].iloc[0] == pytest.approx(1.0)
    assert info["HR_mean"].iloc[0] == pytest.approx(60.0, abs=2.0)


def test_partial_final_chunk_does_not_invent_non_wear_time(make_edf, fake_model, tmp_path):
    """A 1.5-hour recording in 1-hour chunks must report 1.5 hours, not 2.

    The final chunk has only half an hour left in it. Zero-padding it out to a
    full hour would append 180 phantom non-wear segments and understate wear
    time by a quarter.
    """
    path = make_edf(hours=1.5, bpm=60)
    cfg = dataclasses.replace(CFG, output_dir=tmp_path / "output")  # chunk_hours=1

    info = procEDF(str(path), cfg, fake_model)

    qc_file = next((cfg.output_dir / path.stem / "processed_data").glob("*_df_qc.csv.gz"))
    df = pd.read_csv(qc_file, parse_dates=["time"])

    assert len(df) == int(1.5 * 3600 / 10)  # 540, not 720
    assert info["wear_time_ECG_10s"].iloc[0] == pytest.approx(1.0)
    # The time axis must stop at the end of the recording, not a chunk later.
    assert df["time"].max() < pd.Timestamp("2025-01-06 22:30:00")


def test_chunk_starting_past_the_end_yields_nothing(make_edf, fake_model):
    """A chunk index beyond the recording returns empty rather than zeros."""
    path = make_edf(hours=1.0)
    chunk_samples = int(CFG.fs_expected * CFG.chunk_seconds)

    with pyedflib.EdfReader(str(path)) as f:
        df = procECG(f, 5, chunk_samples, path.name, CFG, fake_model)

    assert df.empty


def test_segment_count_is_exact_across_several_chunks(make_edf, fake_model, tmp_path):
    """Total segments must equal the recording length, however it is chunked."""
    path = make_edf(hours=2.0, bpm=60)

    counts = []
    for chunk_hours in (1, 2, 24):
        cfg = dataclasses.replace(
            CFG, chunk_hours=chunk_hours, output_dir=tmp_path / f"out_{chunk_hours}"
        )
        procEDF(str(path), cfg, fake_model)
        qc_file = next((cfg.output_dir / path.stem / "processed_data").glob("*_df_qc.csv.gz"))
        counts.append(len(pd.read_csv(qc_file)))

    assert counts == [720, 720, 720]  # 2 hours of 10-second segments, every time
