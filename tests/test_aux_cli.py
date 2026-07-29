"""End-to-end tests for the two auxiliary commands.

These run main() directly with synthetic output folders, checking exit codes
and that the expected files appear. All fixtures are synthetic - no
participant data is involved.
"""

import numpy as np
import pandas as pd
import pytest

from edfproc import dataset_summary, subject_plots


@pytest.fixture(autouse=True)
def _empty_config(tmp_path, monkeypatch):
    """Run from a directory with no config.yaml, so defaults apply cleanly."""
    monkeypatch.chdir(tmp_path)


def _make_subject(output_dir, subject_id, hours=26):
    processed = output_dir / subject_id / "processed_data"
    processed.mkdir(parents=True)
    n = int(hours * 3600 / 10)
    pd.DataFrame(
        {
            "time": pd.date_range("2025-01-06 00:00:00", periods=n, freq="10s"),
            "RRm_imputed": np.full(n, 1000.0),
            "HRm_imputed": np.full(n, 60.0),
            "acc_imputed": np.linspace(0, 50, n),
        }
    ).to_csv(processed / f"{subject_id}.EDF_df_qc.csv.gz", compression="gzip", index=False)


def _make_summary(output_dir, n=12):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "Name": [f"rec_{i}" for i in range(n)],
            "HR_rest_robust": rng.normal(58, 5, n),
            "median_daily_rmssd": rng.normal(35, 8, n),
            "hours_mvpa": rng.normal(1.0, 0.3, n),
            "hours_light_activity": rng.normal(4.0, 1.0, n),
            "prop_ECG_passed_finalQC": rng.uniform(0.6, 1.0, n),
            "frac_RR_imp": rng.uniform(0.0, 0.3, n),
        }
    ).to_csv(output_dir / "df_info_summary.csv.gz", compression="gzip", index=False)


# subject_plots

def test_list_reports_available_subjects(tmp_path, capsys):
    out = tmp_path / "results"
    _make_subject(out, "subject_a")

    assert subject_plots.main(["--list", "--output", str(out)]) == 0
    assert "subject_a" in capsys.readouterr().out


def test_list_with_no_results_exits_nonzero(tmp_path):
    assert subject_plots.main(["--list", "--output", str(tmp_path / "empty")]) == 1


def test_missing_subject_argument_is_an_error(tmp_path):
    assert subject_plots.main(["--output", str(tmp_path)]) == 2


def test_unknown_subject_exits_nonzero(tmp_path, capsys):
    out = tmp_path / "results"
    _make_subject(out, "real")

    assert subject_plots.main(["--subject", "typo", "--output", str(out)]) == 1
    assert "real" in capsys.readouterr().err  # error names the valid options


def test_all_plot_kinds_are_written(tmp_path):
    out = tmp_path / "results"
    _make_subject(out, "s1")

    assert subject_plots.main(["--subject", "s1", "--output", str(out)]) == 0

    written = sorted(p.name for p in (out / "s1" / "plots").glob("*.png"))
    assert written == [
        "s1_24hr_profile.png",
        "s1_acc_heatmap.png",
        "s1_daily_heart_rate.png",
        "s1_hr_heatmap.png",
        "s1_timeseries.png",
    ]


def test_single_kind_writes_only_that_plot(tmp_path):
    out = tmp_path / "results"
    _make_subject(out, "s1")

    assert subject_plots.main(
        ["--subject", "s1", "--output", str(out), "--kind", "timeseries"]
    ) == 0

    written = list((out / "s1" / "plots").glob("*.png"))
    assert [p.name for p in written] == ["s1_timeseries.png"]


def test_no_save_writes_nothing(tmp_path):
    out = tmp_path / "results"
    _make_subject(out, "s1")

    assert subject_plots.main(["--subject", "s1", "--output", str(out), "--no-save"]) == 0
    assert not (out / "s1" / "plots").exists()


# dataset_summary

def test_summary_generates_both_plots(tmp_path):
    out = tmp_path / "results"
    _make_summary(out)

    assert dataset_summary.main(["--output", str(out)]) == 0
    assert (out / "dataset_summary_histograms.png").is_file()
    assert (out / "dataset_rhr_vs_hrv.png").is_file()


def test_summary_without_a_summary_file_exits_nonzero(tmp_path, capsys):
    assert dataset_summary.main(["--output", str(tmp_path / "empty")]) == 1
    assert "process.py" in capsys.readouterr().err


def test_summary_rejects_a_file_from_an_older_pipeline(tmp_path, capsys):
    """A summary lacking the HRV columns predates the current pipeline."""
    out = tmp_path / "results"
    out.mkdir()
    pd.DataFrame({"Name": ["a"], "HR_mean": [60.0]}).to_csv(
        out / "df_info_summary.csv.gz", compression="gzip", index=False
    )

    assert dataset_summary.main(["--output", str(out)]) == 1
    assert "median_daily_rmssd" in capsys.readouterr().err


def test_summary_tolerates_missing_optional_columns(tmp_path):
    """Panels for absent columns are skipped rather than crashing the run."""
    out = tmp_path / "results"
    out.mkdir()
    pd.DataFrame(
        {"HR_rest_robust": [55.0, 60.0], "median_daily_rmssd": [30.0, 40.0]}
    ).to_csv(out / "df_info_summary.csv.gz", compression="gzip", index=False)

    assert dataset_summary.main(["--output", str(out)]) == 0
    assert (out / "dataset_summary_histograms.png").is_file()
