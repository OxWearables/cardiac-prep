"""Guards that keep the data dictionary describing the real output.

A data dictionary is worth having only if it is true. The important test here
compares it against the columns the pipeline actually writes when it runs, so
a column added in six months cannot go undocumented, and a column removed
cannot linger in the docs describing something that no longer exists.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from cardiacprep.config import Config
from cardiacprep.proc_edf import procEDF

# Short chunks so the synthetic recording exercises the chunk-boundary path,
# matching test_pipeline_e2e.
CFG = dataclasses.replace(Config(), chunk_hours=1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATADICT_CSV = REPO_ROOT / "docs" / "datadict.csv"
GENERATOR = REPO_ROOT / "docs" / "source" / "_generate.py"

VALID_TYPES = {"boolean", "integer", "float", "string", "datetime"}

# Columns actipy only emits when the accelerometer calibration converges. The
# fit needs the device held in several orientations, which a still synthetic
# recording never provides, so the fixture below cannot produce them. They do
# appear in real output. Listed here so the staleness check stays meaningful
# for every other column rather than being abandoned.
CONDITIONAL_COLUMNS = {
    "CalibxIntercept", "CalibyIntercept", "CalibzIntercept",
    "CalibxSlope", "CalibySlope", "CalibzSlope",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("_docs_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def entries():
    return _load_generator().read_datadict(DATADICT_CSV)


@pytest.fixture(scope="module")
def documented(entries):
    return {
        key: {e["Name"] for e in entries if e["file"] == key}
        for key in ("df_info_summary", "df_qc")
    }


# The dictionary itself

def test_every_row_is_complete(entries):
    incomplete = [
        e.get("Name", "?")
        for e in entries
        if not all(e.get(f) for f in ("file", "Name", "Description", "Type", "Unit"))
    ]
    assert not incomplete, f"rows with an empty cell: {incomplete}"


def test_types_come_from_the_agreed_vocabulary(entries):
    wrong = sorted({e["Type"] for e in entries} - VALID_TYPES)
    assert not wrong, f"unexpected Type values: {wrong}. Allowed: {sorted(VALID_TYPES)}"


def test_every_file_label_is_known(entries):
    wrong = sorted({e["file"] for e in entries} - {"df_info_summary", "df_qc"})
    assert not wrong, f"unexpected file labels: {wrong}"


def test_no_duplicate_columns(entries):
    seen = [(e["file"], e["Name"]) for e in entries]
    duplicates = sorted({x for x in seen if seen.count(x) > 1})
    assert not duplicates, f"documented more than once: {duplicates}"


def test_descriptions_are_tidied_on_the_way_in(entries):
    # Punctuation and spacing are normalised by the generator rather than kept
    # in the CSV, so that a save from Excel or Numbers cannot undo them.
    unpunctuated = [
        e["Name"] for e in entries if not e["Description"].rstrip(" ⚠️").endswith(".")
    ]
    assert not unpunctuated, f"descriptions missing a full stop: {unpunctuated}"

    untidy = [e["Name"] for e in entries if "  " in e["Description"]]
    assert not untidy, f"descriptions with doubled spaces: {untidy}"


def test_tidying_handles_the_awkward_cases():
    tidy = _load_generator()._tidy_description
    assert tidy("no full stop") == "no full stop."
    assert tidy("already fine.") == "already fine."
    assert tidy("  padded  out  ") == "padded out."
    assert tidy("warned about ⚠️") == "warned about. ⚠️"
    assert tidy("warned about. ⚠️") == "warned about. ⚠️"
    assert tidy("") == ""


# The dictionary against the real pipeline output

@pytest.fixture
def real_output(make_edf, fake_model, tmp_path):
    """Run the pipeline and hand back the columns it actually wrote."""
    path = make_edf(hours=2.0, bpm=60)
    cfg = dataclasses.replace(CFG, output_dir=tmp_path / "output")
    info = procEDF(
        str(path), cfg, fake_model,
        {"model_file": "stub.keras", "model_sha256": "0" * 64},
    )

    qc_file = next(
        (cfg.output_dir / path.stem / "processed_data").glob("*_df_qc.csv.gz")
    )
    qc = pd.read_csv(qc_file)

    # The segment index is written as an unnamed first column, which pandas
    # reads back as "Unnamed: 0". The dictionary documents it as "index", so
    # rename it here and the checks below cover it like any other column.
    qc_columns = {"index" if c.startswith("Unnamed") else c for c in qc.columns}

    return {"df_info_summary": set(info.columns), "df_qc": qc_columns}


@pytest.mark.parametrize("key", ["df_info_summary", "df_qc"])
def test_every_written_column_is_documented(key, real_output, documented):
    undocumented = sorted(real_output[key] - documented[key])
    assert not undocumented, (
        f"{key} contains columns missing from docs/datadict.csv: {undocumented}"
    )


@pytest.mark.parametrize("key", ["df_info_summary", "df_qc"])
def test_no_documented_column_has_disappeared(key, real_output, documented):
    stale = sorted(documented[key] - real_output[key] - CONDITIONAL_COLUMNS)
    assert not stale, (
        f"docs/datadict.csv documents {key} columns the pipeline no longer "
        f"writes: {stale}"
    )
