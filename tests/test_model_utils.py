"""Tests for model discovery and provenance recording.

Loading the wrong weights would corrupt every downstream result while the
pipeline still appeared to succeed, so the refusal-to-guess behaviour is the
most important thing covered here.
"""

import hashlib

import pytest

from edfproc.model_utils import ModelError, find_model, model_fingerprint, sha256_of


def _fake_model(directory, name, content=b"fake weights"):
    path = directory / name
    path.write_bytes(content)
    return path


# find_model

def test_finds_single_model_regardless_of_filename(tmp_path):
    """Discovery is by extension, so any filename works."""
    expected = _fake_model(tmp_path, "some_other_name_v3.keras")

    assert find_model(tmp_path) == expected


def test_explicit_path_wins_over_discovery(tmp_path):
    _fake_model(tmp_path, "auto.keras")
    chosen = _fake_model(tmp_path, "chosen.keras")

    assert find_model(tmp_path, model_path=chosen) == chosen


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(ModelError, match="Model folder not found"):
        find_model(tmp_path / "does_not_exist")


def test_empty_directory_points_the_user_at_the_readme(tmp_path):
    with pytest.raises(ModelError) as excinfo:
        find_model(tmp_path)

    assert "models/README.md" in str(excinfo.value)


def test_explicit_path_that_does_not_exist_is_reported(tmp_path):
    with pytest.raises(ModelError, match="Model file not found"):
        find_model(tmp_path, model_path=tmp_path / "ghost.keras")


def test_multiple_models_refuses_to_guess(tmp_path):
    """Silently picking one would produce plausible but untraceable results."""
    _fake_model(tmp_path, "detector_v1.keras")
    _fake_model(tmp_path, "detector_v2.keras")

    with pytest.raises(ModelError) as excinfo:
        find_model(tmp_path)

    message = str(excinfo.value)
    assert "detector_v1.keras" in message
    assert "detector_v2.keras" in message
    assert "model_path" in message  # tells the user how to resolve it


def test_non_keras_files_are_ignored(tmp_path):
    _fake_model(tmp_path, "notes.txt")
    _fake_model(tmp_path, "old_model.h5")
    expected = _fake_model(tmp_path, "detector.keras")

    assert find_model(tmp_path) == expected


# Hashing and provenance

def test_sha256_matches_hashlib(tmp_path):
    content = b"some model weights here"
    path = _fake_model(tmp_path, "m.keras", content)

    assert sha256_of(path) == hashlib.sha256(content).hexdigest()


def test_sha256_is_stable_across_chunk_sizes(tmp_path):
    """Chunked reading must not depend on the chunk size."""
    path = _fake_model(tmp_path, "m.keras", b"x" * 5000)

    assert sha256_of(path, chunk_bytes=7) == sha256_of(path, chunk_bytes=4096)


def test_fingerprint_reports_name_and_hash(tmp_path):
    content = b"weights"
    path = _fake_model(tmp_path, "detector.keras", content)

    fingerprint = model_fingerprint(path)

    assert fingerprint["model_file"] == "detector.keras"
    assert fingerprint["model_sha256"] == hashlib.sha256(content).hexdigest()


def test_fingerprint_distinguishes_different_weights_with_the_same_name(tmp_path):
    """The point of hashing: identical filenames, different contents."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    one = _fake_model(first, "detector.keras", b"version one")
    two = _fake_model(second, "detector.keras", b"version two")

    assert model_fingerprint(one)["model_file"] == model_fingerprint(two)["model_file"]
    assert model_fingerprint(one)["model_sha256"] != model_fingerprint(two)["model_sha256"]
