"""Tests for configuration loading and validation.

A bad config file is the most likely thing to go wrong for a new user, so the
failure modes here matter as much as the happy path.
"""

import pytest

from cardiacprep.config import Config, ConfigError, load_config


def _write(tmp_path, text, name="config.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


# Defaults

def test_defaults_are_valid():
    Config().validate()  # must not raise


def test_defaults_match_documented_values():
    cfg = Config()
    assert cfg.segment_samples == 2500  # 10 s at 250 Hz
    assert cfg.chunk_seconds == 86400   # 24 h
    assert cfg.night_start_hour == 21
    assert cfg.night_end_hour == 9


def test_resolved_n_processes_is_at_least_one():
    assert Config().resolved_n_processes >= 1


def test_resolved_n_processes_honours_explicit_value():
    assert Config(n_processes=3).resolved_n_processes == 3


# Loading

def test_missing_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml present
    assert load_config() == Config()


def test_explicit_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(path=tmp_path / "nope.yaml")


def test_file_values_override_defaults(tmp_path):
    path = _write(tmp_path, "n_beats_min: 9\nnight_start_hour: 23\n")

    cfg = load_config(path=path)

    assert cfg.n_beats_min == 9
    assert cfg.night_start_hour == 23
    assert cfg.rr_min_ms == Config().rr_min_ms  # untouched settings keep defaults


def test_empty_file_is_treated_as_no_settings(tmp_path):
    assert load_config(path=_write(tmp_path, "")) == Config()


def test_paths_are_coerced_to_path_objects(tmp_path):
    cfg = load_config(path=_write(tmp_path, "input_dir: /tmp/somewhere\n"))
    assert cfg.input_dir.name == "somewhere"


def test_overrides_beat_file_values(tmp_path):
    path = _write(tmp_path, "n_processes: 2\n")

    cfg = load_config(path=path, overrides={"n_processes": 7})

    assert cfg.n_processes == 7


def test_none_overrides_do_not_clobber_file_values(tmp_path):
    """Unset CLI flags arrive as None and must leave config.yaml alone."""
    path = _write(tmp_path, "n_processes: 2\n")

    cfg = load_config(path=path, overrides={"n_processes": None})

    assert cfg.n_processes == 2


# Error handling

def test_unknown_setting_is_rejected_with_a_hint(tmp_path):
    path = _write(tmp_path, "nite_start_hour: 22\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path=path)

    message = str(excinfo.value)
    assert "nite_start_hour" in message
    assert "night_start_hour" in message  # the valid-settings list guides the fix


def test_malformed_yaml_is_reported_clearly(tmp_path):
    path = _write(tmp_path, "input_dir: [unclosed\n")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path=path)


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = _write(tmp_path, "- just\n- a\n- list\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path=path)


# Validation

def test_unordered_activity_thresholds_are_rejected():
    with pytest.raises(ConfigError, match="must increase"):
        Config(activity_light_mg=100.0).validate()


def test_inverted_rr_limits_are_rejected():
    with pytest.raises(ConfigError, match="rr_min_ms"):
        Config(rr_min_ms=3000, rr_max_ms=250).validate()


def test_out_of_range_fraction_is_rejected():
    with pytest.raises(ConfigError, match="between 0 and 1"):
        Config(rr_cover_min=1.5).validate()


def test_out_of_range_hour_is_rejected():
    with pytest.raises(ConfigError, match="0 to 23"):
        Config(night_start_hour=25).validate()


def test_identical_rest_window_hours_are_rejected():
    """Equal start and end would silently select no rest period at all."""
    with pytest.raises(ConfigError, match="no rest period"):
        Config(night_start_hour=9, night_end_hour=9).validate()


def test_zero_processes_is_rejected():
    with pytest.raises(ConfigError, match="at least 1"):
        Config(n_processes=0).validate()


def test_validation_runs_during_load(tmp_path):
    """An invalid file must fail at load time, not midway through a long run."""
    path = _write(tmp_path, "night_start_hour: 99\n")

    with pytest.raises(ConfigError):
        load_config(path=path)


# Bundled config.yaml

def test_shipped_config_file_is_loadable():
    """The config.yaml committed to the repo must actually parse and validate."""
    from pathlib import Path

    shipped = Path(__file__).resolve().parent.parent / "config.yaml"
    assert shipped.is_file(), "config.yaml is missing from the repository"

    cfg = load_config(path=shipped)

    # It documents the defaults, so it should agree with them.
    assert cfg.activity_thresholds == Config().activity_thresholds
    assert cfg.night_start_hour == Config().night_start_hour
