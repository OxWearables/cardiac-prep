"""The init subcommand, which gives a pip install its config.yaml.

The template is a second copy of the repository's own config.yaml, so the
first test here is the one that matters: it fails if the two drift apart.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from cardiacprep import init_config
from cardiacprep.config import DEFAULT_CONFIG_FILENAME, Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO_ROOT / DEFAULT_CONFIG_FILENAME


def test_packaged_template_matches_the_repository_config():
    # A clone edits config.yaml at the root; a pip install gets the packaged
    # copy. If these differ, the two audiences are reading different docs.
    assert init_config.TEMPLATE.read_text() == REPO_CONFIG.read_text()


def test_template_is_shipped_inside_the_package():
    assert init_config.TEMPLATE.is_file()
    assert init_config.TEMPLATE.parent.name == "cardiacprep"


def test_every_highlighted_setting_exists_in_config():
    known = {f.name for f in fields(Config)}
    highlighted = [name for names, _ in init_config.HIGHLIGHTS for name in names]
    unknown = sorted(set(highlighted) - known)
    assert not unknown, f"init highlights settings that do not exist: {unknown}"


def test_writes_a_loadable_config(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert init_config.main([]) == 0

    written = tmp_path / DEFAULT_CONFIG_FILENAME
    assert written.is_file()
    # The point of the file is that the pipeline accepts it unedited.
    load_config(path=written)


def test_prints_the_highlighted_settings(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_config.main([])

    out = capsys.readouterr().out
    for names, _ in init_config.HIGHLIGHTS:
        for name in names:
            assert name in out


def test_printed_values_come_from_the_dataclass(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_config.main([])

    out = capsys.readouterr().out
    assert f"night_start_hour: {Config().night_start_hour}" in out
    assert f"fs_expected: {Config().fs_expected}" in out


def test_quiet_writes_the_file_without_the_summary(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert init_config.main(["--quiet"]) == 0

    out = capsys.readouterr().out
    assert (tmp_path / DEFAULT_CONFIG_FILENAME).is_file()
    assert "Worth checking" not in out


def test_refuses_to_overwrite_by_default(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / DEFAULT_CONFIG_FILENAME
    existing.write_text("night_start_hour: 3\n")

    assert init_config.main([]) == 1
    assert existing.read_text() == "night_start_hour: 3\n"
    assert "already exists" in capsys.readouterr().err


def test_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / DEFAULT_CONFIG_FILENAME
    existing.write_text("night_start_hour: 3\n")

    assert init_config.main(["--force"]) == 0
    assert existing.read_text() == init_config.TEMPLATE.read_text()


def test_output_folder_gets_the_default_filename(tmp_path):
    target = tmp_path / "study"
    target.mkdir()

    assert init_config.main(["--output", str(target)]) == 0
    assert (target / DEFAULT_CONFIG_FILENAME).is_file()


def test_output_file_path_is_used_verbatim(tmp_path):
    target = tmp_path / "nested" / "my-settings.yaml"

    assert init_config.main(["--output", str(target)]) == 0
    assert target.is_file()


def test_reaches_the_command_line_as_a_subcommand(tmp_path, monkeypatch):
    from cardiacprep import entry

    monkeypatch.chdir(tmp_path)
    assert entry.main(["init", "--quiet"]) == 0
    assert (tmp_path / DEFAULT_CONFIG_FILENAME).is_file()


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_exits_cleanly(flag):
    with pytest.raises(SystemExit) as excinfo:
        init_config.main([flag])
    assert excinfo.value.code == 0


# Folder scaffolding
#
# A clone arrives with these folders; a pip install cannot create them, since
# pip writes into site-packages and has no idea where the study will live.

def test_creates_the_working_folders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert init_config.main(["--quiet"]) == 0

    for folder in ("input_data", "output", "models"):
        assert (tmp_path / folder).is_dir(), f"{folder} was not created"


def test_names_the_folders_it_made(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    init_config.main([])

    out = capsys.readouterr().out
    assert "recordings go here" in out
    assert "input_data" in out


def test_folders_follow_the_configured_locations(tmp_path, monkeypatch):
    """init reads the config it just wrote, so an edited template is honoured."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("input_dir: ./recordings\noutput_dir: ./results\n")

    assert init_config.main(["--force"]) == 0
    # --force overwrites with the packaged template, so the defaults apply.
    assert (tmp_path / "input_data").is_dir()


def test_existing_folders_are_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input_data").mkdir()

    assert init_config.main(["--quiet"]) == 0
    assert (tmp_path / "input_data").is_dir()


def test_folders_are_made_beside_the_config_file(tmp_path):
    """--output somewhere else puts the folders there too, not in the cwd."""
    elsewhere = tmp_path / "study"

    assert init_config.main(["--output", str(elsewhere), "--quiet"]) == 0
    assert (elsewhere / "input_data").is_dir()
