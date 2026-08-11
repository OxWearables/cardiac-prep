"""Guards that keep the generated configuration docs honest.

The configuration reference page is built from the comments in config.yaml, so
a setting that exists in the dataclass but never appears in the YAML would be
silently absent from the documentation. These tests fail instead.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from cardiacprep.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config.yaml"
GENERATOR = REPO_ROOT / "docs" / "source" / "_generate.py"


def _load_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_docs_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sections():
    return _load_generator().parse_config_yaml(CONFIG_YAML)


def test_every_setting_is_documented_in_config_yaml(sections):
    documented = _load_generator().setting_names(sections)
    missing = sorted({f.name for f in fields(Config)} - documented)
    assert not missing, (
        "These settings exist in Config but not in config.yaml, so they would "
        f"be missing from the docs: {', '.join(missing)}"
    )


def test_config_yaml_has_no_settings_that_config_rejects(sections):
    documented = _load_generator().setting_names(sections)
    known = {f.name for f in fields(Config)}
    unknown = sorted(documented - known)
    assert not unknown, (
        "config.yaml documents settings that Config does not accept, so the "
        f"pipeline would refuse to start: {', '.join(unknown)}"
    )


def test_every_setting_has_some_prose(sections):
    # A setting with neither its own comment nor a section intro would render
    # as a bare name and default, which tells a reader nothing.
    bare = [
        setting.name
        for section in sections
        for setting in section.settings
        if not setting.doc and not section.intro
    ]
    assert not bare, f"Settings with no explanatory comment: {', '.join(bare)}"


def test_advanced_settings_are_marked_as_such(sections):
    advanced = {
        name
        for section in sections
        if section.advanced
        for setting in section.settings
        for name in setting.names
    }
    # These sit below the ADVANCED banner in config.yaml and change what the
    # bundled detector sees, so the docs must not present them as everyday.
    for name in ("mains_hz", "qrs_threshold", "ecg_clip_mv"):
        assert name in advanced, f"{name} should be under the ADVANCED banner"
