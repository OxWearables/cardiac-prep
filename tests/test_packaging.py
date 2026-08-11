"""Checks on how the project is packaged.

A packaging mistake does not show up when running from a clone - it shows up
for the first person who installs from PyPI, by which point the release cannot
be corrected, since PyPI refuses reuploads of a version.
"""

import re
from pathlib import Path

import pytest

import cardiacprep
from cardiacprep import init_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject_text():
    return PYPROJECT.read_text(encoding="utf-8")


def test_version_is_declared_in_exactly_one_place(pyproject_text):
    # pyproject reads the version from cardiacprep.__version__. A literal
    # version key would reintroduce two places to bump, and they would drift.
    assert 'version = { attr = "cardiacprep.__version__" }' in pyproject_text
    assert not re.search(r'^version\s*=\s*"', pyproject_text, re.MULTILINE)


def test_version_looks_like_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", cardiacprep.__version__)


def test_installed_metadata_matches_the_package_version():
    # Catches a stale editable install as well as a genuine mismatch.
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("cardiacprep")
    except PackageNotFoundError:
        pytest.skip("cardiacprep is not installed in this environment")
    assert installed == cardiacprep.__version__


def test_distribution_name_matches_the_import_name(pyproject_text):
    # pip install cardiacprep -> import cardiacprep. PyPI treats only '-' and
    # '_' as equivalent, so 'cardiac-prep' would be a different project.
    assert 'name = "cardiacprep"' in pyproject_text


def test_settings_template_is_declared_as_package_data(pyproject_text):
    assert 'cardiacprep = ["default_config.yaml"]' in pyproject_text
    assert init_config.TEMPLATE.is_file()


def test_entry_point_targets_the_dispatcher(pyproject_text):
    assert 'cardiac-prep = "cardiacprep.entry:main"' in pyproject_text


def test_publish_workflow_uses_trusted_publishing():
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text()
    # An API token in the repository would be a long-lived credential with
    # upload rights to the whole project; OIDC avoids storing one at all.
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "password:" not in workflow
