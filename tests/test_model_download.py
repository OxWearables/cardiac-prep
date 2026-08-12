"""Downloading the QRS detector weights.

Every test here serves a local HTTP server rather than mocking urlopen, so the
streaming, the temporary file and the rename are all genuinely exercised. The
payloads are a few bytes of nonsense; nothing here needs real weights.
"""

import functools
import http.server
import threading
from pathlib import Path

import pytest

from cardiacprep import model_download
from cardiacprep.model_utils import ModelError, sha256_of

PAYLOAD = b"not really a keras model, but it downloads the same"

# The digest shipped with the package, captured before any test patches it.
CONFIGURED_SHA256 = model_download.MODEL_SHA256


@pytest.fixture(autouse=True)
def expect_the_test_payload(monkeypatch):
    """Point the configured checksum at the stub these tests serve.

    Without this every happy-path test would fail verification against the
    digest of the real published weights, which is not what is being served.
    """
    import hashlib

    monkeypatch.setattr(
        model_download, "MODEL_SHA256", hashlib.sha256(PAYLOAD).hexdigest()
    )


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A directory served over HTTP, yielding (base_url, directory)."""
    root = tmp_path_factory.mktemp("served")
    (root / model_download.model_filename()).write_bytes(PAYLOAD)

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(root)
    )
    # Port 0 lets the OS pick a free one, so tests do not collide.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", root
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def digest():
    import hashlib

    return hashlib.sha256(PAYLOAD).hexdigest()


def test_downloads_the_weights(served, tmp_path):
    base_url, _ = served
    path = model_download.download_model(tmp_path, base_url=base_url)

    assert path == tmp_path / model_download.model_filename()
    assert path.read_bytes() == PAYLOAD


def test_creates_a_missing_model_folder(served, tmp_path):
    base_url, _ = served
    target = tmp_path / "nested" / "models"

    path = model_download.download_model(target, base_url=base_url)
    assert path.is_file()


def test_verifies_a_correct_checksum(served, tmp_path, digest):
    base_url, _ = served
    path = model_download.download_model(
        tmp_path, base_url=base_url, expected_sha256=digest
    )
    assert sha256_of(path) == digest


def test_rejects_and_discards_a_bad_checksum(served, tmp_path):
    base_url, _ = served
    with pytest.raises(ModelError, match="checksum"):
        model_download.download_model(
            tmp_path, base_url=base_url, expected_sha256="a" * 64
        )

    # The point of verifying: a mismatched file must not be left behind
    # looking like usable weights.
    assert not (tmp_path / model_download.model_filename()).exists()


def test_leaves_no_partial_file_behind_on_failure(tmp_path):
    with pytest.raises(ModelError):
        model_download.download_model(
            tmp_path, base_url="http://127.0.0.1:1/nowhere/"
        )

    assert list(tmp_path.iterdir()) == []


def test_missing_file_gives_a_readable_error(served, tmp_path):
    base_url, _ = served
    with pytest.raises(ModelError, match="404"):
        model_download.download_model(
            tmp_path, version="no_such_model", base_url=base_url
        )


def test_existing_weights_are_not_downloaded_again(served, tmp_path):
    base_url, _ = served
    target = tmp_path / model_download.model_filename()
    target.write_bytes(b"already here")

    path = model_download.download_model(tmp_path, base_url=base_url)
    assert path.read_bytes() == b"already here"


def test_force_overwrites_existing_weights(served, tmp_path):
    base_url, _ = served
    target = tmp_path / model_download.model_filename()
    target.write_bytes(b"stale")

    model_download.download_model(tmp_path, base_url=base_url, force=True)
    assert target.read_bytes() == PAYLOAD


# ensure_model

def test_ensure_model_downloads_when_the_folder_is_empty(served, tmp_path):
    base_url, _ = served
    path = model_download.ensure_model(tmp_path, base_url=base_url)
    assert path.is_file()


def test_ensure_model_uses_what_is_already_there(tmp_path):
    existing = tmp_path / "someone_elses_name.keras"
    existing.write_bytes(b"local weights")

    # No base_url given, so any download attempt would fail loudly.
    assert model_download.ensure_model(tmp_path) == existing


def test_ensure_model_does_not_download_when_disabled(tmp_path):
    with pytest.raises(ModelError):
        model_download.ensure_model(tmp_path, auto_download=False)


def test_ensure_model_refuses_to_guess_between_two_models(served, tmp_path):
    base_url, _ = served
    (tmp_path / "one.keras").write_bytes(b"a")
    (tmp_path / "two.keras").write_bytes(b"b")

    # Downloading a third file would make an ambiguous folder worse.
    with pytest.raises(ModelError, match="Refusing to guess"):
        model_download.ensure_model(tmp_path, base_url=base_url)


def test_ensure_model_respects_an_explicit_path(served, tmp_path):
    base_url, _ = served
    chosen = tmp_path / "chosen.keras"
    chosen.write_bytes(b"chosen")

    assert model_download.ensure_model(
        tmp_path, model_path=chosen, base_url=base_url
    ) == chosen


# command line

def test_download_subcommand_reports_success(served, tmp_path, capsys):
    base_url, _ = served
    assert model_download.main(
        ["--model-dir", str(tmp_path), "--url", base_url]
    ) == 0
    assert "Weights ready at" in capsys.readouterr().out


def test_download_subcommand_reports_failure(tmp_path, capsys):
    assert model_download.main(
        ["--model-dir", str(tmp_path), "--url", "http://127.0.0.1:1/nowhere/"]
    ) == 1
    assert "Could not download" in capsys.readouterr().err


def test_download_is_reachable_as_a_subcommand(served, tmp_path, capsys):
    from cardiacprep import entry

    base_url, _ = served
    assert entry.main(
        ["download", "--model-dir", str(tmp_path), "--url", base_url]
    ) == 0


def test_default_url_points_at_the_group_file_server():
    # Matches where actinet and stepcount publish their models.
    assert model_download.MODEL_BASE_URL.startswith(
        "https://wearables-files.ndph.ox.ac.uk/files/models/"
    )
    assert model_download.model_url().endswith(".keras")


def test_the_published_weights_have_a_configured_checksum():
    # An unset digest would mean every download is accepted unverified, and a
    # substituted file would produce plausible but wrong beat detections.
    assert CONFIGURED_SHA256 is not None, "MODEL_SHA256 must not be left unset"
    assert len(CONFIGURED_SHA256) == 64
    int(CONFIGURED_SHA256, 16)  # raises if it is not hexadecimal


def test_unverified_download_warns(served, tmp_path, caplog):
    base_url, _ = served
    with caplog.at_level("WARNING"):
        model_download.download_model(
            tmp_path, base_url=base_url, expected_sha256=None
        )
    assert any("could not be verified" in r.message for r in caplog.records)


def test_model_dir_is_expanded(served, tmp_path, monkeypatch):
    base_url, _ = served
    monkeypatch.setenv("HOME", str(tmp_path))
    path = model_download.download_model(Path("~/models"), base_url=base_url)
    assert path.is_file()
    assert "~" not in str(path)
