"""Fetching the QRS detector weights.

The weights are too large to keep in the repository, so they are hosted on the
group's file server, in the same place and the same way as actinet's and
stepcount's models. Someone installing from PyPI gets no weights with the
package and would otherwise have to find and place a file by hand before the
pipeline would run at all.

Two properties matter more than convenience here:

* **The download is verified.** A truncated or substituted file would load and
  produce plausible-looking beat detections that were subtly wrong, which is
  far worse than an error. The expected hash is checked before the file is put
  into place.
* **The download is atomic.** It is written to a temporary file alongside the
  destination and renamed only once verified, so an interrupted download can
  never leave a half-written file that looks like usable weights.
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from . import __version__
from .config import ConfigError, load_config
from .logging_utils import configure_logging, get_logger
from .model_utils import MODEL_GLOB, ModelError, find_model, sha256_of

log = get_logger("model")

# Hosted alongside the group's other models. See docs for how to request that
# a new version is published here.
MODEL_BASE_URL = "https://wearables-files.ndph.ox.ac.uk/files/models/cardiac-prep/"

# The weights this release expects, without the .keras suffix.
MODEL_VERSION = "QRS_detector_125Hz_080525"

# SHA-256 of the published file, checked before a download is put into place.
# Publishing new weights means updating this at the same time, otherwise every
# download is rejected.
MODEL_SHA256: Optional[str] = (
    "ce87c8974b58c5cb9f71bec1f438f429dda92a85942d06771af09cc16d15f091"
)

DOWNLOAD_TIMEOUT_SECONDS = 60

# Distinguishes "use the configured checksum" from "deliberately skip
# verification". A plain None default would conflate the two, and would also
# bind MODEL_SHA256 at import time, so overriding it would have no effect.
_USE_CONFIGURED = object()


def model_filename(version: str = MODEL_VERSION) -> str:
    return f"{version}.keras"


def model_url(version: str = MODEL_VERSION, base_url: str = MODEL_BASE_URL) -> str:
    return f"{base_url.rstrip('/')}/{model_filename(version)}"


def _fetch(url: str, destination: Path) -> None:
    """Stream a URL to a path, raising ModelError with something readable."""
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with open(destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        raise ModelError(
            f"The server returned {exc.code} for:\n  {url}\n\n"
            "If this is a 404 the weights for this version may not be "
            "published yet. You can download them separately and place the "
            ".keras file in your model folder instead."
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelError(
            f"Could not reach:\n  {url}\n\n{exc.reason}\n\n"
            "Check your network connection, or download the weights separately "
            "and place the .keras file in your model folder."
        ) from exc
    except OSError as exc:
        raise ModelError(f"Could not write '{destination}': {exc}") from exc


def download_model(
    model_dir: Path,
    version: str = MODEL_VERSION,
    base_url: str = MODEL_BASE_URL,
    expected_sha256=_USE_CONFIGURED,
    force: bool = False,
) -> Path:
    """Download the QRS detector weights into ``model_dir``.

    Returns the path to the weights. An existing file is left alone unless
    ``force`` is set, so this is cheap to call before every run.

    Args:
        expected_sha256: Digest to verify against. Defaults to the configured
            ``MODEL_SHA256``; pass None to skip verification explicitly.

    Raises:
        ModelError: The download failed, or the file did not match its hash.
    """
    if expected_sha256 is _USE_CONFIGURED:
        expected_sha256 = MODEL_SHA256

    model_dir = Path(model_dir).expanduser()
    target = model_dir / model_filename(version)

    if target.is_file() and not force:
        log.debug("Weights already present at %s", target)
        return target

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelError(f"Could not create '{model_dir}': {exc}") from exc

    url = model_url(version, base_url)
    log.info("Downloading QRS detector weights from %s", url)

    # Written next to the destination rather than in the system temp folder,
    # so the rename below is on one filesystem and therefore atomic.
    handle, temp_name = tempfile.mkstemp(dir=str(model_dir), suffix=".part")
    os.close(handle)
    temp_path = Path(temp_name)

    try:
        _fetch(url, temp_path)

        if expected_sha256:
            actual = sha256_of(temp_path)
            if actual != expected_sha256:
                raise ModelError(
                    "The downloaded weights do not match the expected "
                    f"checksum, so they have been discarded.\n\n"
                    f"  expected  {expected_sha256}\n"
                    f"  actual    {actual}\n\n"
                    "This usually means the download was corrupted; trying "
                    "again often fixes it. If it persists, please open an "
                    "issue rather than using the file."
                )
        else:
            log.warning(
                "No expected checksum is configured for this model version, so "
                "the download could not be verified."
            )

        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    log.info("Weights saved to %s", target)
    return target


def ensure_model(
    model_dir: Path,
    model_path: Optional[Path] = None,
    auto_download: bool = True,
    **download_kwargs,
) -> Path:
    """Return usable weights, downloading them first if the folder is empty.

    Only a genuinely absent model triggers a download. If the folder already
    holds more than one candidate, ``find_model`` refuses to guess and adding
    another file would make that worse rather than better.
    """
    model_dir = Path(model_dir).expanduser()

    if model_path is None and auto_download:
        present = sorted(model_dir.glob(MODEL_GLOB)) if model_dir.is_dir() else []
        if not present:
            download_model(model_dir, **download_kwargs)

    return find_model(model_dir, model_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the QRS detector weights.",
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --model-dir /shared/models\n"
            "  %(prog)s --force\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model-dir", metavar="FOLDER",
        help="Where to save the weights. Overrides model_dir in config.yaml.",
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE",
        help="Path to a settings file. Defaults to config.yaml if present.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Download again even if the weights are already present.",
    )
    parser.add_argument(
        "--url", metavar="URL", default=MODEL_BASE_URL,
        help="Base URL to fetch from. Only needed for a mirror or a local copy.",
    )
    parser.add_argument(
        "--model-version", metavar="NAME", default=MODEL_VERSION,
        help=f"Weights to fetch. Default: {MODEL_VERSION}",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed progress messages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        config = load_config(
            path=Path(args.config) if args.config else None,
            overrides={"model_dir": args.model_dir},
        )
    except ConfigError as exc:
        print(f"\nConfiguration problem:\n\n{exc}\n", file=sys.stderr)
        return 2

    try:
        path = download_model(
            config.model_dir,
            version=args.model_version,
            base_url=args.url,
            force=args.force,
        )
    except ModelError as exc:
        print(f"\nCould not download the QRS detector:\n\n{exc}\n", file=sys.stderr)
        return 1

    print(f"Weights ready at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
