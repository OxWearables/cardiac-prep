"""Locating the QRS detector weights and recording which weights were used.

The weights are distributed separately from this repository, so the pipeline
has to find whatever the user downloaded rather than assume a fixed filename.
Discovery is by extension, but the exact file and its SHA-256 are recorded in
every output row - a filename is a naming convention, whereas a hash is a
verifiable claim about which model produced a given result.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional

MODEL_GLOB = "*.keras"

_DOWNLOAD_HINT = (
    "Download the QRS detector weights and place the .keras file in that "
    "folder. See models/README.md for details."
)


class ModelError(Exception):
    """Raised when the QRS detector weights cannot be located unambiguously."""


def find_model(model_dir: Path, model_path: Optional[Path] = None) -> Path:
    """Return the QRS detector weights file to load.

    An explicit ``model_path`` is used as-is. Otherwise ``model_dir`` is
    searched for a single ``.keras`` file.

    Refusing to choose between multiple candidates is deliberate: silently
    loading the wrong weights would corrupt every downstream result while the
    pipeline appeared to succeed.

    Raises:
        ModelError: No weights found, or more than one candidate.
    """
    if model_path is not None:
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise ModelError(
                f"Model file not found: '{model_path}'\n"
                "Check the model_path setting in your config file, or unset it "
                "to search the models folder automatically."
            )
        return model_path

    model_dir = Path(model_dir).expanduser()
    if not model_dir.is_dir():
        raise ModelError(f"Model folder not found: '{model_dir}'\n{_DOWNLOAD_HINT}")

    candidates = sorted(model_dir.glob(MODEL_GLOB))

    if not candidates:
        raise ModelError(
            f"No '{MODEL_GLOB}' file found in '{model_dir}'.\n{_DOWNLOAD_HINT}"
        )

    if len(candidates) > 1:
        names = "\n  ".join(p.name for p in candidates)
        raise ModelError(
            f"Found {len(candidates)} model files in '{model_dir}':\n  {names}\n"
            "Refusing to guess which one to use. Either remove the ones you do "
            "not want, or set model_path in your config file to choose "
            "explicitly."
        )

    return candidates[0]


def sha256_of(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file in chunks so large weights do not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(path: Path) -> Dict[str, str]:
    """Provenance columns identifying the weights used for a run.

    Written into every row of the summary output so a result can always be
    traced back to the exact model that produced it, and so a third party can
    verify their copy matches by hashing it.
    """
    path = Path(path)
    return {
        "model_file": path.name,
        "model_sha256": sha256_of(path),
    }
