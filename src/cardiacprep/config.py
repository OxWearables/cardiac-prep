"""Configuration for the EDF preprocessing pipeline.

All tunable settings live in a single ``config.yaml`` at the project root, so
running the pipeline never requires editing Python source. Every setting has a
sensible default, which means ``config.yaml`` is optional - delete it and the
pipeline still runs with the published defaults below.

Precedence, lowest to highest:

    dataclass defaults  ->  config.yaml  ->  command-line flags
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_FILENAME = "config.yaml"

# Settings that name a location on disk and are coerced to pathlib.Path.
_PATH_FIELDS = ("input_dir", "output_dir", "model_dir", "model_path")


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or self-inconsistent.

    Messages are written for someone who has never read the source, since a
    bad config file is the most likely thing to go wrong for a new user.
    """


@dataclass(frozen=True)
class Config:
    """Every tunable setting in the pipeline.

    Frozen so that a worker process cannot mutate settings mid-run, and so the
    object can be handed to ``multiprocessing.Pool`` safely.
    """

    # Where things live
    input_dir: Path = Path("./input_data")
    output_dir: Path = Path("./output")
    model_dir: Path = Path("./models")
    # Leave model_path unset to auto-discover the single .keras file in model_dir.
    model_path: Optional[Path] = None

    # Fetch the weights automatically when model_dir holds none.
    auto_download_model: bool = True

    # How much of the machine to use. None means "all cores but one".
    n_processes: Optional[int] = None

    # Signal segmentation. The bundled QRS detector was trained on 10-second
    # segments of 250 Hz ECG; changing these requires a retrained model.
    segment_seconds: int = 10
    fs_expected: int = 250
    chunk_hours: int = 24

    # ECG amplitude clipping, in millivolts. ecg_clip_mv is a hard limit applied
    # before filtering; a segment with >5% of samples beyond it fails QC.
    # ecg_clip_margin_mv widens the adaptive band used just before the detector,
    # which sits at the median per-segment peak plus this margin.
    ecg_clip_mv: float = 4.0
    ecg_clip_margin_mv: float = 1.0

    # ECG quality control
    rr_cover_min: float = 0.75      # min fraction of a segment covered by valid RR intervals
    n_beats_min: int = 5            # min detected beats for a segment to be scored
    rr_min_ms: int = 250            # shortest physiologically plausible RR interval
    rr_max_ms: int = 2500           # longest physiologically plausible RR interval
    max_rr_outliers: int = 1        # max intervals >1.8x median before a segment fails
    qc_warn_below: float = 0.75     # warn and dump example ECGs below this pass rate

    # Activity intensity cut-points in milli-g, from Etzkorn et al. (2024),
    # Zio XT chest-worn MAD cut-points derived in n=381 older adults (ARIC).
    activity_very_light_mg: float = 9.04
    activity_light_mg: float = 28.19
    activity_moderate_mg: float = 58.08

    # Movement below this counts as rest when inside the rest window.
    sleep_threshold_mg: float = 9.04

    # Fixed clock window used to identify overnight rest. This assumes a
    # conventional sleep schedule and misclassifies shift workers and anyone
    # who habitually sleeps outside it.
    night_start_hour: int = 21
    night_end_hour: int = 9

    # ---- Advanced -----------------------------------------------------------
    # Signal-processing internals. The defaults match the conditions the bundled
    # QRS detector was trained under, so changing anything below alters what the
    # model sees and should be checked against the detector's output.

    # Mains interference removed by a notch filter. 50 Hz across most of the
    # world, 60 Hz in North America and parts of Asia.
    mains_hz: float = 50.0
    mains_notch_q: float = 30.0

    # ECG bandpass corners in Hz, applied after the notch.
    ecg_bandpass_low_hz: float = 2.0
    ecg_bandpass_high_hz: float = 40.0

    # Per-segment noise gate. Variance outside this range, or a peak-to-peak
    # below the minimum, marks a segment as too noisy or too flat to score.
    ecg_var_min: float = 0.0001
    ecg_var_max: float = 2.0
    ecg_min_ptp_mv: float = 0.025

    # Probability above which the detector calls a sample part of a QRS complex.
    # Lower detects more beats at the cost of more false positives.
    qrs_threshold: float = 0.5

    # An R-R interval longer than this multiple of the segment median counts as
    # an outlier, most often a missed beat. See also max_rr_outliers.
    rr_outlier_factor: float = 1.8

    # Accelerometer saturation in milli-g. 4000 suits a +/-4 g device.
    acc_clip_mg: float = 4000.0

    # Auto-calibrate the accelerometer against gravity before use.
    acc_calibrate: bool = True

    # Width, in samples, of the median filter that removes baseline steps from
    # the acceleration magnitude.
    acc_median_filter_samples: int = 120

    # Longest gap, in seconds, filled by linear interpolation. Longer gaps fall
    # back to the average for that time of day.
    impute_gap_max_s: int = 600

    @property
    def activity_thresholds(self) -> Dict[str, float]:
        """Cut-points in the dict form the plotting and reporting code expects."""
        return {
            "very_light": self.activity_very_light_mg,
            "light": self.activity_light_mg,
            "moderate": self.activity_moderate_mg,
        }

    @property
    def segment_samples(self) -> int:
        """Samples per analysis segment (2500 = 10 s at 250 Hz)."""
        return int(self.segment_seconds * self.fs_expected)

    @property
    def chunk_seconds(self) -> int:
        """Seconds of signal loaded into memory at a time."""
        return int(self.chunk_hours * 3600)

    @property
    def resolved_n_processes(self) -> int:
        """Worker count, leaving one core free unless explicitly configured."""
        if self.n_processes is not None:
            return self.n_processes
        import os

        return max(1, (os.cpu_count() or 1) - 1)

    def validate(self) -> None:
        """Check settings are internally consistent, or explain what is wrong."""
        if not (self.activity_very_light_mg < self.activity_light_mg < self.activity_moderate_mg):
            raise ConfigError(
                "Activity thresholds must increase: "
                f"very_light ({self.activity_very_light_mg}) < "
                f"light ({self.activity_light_mg}) < "
                f"moderate ({self.activity_moderate_mg}). "
                "Check activity_*_mg in your config file."
            )

        if self.rr_min_ms >= self.rr_max_ms:
            raise ConfigError(
                f"rr_min_ms ({self.rr_min_ms}) must be less than "
                f"rr_max_ms ({self.rr_max_ms})."
            )

        if not 0.0 <= self.rr_cover_min <= 1.0:
            raise ConfigError(
                f"rr_cover_min must be a fraction between 0 and 1, got {self.rr_cover_min}."
            )

        if not 0.0 <= self.qc_warn_below <= 1.0:
            raise ConfigError(
                f"qc_warn_below must be a fraction between 0 and 1, got {self.qc_warn_below}."
            )

        for name in ("night_start_hour", "night_end_hour"):
            hour = getattr(self, name)
            if not 0 <= hour <= 23:
                raise ConfigError(f"{name} must be an hour from 0 to 23, got {hour}.")

        if self.night_start_hour == self.night_end_hour:
            raise ConfigError(
                "night_start_hour and night_end_hour are identical "
                f"({self.night_start_hour}), which selects no rest period at all."
            )

        if self.n_processes is not None and self.n_processes < 1:
            raise ConfigError(f"n_processes must be at least 1, got {self.n_processes}.")

        if self.segment_seconds < 1:
            raise ConfigError(f"segment_seconds must be at least 1, got {self.segment_seconds}.")

        if self.chunk_hours < 1:
            raise ConfigError(f"chunk_hours must be at least 1, got {self.chunk_hours}.")

        for name in (
            "ecg_clip_mv",
            "ecg_clip_margin_mv",
            "mains_hz",
            "mains_notch_q",
            "acc_clip_mg",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"{name} must be greater than 0, got {value}.")

        # Filters are designed against the sampling rate, so any corner at or
        # above Nyquist makes scipy fail with a message that does not name the
        # setting responsible.
        nyquist = self.fs_expected / 2
        for name in ("mains_hz", "ecg_bandpass_low_hz", "ecg_bandpass_high_hz"):
            value = getattr(self, name)
            if value >= nyquist:
                raise ConfigError(
                    f"{name} ({value} Hz) must be below half the sampling rate "
                    f"({nyquist} Hz for fs_expected={self.fs_expected})."
                )

        if not 0 < self.ecg_bandpass_low_hz < self.ecg_bandpass_high_hz:
            raise ConfigError(
                f"ecg_bandpass_low_hz ({self.ecg_bandpass_low_hz}) must be greater "
                f"than 0 and less than ecg_bandpass_high_hz "
                f"({self.ecg_bandpass_high_hz})."
            )

        if not 0 <= self.ecg_var_min < self.ecg_var_max:
            raise ConfigError(
                f"ecg_var_min ({self.ecg_var_min}) must be at least 0 and less "
                f"than ecg_var_max ({self.ecg_var_max})."
            )

        if self.ecg_min_ptp_mv < 0:
            raise ConfigError(
                f"ecg_min_ptp_mv cannot be negative, got {self.ecg_min_ptp_mv}."
            )

        if not 0 < self.qrs_threshold < 1:
            raise ConfigError(
                f"qrs_threshold must be a probability between 0 and 1 "
                f"(exclusive), got {self.qrs_threshold}."
            )

        if self.rr_outlier_factor <= 1:
            raise ConfigError(
                f"rr_outlier_factor must be greater than 1, got "
                f"{self.rr_outlier_factor}. A value of 1 or less would mark "
                "roughly half of all normal intervals as outliers."
            )

        if self.acc_median_filter_samples < 1:
            raise ConfigError(
                "acc_median_filter_samples must be at least 1, got "
                f"{self.acc_median_filter_samples}."
            )

        if self.impute_gap_max_s < 0:
            raise ConfigError(
                f"impute_gap_max_s cannot be negative, got {self.impute_gap_max_s}."
            )

        if self.sleep_threshold_mg < 0:
            raise ConfigError(
                f"sleep_threshold_mg cannot be negative, got {self.sleep_threshold_mg}."
            )


def _coerce(data: Dict[str, Any]) -> Dict[str, Any]:
    """Turn raw YAML values into the types Config expects."""
    coerced = dict(data)
    for name in _PATH_FIELDS:
        value = coerced.get(name)
        if value is not None:
            coerced[name] = Path(str(value)).expanduser()
    return coerced


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"Could not read config file '{path}': {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Config file '{path}' is not valid YAML.\n{exc}\n"
            "Common causes: tabs used for indentation (use spaces), or a "
            "missing space after a colon."
        ) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file '{path}' must contain a mapping of setting names to "
            f"values, but the top level is a {type(data).__name__}."
        )
    return data


def load_config(
    path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Build a validated Config from a YAML file plus command-line overrides.

    Args:
        path: Explicit config file. If None, ``config.yaml`` in the current
            directory is used when present; otherwise defaults apply.
        overrides: Values that win over the file. Entries whose value is None
            are ignored, so unset CLI flags do not clobber file settings.

    Raises:
        ConfigError: The file is missing, malformed, contains unknown settings,
            or the resulting configuration is self-inconsistent.
    """
    data: Dict[str, Any] = {}

    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: '{path}'")
        data = _read_yaml(path)
    else:
        default_path = Path(DEFAULT_CONFIG_FILENAME)
        if default_path.is_file():
            data = _read_yaml(default_path)

    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) in config file: {', '.join(sorted(unknown))}.\n"
            f"Valid settings are: {', '.join(sorted(known))}."
        )

    try:
        config = Config(**_coerce(data))
    except TypeError as exc:
        raise ConfigError(f"Could not apply config file settings: {exc}") from exc

    if overrides:
        applied = {k: v for k, v in overrides.items() if v is not None}
        bad = set(applied) - known
        if bad:
            raise ConfigError(f"Unknown override(s): {', '.join(sorted(bad))}")
        config = replace(config, **_coerce(applied))

    config.validate()
    return config
