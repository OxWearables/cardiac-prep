# Interpreting results

Assumptions worth knowing about before drawing conclusions.

## Resting heart rate and HRV use a fixed clock window

Both are computed from movement below `sleep_threshold_mg` inside a fixed
overnight window, 21:00-09:00 by default.

Participants who sleep outside those hours - shift workers, or someone who
regularly sleeps 02:00-11:00 - will have these metrics computed from the wrong
hours, or not at all. You can move the window with `night_start_hour` and
`night_end_hour`, but it is one window applied to everybody in the dataset.

:::{admonition} Planned improvement
:class: seealso

Detecting each participant's main sleep period from their own accelerometer
data, instead of assuming a clock window.
:::

## Activity cut-points come from older adults

The milli-g boundaries separating activity intensities are from Etzkorn et al.
(2024): Zio XT chest-worn cut-points derived in 381 older adults in the ARIC
study, mapped from waist-worn ActiGraph GT3X.

The median age in that sample was 78. If your participants are substantially
younger, these boundaries may not describe them well, and the hours-per-
intensity columns should be interpreted cautiously. They are adjustable via
`activity_very_light_mg`, `activity_light_mg` and `activity_moderate_mg`.

## HRV is normalised by taking a logarithm

The pipeline reports both `rmssd` and a log-normalised form. Taking the natural
logarithm of RMSSD removes much of the mathematical dependence on a person's
average heart rate, which makes comparisons between people fairer.

For the reasoning, see [Should we normalize HRV by heart
rate?](https://marcoaltini.substack.com/p/should-we-normalize-hrv-by-heart)
by Marco Altini.

## Signal processing is tuned for the bundled detector

The filtering, clipping and scaling defaults match the conditions the bundled
QRS detector was trained under. They are all adjustable - see the advanced
section of the [configuration reference](configuration.md) - but changing them
changes what the model sees, so check the effect on detection before trusting
the results.

Two that are worth knowing about specifically:

- **`mains_hz` defaults to 50 Hz.** In North America and parts of Asia the
  mains frequency is 60 Hz, and leaving this at 50 means mains interference is
  not removed. Set it correctly for where the data was collected.
- **Scaling is per-segment.** Each 10-second segment is standardised on its own
  mean and standard deviation, which adapts to noisy stretches but means a
  segment dominated by motion artefact will have its heartbeats scaled down
  relative to that artefact.

## Quality control is deliberately strict

Segments are discarded rather than salvaged. This means a recording with a lot
of movement will show a low pass rate, and that is the intended behaviour: the
metrics that survive are the ones worth trusting.

Judge a recording by `prop_ECG_worn_passed_finalQC`, which is computed among
worn time only. `prop_ECG_passed_finalQC` includes non-wear time in its
denominator and will look poor for anyone who took the device off.
