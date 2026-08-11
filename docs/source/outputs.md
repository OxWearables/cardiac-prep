# Outputs

A run writes an `output/` directory containing:

- **One folder per participant**, holding a `plots` directory with the PDF
  report and its component figures, and a `processed_data` directory with the
  detailed CSVs.
- **Top-level files**: an aggregated `df_info_summary.csv.gz` plus
  dataset-wide summary plots.

Every row of `df_info_summary.csv.gz` records `model_file` and `model_sha256`,
identifying exactly which detector produced those results. A filename is a
naming convention; a hash is a verifiable claim. Recordings that failed get
`failed = 1` and a `failure_reason` explaining why.

:::{note}
A full data dictionary listing every column, its type and its units is being
added. Until then, this page covers the structure and the columns most people
need.
:::

## `*_df_qc.csv.gz` - one row per 10-second segment

The index is seconds from the start of the recording, so row `0` covers
00:00:00-00:00:10, row `10` the next segment, and so on.

| Column group | Meaning |
|---|---|
| `device_worn`, `clipped_5perc_thrs`, `passed_initialQC` | Per-segment screening: was the device worn, was the signal saturated, did it pass basic signal checks |
| `passed_finalQC` | Whether the segment's heartbeats were good enough to trust |
| `N_beats`, `N_RR`, `rr_Cover`, `rr_sd`, `rr_outliers`, `qrs_snr`, `qrs_amp`, `rmssd` | Beat-level measurements |
| `*_raw` | The measured value, blank wherever QC failed |
| `*_imputed` | The gap-filled value. **Use these for analysis** |
| `*_isImputed` | `True` where that value was filled rather than measured |

### Why are the beat columns so often blank?

They are only computed for segments that were actually analysed. A segment is
skipped when the device was not worn, when the signal was too noisy or clipped
to pass initial checks, or when the detector found fewer than `n_beats_min`
beats. Separately, `RRm_raw` is deliberately blanked wherever `passed_finalQC`
is `False`.

So a blank means "not measurable here", not "data went missing". Lots of blanks
means a lot of non-wear time or poor signal quality, and is expected rather
than a bug.

To judge quality fairly, use `prop_ECG_worn_passed_finalQC` in the summary
file. It measures quality **among worn time only**, so it is not diluted by
periods when the device was off.

## `df_info_summary.csv.gz` - one row per recording

The dataset-level file. Alongside the model provenance and failure columns
above, it holds the per-participant summary metrics: resting, minimum, maximum
and mean heart rate, median daily RMSSD and its log-normalised form, hours
spent at each activity intensity, wear time, and the proportion of ECG passing
quality control.

## Plots

Written to each participant's `plots` folder during processing:

| File | Contents |
|---|---|
| `*_report.pdf` | Two-page summary report |
| `*_24hr_profile.png` | Typical 24-hour heart rate rhythm, for recordings over three days with all 24 hours covered |
| `*_activity_pie.png` | Share of time at each activity intensity |
| `*_hr_distribution.png` | Distribution of heart rate |
| `*_daily_bars.png` | Activity intensity per day |
| `*_ECGs_failedQC.pdf` | Example failing traces, written only when the pass rate falls below `qc_warn_below` |

`cardiac-prep inspect` adds `*_daily_heart_rate.png`, `*_hr_heatmap.png` and
`*_acc_heatmap.png` on demand, reading the saved CSVs rather than reprocessing.
