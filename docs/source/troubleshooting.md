# Troubleshooting

## Common messages

| What you see | What to do |
|---|---|
| `command not found: conda` | Install [Miniforge](https://conda-forge.org/download/), then close and reopen your terminal |
| `No module named ...` | Your environment is not active. Run `conda activate cardiacprep` |
| `No '*.keras' file found` | The detector model is missing. Redo step 4 of [Installation](installation.md) |
| `Found 2 model files` | Remove the spare `.keras` file from `models` |
| `No .edf files found` | Your recordings are not in `input_data`, or do not end in `.edf` |
| `Configuration problem: ...` | A mistake in `config.yaml`. The message names the setting at fault |
| One file failed | The others still process. Rerun with `--jobs 1` to see the error clearly |

## A recording failed

Processing continues past a failed recording. The failure is named in the final
summary, and `df_info_summary.csv.gz` records `failed = 1` with a
`failure_reason` for that row.

To see the full error, rerun that one recording without parallelism:

```text
cardiac-prep process --jobs 1 --verbose
```

Parallel workers make tracebacks interleave, so `--jobs 1` is almost always the
first step in diagnosing anything.

## Most segments are failing quality control

If fewer than `qc_warn_below` of worn-time segments pass, the pipeline prints a
warning and saves example failing traces to `*_ECGs_failedQC.pdf` in the
participant's `plots` folder. Look at those first - they usually show whether
the problem is electrode contact, movement, or interference.

Things worth checking, in order:

1. **Mains interference.** If the data was collected outside Europe, set
   `mains_hz: 60`. A 50 Hz notch does nothing for 60 Hz hum.
2. **Amplitude.** If the trace looks saturated, the device may be clipping
   before the pipeline sees it. `ecg_clip_mv` controls the threshold at which a
   segment is rejected as saturated.
3. **Detection sensitivity.** `qrs_threshold` is the detector's probability
   cutoff, 0.5 by default. Lowering it finds more beats at the cost of more
   false positives.

Change one at a time, then reprocess a single recording and look at the result
with `cardiac-prep inspect`.

## Results look wrong rather than missing

- **Resting heart rate is absent or implausible.** Check the participant sleeps
  inside `night_start_hour` to `night_end_hour`. See
  [Interpreting results](interpreting.md).
- **Activity hours look off.** The cut-points were derived in older adults; see
  the same page.
- **Lots of blank columns.** Usually expected, and explained under "Why are the
  beat columns so often blank?" in [Outputs](outputs.md).

## Getting help

Please [open an issue](https://github.com/OxWearables/cardiac-prep/issues)
rather than emailing, so the answer is where the next person will find it.
Include the error message and the command you ran.
