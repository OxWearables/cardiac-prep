# Quickstart

Put your `.edf` files into the `input_data` folder, then run these three
commands in order. Add `--help` to any of them to see its options.

## 1. Process your recordings

```text
cardiac-prep process
```

Reads every `.edf` file in `input_data` and writes results to `output`. About
30 seconds per file, and files are processed in parallel across your CPU cores.

To preview what would be processed without processing anything:

```text
cardiac-prep process --dry-run
```

To use folders elsewhere on your computer:

```text
cardiac-prep process --input /path/to/my/edfs --output /path/to/my/results
```

If one recording fails, the rest still process. The failure is named in the
final summary and recorded in `df_info_summary.csv.gz` with a `failure_reason`.

## 2. Summarise the whole dataset

```text
cardiac-prep summarise
```

Creates population-level plots across all participants.

## 3. Inspect one participant (optional)

```text
cardiac-prep inspect --list
```

```text
cardiac-prep inspect --subject NAME_FROM_THE_LIST
```

Add `--show` to open the plots in a window as well as saving them, and
`--kind` to pick just one (`daily`, `profile` or `heatmap`).

This reads the CSVs already written to `output`, so it is instant and does not
reprocess anything. The heart-rate-and-movement trace is not offered here
because the per-participant PDF report written during processing already
contains it.

## Running from a clone

A checkout has `process.py`, `summarise_dataset.py` and `plot_subject.py` at
the root. They forward to the same subcommands, so `python process.py` and
`cardiac-prep process` do exactly the same thing. Useful when you want to run
straight from a clone without installing.

`cardiac-prep --help` lists the subcommands, and `cardiac-prep <command>
--help` shows the options for one of them.

In Python code the package is imported as `cardiacprep`, without the hyphen,
since a hyphen is not valid in an import name.
