# Quickstart

Put your `.edf` files into the `input_data` folder, then run these three
commands in order. Add `--help` to any of them to see its options.

## 1. Process your recordings

```text
python process.py
```

Reads every `.edf` file in `input_data` and writes results to `output`. About
30 seconds per file, and files are processed in parallel across your CPU cores.

To preview what would be processed without processing anything:

```text
python process.py --dry-run
```

To use folders elsewhere on your computer:

```text
python process.py --input /path/to/my/edfs --output /path/to/my/results
```

If one recording fails, the rest still process. The failure is named in the
final summary and recorded in `df_info_summary.csv.gz` with a `failure_reason`.

## 2. Summarise the whole dataset

```text
python summarise_dataset.py
```

Creates population-level plots across all participants.

## 3. Inspect one participant (optional)

```text
python plot_subject.py --list
```

```text
python plot_subject.py --subject NAME_FROM_THE_LIST
```

Add `--show` to open the plots in a window as well as saving them, and
`--kind` to pick just one (`daily`, `profile` or `heatmap`).

This reads the CSVs already written to `output`, so it is instant and does not
reprocess anything. The heart-rate-and-movement trace is not offered here
because the per-participant PDF report written during processing already
contains it.

## The installed command

The three scripts above are shortcuts. Installing the package provides one
command with the same three steps as subcommands, runnable from any folder:

```text
pip install -e .
```

```text
cardiac-prep process      # same as python process.py
cardiac-prep summarise    # same as python summarise_dataset.py
cardiac-prep inspect      # same as python plot_subject.py
```

`cardiac-prep --help` lists the subcommands, and `cardiac-prep <command>
--help` shows the options for one of them. `edfproc` works as an alias, since
that is still the name the package is imported under.
