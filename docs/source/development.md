# Development

## Setup

```text
pip install -e ".[dev]"
```

```text
pytest
```

Tests use synthetic data only, so no recordings and no model weights are
needed. GitHub Actions runs the tests and `ruff check .` on Python 3.9 to 3.12
for every push.

## Layout

```text
src/edfproc/
  entry.py          dispatches the process / summarise / inspect subcommands
  cli.py            the process subcommand
  dataset_summary.py  the summarise subcommand
  subject_plots.py  the inspect subcommand
  config.py         every tunable setting, loaded from config.yaml
  read_utils.py     reading EDF, filtering, per-segment screening
  proc_utils.py     downsampling, beat detection, quality metrics, imputation
  proc_edf.py       orchestration for one recording
  plot_utils.py     figures used in the PDF report
  subject_data.py   loading processed CSVs back in
  model_utils.py    locating and hashing the detector weights
  io_utils.py       atomic CSV writes
```

The three scripts at the repository root - `process.py`,
`summarise_dataset.py`, `plot_subject.py` - are thin shortcuts that let a fresh
clone run without an install step. Each forwards to the matching subcommand.

## Building the documentation

```text
pip install -r docs/requirements.txt
```

```text
sphinx-build -b html docs/source docs/_build/html
```

Then open `docs/_build/html/index.html`.

The [configuration reference](configuration.md) is **generated** at build time
from the comments in `config.yaml` by `docs/source/_generate.py`. Do not edit
`docs/source/configuration.md` - it is overwritten on every build and ignored
by git. Edit the comments in `config.yaml` instead.

`tests/test_docs_config.py` fails if a setting exists in the `Config` dataclass
but is absent from `config.yaml`, which is what stops the two drifting apart.

The docs build does not install the package. Importing it would pull in
TensorFlow, so `docs/source/conf.py` puts `src` on the path and mocks the heavy
third-party imports for autodoc instead.

## Contributing

See [CONTRIBUTING.md](https://github.com/OxWearables/cardiac-prep/blob/main/CONTRIBUTING.md).
In short: open an issue before a pull request, and if a change alters a default
or a signal-processing step, include the evidence - which recordings you tested
on, what you measured, and how much it moved.
