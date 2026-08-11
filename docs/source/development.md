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
src/cardiacprep/
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

## Releasing

Releases are published to PyPI automatically by
`.github/workflows/publish.yml`. Nobody uploads by hand, and there is no API
token stored anywhere.

### Making a release

1. Bump `__version__` in `src/cardiacprep/__init__.py`. That is the only place
   a version number is written; `pyproject.toml` reads it from there.
2. Commit and push.
3. On GitHub, go to Releases, draft a new release with a tag matching the
   version, prefixed with `v` - e.g. `v1.1.0` for version `1.1.0`.
4. Publish the release. The workflow builds the distributions, checks them,
   and uploads to PyPI.

The workflow refuses to publish if the tag does not match the packaged
version, because PyPI never allows a version to be re-uploaded once taken.

To rehearse without touching the real index, run the workflow manually from
the Actions tab with the target set to `testpypi`. This needs a matching
publisher registered on `test.pypi.org`, which is an entirely separate service
from `pypi.org` with its own accounts.

### How publishing is authenticated

The workflow uses **PyPI Trusted Publishing** rather than an API token. When
it runs, GitHub mints a short-lived signed token asserting the repository,
workflow filename and environment; PyPI checks that against a registered
publisher and, if it matches, issues an upload token valid for minutes.

The practical consequence for maintenance: **publishing rights follow the
repository, not a person.** Anyone able to publish a release in this repo can
publish to PyPI, and PyPI ownership can change hands without reissuing or
revoking any credential. There is no secret to hand over.

### Configuring the trusted publisher

Needed once, by whoever owns the project on PyPI, under
Manage project → Publishing. Before the first release the project does not yet
exist, so it is registered as a *pending* publisher from the account settings
instead, and converts automatically on first upload.

| Field | Value |
|---|---|
| PyPI Project Name | `cardiacprep` |
| Owner | `OxWearables` |
| Repository name | `cardiac-prep` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Note that the project name has no hyphen while the repository does. Workflow
name is the filename, not the workflow's display name. The environment field
is optional: leaving it blank matches any environment, which is simpler if you
cannot create environments on the repository.

### Ownership

The account that registers the pending publisher becomes the project owner
when the first release uploads. PyPI supports several owners, and this project
should have more than one so that it does not depend on any single person
remaining in post - add the others under Manage project → Collaborators, with
the Owner role.

Keep the `maintainers` field in `pyproject.toml` in step with whoever is
actually looking after releases.

## Contributing

See [CONTRIBUTING.md](https://github.com/OxWearables/cardiac-prep/blob/main/CONTRIBUTING.md).
In short: open an issue before a pull request, and if a change alters a default
or a signal-processing step, include the evidence - which recordings you tested
on, what you measured, and how much it moved.
