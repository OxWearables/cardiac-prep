# Contributing

Thanks for using this pipeline. Feedback from real datasets is genuinely useful.

## Questions, bugs, and ideas

Please [open an issue](../../issues) rather than emailing. It keeps the answer
where the next person will find it.

For a bug, include the error message and what you ran. For a suggestion, say
what you tried and what you saw.

## Before changing settings

Most tuning does not need a code change. Everything adjustable lives in
`config.yaml`, and the comments there explain what each setting does. If you
need a different threshold for your cohort, change it in your own copy - the
defaults are chosen for the population this was validated in and will not suit
every study.

## Pull requests

Please open an issue first so we can agree the change is wanted. Unprompted
pull requests are likely to sit unreviewed.

If a PR changes a default or a signal-processing step, include the evidence:
which recordings you tested on, what you measured, and how much it moved. This
is a scientific pipeline and other people's results depend on its defaults, so
changes to them need more than a promising first pass.

Run the tests before submitting:

```bash
pytest
```

## Scope

This pipeline does quality control, beat detection, and heart-rate and HRV
summaries from EDF recordings. It is not an experimentation framework, and
methods research - new artefact-removal approaches, alternative detectors - is
better done in your own fork. Do tell us how it goes.
