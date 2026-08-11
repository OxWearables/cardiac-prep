# Models

The QRS detector weights are **not** distributed through this repository. They
are downloaded on demand from the group's file server, verified against a known
SHA-256, and saved here.

You do not normally need to do anything: the first run fetches them. To fetch
them ahead of time:

```
cardiac-prep download
```

Expected filename:

```
models/QRS_detector_125Hz_080525.keras
```

Set `auto_download_model: false` in `config.yaml` to turn the automatic fetch
off, for example on a machine with no internet access, and copy the file here
by hand instead.

Everything else in this folder is ignored by git.
