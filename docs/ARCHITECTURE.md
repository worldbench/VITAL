# Architecture

VITAL is an installable Python inference and evaluation project. Its package
uses a `src/` layout so tests and command-line tools exercise the installed
package rather than accidentally importing files from the current directory.

```text
src/vital/
  cli/             argument parsing and command dispatch
  services/        inference, metric-cache generation, official scoring
  models/          VITAL policy, adapter, trajectory decoder, residual policy
  data/            NAVSIM reader and versioned public split resource
  evaluation/      trajectory metrics and NAVSIM v1 component aggregation
  infrastructure/  downloads and validated archive extraction
  resources/       packaged checkpoint manifest
  vendor/          isolated third-party inference primitives
  weights.py       checkpoint schema, upstream pins and checksum validation
tests/             CPU regression tests; optional checkpoint-dependent tests
tools/             checkpoint export and source packaging
configs/           reference inference contract
docs/              architecture, development, validation and release guidance
THIRD_PARTY_LICENSES/  upstream source license texts
checkpoints/       local weights; excluded from source distributions
data/              local datasets and caches; excluded from source distributions
```

Dependencies flow from `cli` to `services`, then to `models`, `data`,
`evaluation`, `weights`, and the isolated upstream implementation in `vendor`.
Core model code does not depend on CLI parsing or repository paths. The
download commands share standard-library I/O in `infrastructure`. Build and
checkpoint-export tools remain outside the runtime package.

All eight public commands are declared in `pyproject.toml`; equivalent
`python -m vital.cli.<module>` entry points are available. Their `--help`
paths do not initialize models or import the optional NAVSIM devkits.
Runtime output and input paths are relative to the caller's working directory
unless absolute paths are supplied. Packaged resources are read from the
installed distribution, independently of that directory.

The inference contract remains NFE=1, seed=0, current front-camera RGB and an
eight-dimensional ego state, producing 40 poses at 0.1-second intervals.
The model configuration and normalization buffers come from `vital.pt`.
`configs/inference.yaml` documents this contract; it is not an automatically
loaded configuration file. Runtime paths and device are selected by CLI flags.

The standard installation supports inference and standalone metric aggregation.
Official metric-cache creation and scoring additionally require the pinned
NAVSIM/nuPlan revisions and `requirements-scoring.txt` described in the README.
Model and data downloads are explicit operations, never installation hooks.
