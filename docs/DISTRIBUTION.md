# Distribution contents

VITAL is distributed as a source ZIP, a Python source distribution, and an
installable wheel. Model weights are hosted separately on
[Hugging Face](https://huggingface.co/datasets/vital-wam/vital/tree/main).
Datasets remain available from their original providers.

The source packages include Python code, tests, documentation, configuration,
the public NAVSIM navtest split, project figures, the Apache 2.0 project
license, and third-party license texts. The wheel
contains the runtime package, public split, weight manifest, and license
notices. Weights, datasets, metric caches and evaluation outputs are excluded.

Source archive ownership and timestamps are normalized. Source packaging uses
an explicit file allowlist and rejects symlinks, unsupported file types and
specified disallowed strings. Repository history, local environments and editor
state are excluded. These checks concern release contents; they do not change
model tensors or the evaluation protocol.

Third-party project names, dependency URLs, public split identifiers and
license attribution are retained. Build and source-audit commands are
documented in [Development](DEVELOPMENT.md).
