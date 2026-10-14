# Development

Use Python 3.10 or newer. The reference GPU environment uses Python 3.12;
install the PyTorch build shown in the README before installing development
dependencies. Run these commands from the repository root:

```bash
python -m pip install -e '.[dev]'
make check
```

`make check` runs static error checks, CPU tests and the source release audit.
The audit also rejects unfinished documentation and broken local document links.
The public README's roadmap checklist records planned releases and is allowed;
inline draft markers and missing links still fail the audit. The source
archives include both README figures and the project license.
Tests do not download data. The checkpoint-dependent test is explicitly
skipped when `checkpoints/vital.pt` is absent. Download the checkpoint with
`vital-download-weights` to enable that test. Full inference and official
scoring additionally require the public backbone/VAE, NAVSIM data and devkits.

Ruff checks first-party code for syntax and undefined/unused names. Upstream
source formatting is left intact. Keep adaptations in `src/vital/vendor`
minimal and preserve their attribution. Numerical changes require comparison
with reference trajectories and official score components.

## Build distributions

```bash
make build
make release
```

`make build` produces the wheel and source tarball in `dist/`. It fixes wheel
timestamps and removes local owner names, uid/gid, timestamps and PAX headers
from the source tarball. Use this build command for distributable artifacts.

`make release` runs the checks and creates `dist/vital-source.zip` from an
explicit allowlist. It refuses to replace an existing ZIP. For another output
path, use the packaging tool directly:

```bash
python tools/release.py --output dist/vital-source-checked.zip
```

To run the source audit without creating an archive:

```bash
python tools/release.py
```

If weights are present, `--weights checkpoints/vital.pt` also verifies their
checksum/schema and scans non-tensor metadata. No local weights are required
for source-only builds. See [Distribution contents](DISTRIBUTION.md) and
[Validation](VALIDATION.md) for artifact scope and the checks performed.
