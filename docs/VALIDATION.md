# Validation

This document distinguishes source checks, checkpoint integrity, and numerical
inference/evaluation checks. CPU tests and small-scene comparisons do not
establish a new full-benchmark result.

## Source checks

Run from the source repository after installing the package:

```bash
python -m unittest discover -s tests -v
```

The suite covers metrics, public split size, inference output ordering and
coverage checks, installed command entry points, packaged resources, checksum
rejection, and archive safety. These checks use synthetic inputs and require
no dataset or GPU. When `checkpoints/vital.pt` is present, an additional test
loads its trajectory decoder and residual policy and checks output shapes and
finite values. That test is explicitly skipped without the checkpoint.

The checks have been run on Python 3.12, PyTorch 2.10.0 and torchvision 0.25.0.
Installed wheel commands have also been checked outside the source directory.
The CI configuration targets Python 3.10 and 3.12; this is separate from the
reference GPU environment documented in the README.

## Checkpoint integrity

```bash
vital-verify-weights checkpoints/vital.pt --upstream-dir checkpoints/upstream
```

This verifies the VITAL file against the packaged size/SHA256 manifest and
checks its schema, NFE=1 and seed=0 settings, and pinned upstream identifiers.
With `--upstream-dir`, it also verifies the two public model files. The pinned
VITAL download's HTTP metadata and Hugging Face LFS SHA256 match the manifest;
these remote metadata checks are distinct from hashing a complete local file.
Exact download revisions and the VITAL checksum are listed in [Weights](WEIGHTS.md).

## Recorded numerical equivalence checks

The following comparisons were recorded during preparation of the inference
release:

- Four raw NAVSIM scenes were run with the single-file VITAL checkpoint and
  the separately loaded public backbone/VAE. Predictions matched reference
  NFE=1 trajectories bitwise (maximum absolute difference 0), and the official
  v1 scorer completed successfully on those predictions.
- Sixteen additional scenes were sampled across the sorted token inventory,
  independently of scores. A separately installed wheel produced trajectories
  identical to the reference exports. This check used an earlier split-file
  checkpoint layout; the single-file path was rechecked on the four scenes
  above, not on all 20 scenes.
- Metric caches were generated from raw logs and maps for all 20 scenes.
  Official v1 PDMS and its collision, drivable-area, progress, TTC and comfort
  components matched the corresponding reference results.
- All 891 tensors in the distributed VITAL checkpoint were compared
  byte-for-byte with the source inference checkpoint, including dtype and
  shape. Public backbone/VAE tensors were also compared with the reference
  inference weights after BF16 conversion.

These checks establish equivalence on the specified small samples. They are
not a fresh evaluation of all 12,146 navtest scenes, and do not validate NAVSIM
v2 or NAVHARD. The reported full-split PDMS reference is given in the README.
The current documentation and packaging checks do not rerun GPU inference;
reproducing it requires the separately downloaded weights, dataset and devkits.
