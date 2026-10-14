# Weights

The VITAL checkpoint is available from the Hugging Face dataset repository
[vital-wam/vital](https://huggingface.co/datasets/vital-wam/vital/tree/main).
Despite the repository type, it distributes **model weights only**, not data.
The source checkout and source distributions do not contain the checkpoint.

## Download for inference

After installing VITAL as described in the README, run:

```bash
vital-download-weights --output checkpoints/vital.pt
vital-verify-weights checkpoints/vital.pt
```

The command uses the pinned public URL in the packaged manifest by default.
No Hugging Face CLI installation or URL environment variable is needed for
this public file. It verifies SHA256, size and checkpoint structure before
installing the file, and refuses to overwrite an existing destination. If a
local file already exists, verify it with the second command instead.

| Field | Value |
| --- | --- |
| Repository | `vital-wam/vital` (dataset repository) |
| File | `vital.pt` |
| Revision | `046f00531d306a287f19f0f825108761728cff38` |
| Size | 2,053,505,801 bytes (approximately 2.054 GB / 1.91 GiB) |
| SHA256 | `9ddbc0d4d5b9ce2dcdf48db16f2d39bdcb8f72a86c4701d9f21e9b966094efa9` |

[Download the pinned checkpoint](https://huggingface.co/datasets/vital-wam/vital/resolve/046f00531d306a287f19f0f825108761728cff38/vital.pt).
The `/tree/main` link is a browser page, not a direct file URL. A manual
alternative is:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/vital.pt \
  'https://huggingface.co/datasets/vital-wam/vital/resolve/046f00531d306a287f19f0f825108761728cff38/vital.pt'
vital-verify-weights checkpoints/vital.pt
```

To use a mirror or a local copy, pass `--url` or set `VITAL_WEIGHTS_URL`.
Resolution order is `--url`, then `VITAL_WEIGHTS_URL`, then the manifest URL.
The selected file must still match the packaged checksum. For a checkpoint
from another release, supply its corresponding manifest with `--manifest`.

## Contents and other downloads

The single file contains the action policy, F0 adapters, proprioception
encoder, trajectory decoder, residual policy, normalization/basis/constant
prompt buffers, model configuration, pinned public dependency identifiers,
and third-party notices. It contains no dataset or optimizer state.

The public visual backbone and WAN VAE are separate downloads:

```bash
vital-download-upstream --output checkpoints/upstream --accept-model-terms
vital-verify-weights checkpoints/vital.pt --upstream-dir checkpoints/upstream
```

Review the model terms linked in README section 2 before running that
command. Use the pinned trained SimWAM visual expert; vanilla WAN DiT is not
interchangeable. Users obtain NAVSIM logs, cameras and maps from the official
providers and generate metric caches locally, following README sections 3–5.

## Reuse existing public weights

If the public files are already stored elsewhere, pass their locations to
inference instead of downloading them again:

```bash
vital-infer --data-root data/navsim --weights checkpoints/vital.pt \
  --simwam-weights /path/to/SimWAM.pt --wan-vae /path/to/Wan2.2_VAE.pth \
  --limit 4 --output outputs/smoke_predictions.npz
```

The required public revisions and SHA256 digests are:

| File | Revision | SHA256 |
| --- | --- | --- |
| `SimWAM.pt` | `a862a0dc8576be777872c128266a5573d8ae3c40` | `1d05ec214008efa15c5b2fc5029e139fecc6cd0865eb7ca0160a28708db9398b` |
| `Wan2.2_VAE.pth` | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |

All three checkpoints are loaded with `weights_only=True`. Public checkpoint
checksums are verified before inference. A checksum mismatch means the local
file is incomplete or differs from the pinned release; obtain the exact file
listed above before continuing.
