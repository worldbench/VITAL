"""One-step inference orchestration, independent of the CLI parser."""
from pathlib import Path

import numpy as np
import torch

from vital.data.navsim import scenes, observation
from vital.models.policy import VITALPolicy


def run_inference(*, data_root, weights, simwam_weights, wan_vae, output, device='cuda', limit=0, tokens=None):
    data_root, weights, output = Path(data_root), Path(weights), Path(output)
    tokens = Path(tokens) if tokens is not None else None
    if limit < 0:
        raise ValueError('--limit must be non-negative')
    if output.exists():
        raise ValueError('Output already exists; choose a new output path')
    requested = tokens.read_text().splitlines() if tokens else None
    torch.set_num_threads(1)
    model = VITALPolicy(weights, device, simwam_weights, wan_vae)
    tokens, predictions = [], []
    for token, frames in scenes(data_root, requested, limit):
        rgb, proprio = observation(data_root, frames)
        predictions.append(model.predict(rgb, proprio))
        tokens.append(token)
        if len(tokens) == 1 or len(tokens) % 100 == 0:
            print(f'Predicted {len(tokens)} scenes (NFE=1)', flush=True)
    if not tokens:
        raise RuntimeError('No eligible scenes')
    if requested is not None and not limit and set(tokens) != set(requested):
        raise RuntimeError('Requested tokens are missing from the downloaded data')
    if not limit and requested is None and len(tokens) != 12146:
        raise RuntimeError(f'Full navtest requires 12,146 scenes, got {len(tokens)}')
    output.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(tokens)
    np.savez_compressed(output, tokens=np.asarray(tokens)[order],
                        poses=np.stack(predictions)[order], nfe=np.asarray(1),
                        sampling_interval_s=np.asarray(.1))
    print(f'Saved {len(tokens)} trajectories to {output}', flush=True)

