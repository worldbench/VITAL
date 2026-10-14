"""Single-file VITAL checkpoint and pinned, separately downloaded base models."""
import hashlib
import json
from importlib.resources import files
from pathlib import Path


FORMAT = 'vital_weights_v2'


def release_manifest(path=None):
    """Read the packaged manifest, or an explicitly supplied release manifest."""
    source = Path(path) if path is not None else files('vital.resources').joinpath('weights_manifest.json')
    metadata = json.loads(source.read_text(encoding='utf-8'))
    if metadata.get('format') != FORMAT or metadata.get('filename') != 'vital.pt':
        raise ValueError('Unexpected release manifest')
    return metadata


UPSTREAM = {
    'simwam': {
        'repository': 'H-EmbodVis/SimWAM',
        'revision': 'a862a0dc8576be777872c128266a5573d8ae3c40',
        'filename': 'weights/SimWAM.pt',
        'local_name': 'SimWAM.pt',
        'bytes': 12041719353,
        'sha256': '1d05ec214008efa15c5b2fc5029e139fecc6cd0865eb7ca0160a28708db9398b',
    },
    'wan_vae': {
        'repository': 'Wan-AI/Wan2.2-TI2V-5B',
        'revision': '921dbaf3f1674a56f47e83fb80a34bac8a8f203e',
        'filename': 'Wan2.2_VAE.pth',
        'local_name': 'Wan2.2_VAE.pth',
        'bytes': 2818839170,
        'sha256': '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, entry):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        raise ValueError(f'Missing or truncated weights: {path}')
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Wrong checkpoint version or checksum mismatch: {path}')


def load_bundle(path):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Expected the single VITAL weight file: {path}')
    bundle = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    required = {'format', 'nfe', 'seed', 'model_config', 'action_policy',
                'trajectory_decoder', 'residual_policy', 'upstream', 'notices'}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise ValueError('Unexpected VITAL checkpoint structure')
    if bundle['format'] != FORMAT or bundle['nfe'] != 1 or bundle['seed'] != 0:
        raise ValueError('Expected VITAL NFE=1 checkpoint format v2')
    if bundle['upstream'] != UPSTREAM:
        raise ValueError('Checkpoint requires a different upstream model version')
    return bundle
