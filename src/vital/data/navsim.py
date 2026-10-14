"""Read public NAVSIM/OpenScene raw logs, without author-prepared samples."""
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


def scenes(data_root, tokens=None, limit=0):
    """Official NAVSIM v1 navtest filter: 4 history + 10 future, 2 Hz."""
    root = Path(data_root)
    config = yaml.safe_load(Path(__file__).with_name('navtest.yaml').read_text())
    allowed = set(config['tokens'])
    requested = allowed if tokens is None else set(tokens)
    if not requested <= allowed:
        raise ValueError('Requested tokens are outside the public navtest split')
    count = 0
    for log in sorted(config['log_names']):
        path = root / 'navsim_logs/test' / (log + '.pkl')
        if not path.is_file():
            raise FileNotFoundError(f'Missing public navtest log: {path}; download the test metadata')
        # Only unpickle trusted official dataset files, never untrusted uploads.
        with path.open('rb') as f:
            frames = pickle.load(f)
        for start in range(len(frames) - 13):
            window = frames[start:start + 14]
            current = window[3]
            token = current['token']
            if not current['roadblock_ids'] or token not in requested:
                continue
            yield token, window
            count += 1
            if limit and count >= limit:
                return


def observation(data_root, frames):
    current = frames[3]
    path = Path(data_root) / 'sensor_blobs/test' / current['cams']['CAM_F0']['data_path']
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    proprio = np.concatenate([np.asarray(current['ego_dynamic_state'], dtype=np.float32),
                              np.asarray(current['driving_command'], dtype=np.float32)])
    if proprio.shape != (8,):
        raise ValueError('Expected raw ego state[4] and driving command[4]')
    return rgb, proprio
