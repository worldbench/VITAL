#!/usr/bin/env python3
"""Download official test logs/cameras/maps, not an author-prepared data archive."""
import argparse
from pathlib import Path
import shutil
import tempfile
from vital.infrastructure.download import download, extract_archive

OPENSCENE = 'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1'
MAPS = 'https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip'


def install(url, subtree, destination, work):
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='navsim-', dir=work) as tmp:
        temp = Path(tmp)
        archive = download(url, temp / 'download.archive')
        extract_archive(archive, temp / 'extracted')
        source = temp / 'extracted' / subtree
        if not source.is_dir():
            raise RuntimeError(f'Upstream archive layout changed: missing {subtree}')
        shutil.copytree(source, destination, dirs_exist_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/navsim'))
    p.add_argument('--components', nargs='+', choices=['logs', 'cameras', 'maps'],
                   default=['logs', 'cameras', 'maps'])
    p.add_argument('--camera-shards', type=int, nargs='+', default=list(range(32)))
    p.add_argument('--accept-data-terms', action='store_true',
                   help='Confirm you have read and accept the linked upstream dataset terms')
    args = p.parse_args(argv)
    if not args.accept_data_terms:
        p.error('Read the dataset terms linked in README, then pass --accept-data-terms')
    if any(n < 0 or n > 31 for n in args.camera_shards):
        p.error('Test camera shards are numbered 0 through 31')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if 'logs' in args.components:
        install(OPENSCENE + '/openscene_metadata_test.tgz', 'openscene-v1.1/meta_datas',
                root / 'navsim_logs/test', root)
    if 'cameras' in args.components:
        for n in args.camera_shards:
            print(f'Downloading camera shard {n}/31', flush=True)
            install(OPENSCENE + f'/openscene_sensor_test_camera/openscene_sensor_test_camera_{n}.tgz',
                    'openscene-v1.1/sensor_blobs', root / 'sensor_blobs/test', root)
    if 'maps' in args.components:
        install(MAPS, 'nuplan-maps-v1.0', root / 'maps', root)
    print('Public NAVSIM files installed under', root)


if __name__ == '__main__':
    main()
