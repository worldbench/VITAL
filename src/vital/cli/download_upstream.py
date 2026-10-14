#!/usr/bin/env python3
"""Download only the exact public WAN-derived backbone and WAN VAE needed by VITAL."""
import argparse
from pathlib import Path
from vital.infrastructure.download import download
from vital.weights import UPSTREAM, verify_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('checkpoints/upstream'))
    p.add_argument('--components', nargs='+', choices=tuple(UPSTREAM), default=list(UPSTREAM))
    p.add_argument('--accept-model-terms', action='store_true', required=True,
                   help='Confirm that you have reviewed and accept the upstream model terms')
    args = p.parse_args(argv)
    for component in dict.fromkeys(args.components):
        entry = UPSTREAM[component]
        path = args.output / entry['local_name']
        if path.exists():
            verify_file(path, entry)
            print(f'Already verified: {path}', flush=True)
            continue
        url = (f"https://huggingface.co/{entry['repository']}/resolve/"
               f"{entry['revision']}/{entry['filename']}")
        print(f"Downloading {component}: {entry['bytes'] / 1e9:.2f} GB", flush=True)
        download(url, path, entry['sha256'])
        verify_file(path, entry)
        print(f'OK {path}', flush=True)


if __name__ == '__main__':
    main()
