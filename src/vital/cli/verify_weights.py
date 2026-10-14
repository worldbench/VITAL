#!/usr/bin/env python3
"""Verify the single VITAL checkpoint; optionally verify public base weights."""
import argparse
from pathlib import Path
from vital.weights import UPSTREAM, load_bundle, release_manifest, verify_file


def verify(path, manifest=None):
    metadata = release_manifest(manifest)
    verify_file(path, metadata)
    load_bundle(path)
    print(f'OK {path}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('file', nargs='?', default='checkpoints/vital.pt')
    p.add_argument('--manifest', type=Path, help='Override the packaged release manifest')
    p.add_argument('--upstream-dir', type=Path, help='Also check the two public model files')
    args = p.parse_args(argv)
    verify(args.file, args.manifest)
    if args.upstream_dir:
        for entry in UPSTREAM.values():
            path = args.upstream_dir / entry['local_name']
            verify_file(path, entry)
            print(f'OK {path}', flush=True)


if __name__ == '__main__':
    main()
