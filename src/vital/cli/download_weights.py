#!/usr/bin/env python3
"""Download the single VITAL checkpoint and verify it before installing."""
import argparse
import os
from pathlib import Path
import tempfile
import shutil
from vital.infrastructure.download import download
from vital.cli.verify_weights import verify
from vital.weights import release_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default=os.getenv('VITAL_WEIGHTS_URL'),
                   help='Override VITAL_WEIGHTS_URL or the pinned Hugging Face URL in the manifest')
    p.add_argument('--manifest', type=Path, help='Override the packaged release manifest')
    p.add_argument('--output', type=Path, default=Path('checkpoints/vital.pt'))
    args = p.parse_args(argv)
    manifest = release_manifest(args.manifest)
    url = args.url or manifest.get('url')
    if not url:
        p.error('The manifest has no download URL; provide --url or set VITAL_WEIGHTS_URL')
    if args.output.exists():
        p.error('Destination exists; verify it or choose a new destination')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vital-install-', dir=args.output.parent) as tmp:
        tmp = Path(tmp)
        path = download(url, tmp / 'vital.pt', manifest['sha256'])
        verify(path, args.manifest)
        shutil.move(str(path), args.output)
    print('Installed verified weights:', args.output)


if __name__ == '__main__':
    main()
