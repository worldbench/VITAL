#!/usr/bin/env python3
"""Build distributions and remove local ownership/timestamps from the sdist."""
from __future__ import annotations

import argparse
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def normalize_sdist(path: Path) -> None:
    """Repack a generated tarball without uid/gid, owner names or PAX metadata."""
    output = io.BytesIO()
    with tarfile.open(path, 'r:gz') as source:
        with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as target:
                for member in source.getmembers():
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f'Unsupported distribution member: {member.name}')
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ''
                    member.pax_headers = {}
                    member.mode = 0o755 if member.isdir() else 0o644
                    target.addfile(member, source.extractfile(member) if member.isfile() else None)
    path.write_bytes(output.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    environment = dict(os.environ, SOURCE_DATE_EPOCH='315532800')
    subprocess.run([sys.executable, '-m', 'build', '--outdir', str(output)],
                   cwd=ROOT, env=environment, check=True)
    for archive in output.glob('vital-*.tar.gz'):
        normalize_sdist(archive)
    print('Distribution ownership and timestamps normalized.')


if __name__ == '__main__':
    main()
