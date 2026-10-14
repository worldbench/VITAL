"""Small standard-library downloader shared by release tools."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def download(url: str, destination: Path, sha256: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="vital-download-", dir=destination.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as source, tmp_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        if sha256:
            checksum = hashlib.sha256()
            with tmp_path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    checksum.update(block)
            digest = checksum.hexdigest()
            if digest.lower() != sha256.lower():
                raise RuntimeError(f"SHA256 mismatch: expected {sha256}, got {digest}")
        shutil.move(str(tmp_path), destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    return destination


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Archive symlinks are not supported')
                target = (destination / member.filename).resolve()
                if destination_resolved not in target.parents and target != destination_resolved:
                    raise RuntimeError(f"unsafe archive member: {member.filename}")
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError(f'Unsupported archive member: {member.name}')
                target = (destination / member.name).resolve()
                if destination_resolved not in target.parents and target != destination_resolved:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            handle.extractall(destination)
        return
    raise ValueError(f"unsupported archive format: {archive}")
