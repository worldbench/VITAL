#!/usr/bin/env python3
"""Audit and package the public VITAL source release using an explicit allowlist."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml', 'requirements.txt',
    'requirements-scoring.txt', '.gitignore', '.editorconfig', 'Makefile', 'MANIFEST.in',
    'data/README.md', 'checkpoints/.gitkeep',
)
TREES = {
    'src/vital': {'.py', '.yaml', '.json'},
    'tests': {'.py'},
    'tools': {'.py'},
    'docs': {'.md'},
    'configs': {'.yaml'},
    'THIRD_PARTY_LICENSES': {'.txt'},
    '.github/workflows': {'.yml'},
}
README_ASSETS = {'docs/assets/teaser.png', 'docs/assets/method.png'}
SKIP = {'__pycache__', '.pytest_cache', '.ruff_cache'}
PATTERNS = {
    'obsolete project spelling': re.compile('v' + '(?:alta|ilta)', re.I),
    'private absolute path': re.compile(r'(?:/(?:home|Users|mnt|media)/[^\s\x22\x27]+|[A-Z]:\\Users\\)', re.I),
    'email address': re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b'),
    'private key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    'access token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b'),
}
DOC_DRAFT_MARKERS = re.compile(
    r'\b(?:TODO|TBD|FIXME)\b|to be filled(?: in)?|coming soon|'
    r'<direct-download-link[^>]*>|replace (?:the|this) .{0,40}placeholder', re.I
)


def source_files(root: Path) -> list[Path]:
    """Exclude history, binaries, datasets, caches, builds and local environments."""
    selected = []
    for name in ROOT_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f'Symlink cannot be released: {name}')
        if path.is_file():
            selected.append(path)
    for directory, extensions in TREES.items():
        base = root / directory
        if base.is_symlink():
            raise ValueError(f'Symlink cannot be released: {directory}')
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(f'Symlink cannot be released: {relative}')
            if SKIP.intersection(relative.parts) or path.is_dir():
                continue
            if path.suffix not in extensions and relative.as_posix() not in README_ASSETS:
                raise ValueError(f'Unexpected source-release file: {relative}')
            selected.append(path)
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def audit_text(text: str, label: str, forbidden=()) -> list[str]:
    findings = []
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count('\n', 0, match.start()) + 1
            findings.append(f'{label}:{line}: {name}')
    for term in forbidden:
        if term and term.casefold() in text.casefold():
            findings.append(f'{label}: custom identity marker')
    return findings


def audited_sources(root: Path, forbidden=()) -> dict[str, bytes]:
    contents = {}
    findings = []
    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        findings.extend(audit_text(name, name, forbidden))
        if name in README_ASSETS:
            contents[name] = raw
            continue
        document = raw.decode('utf-8')
        findings.extend(audit_text(document, name, forbidden))
        if path.suffix == '.md':
            lines = document.splitlines()
            for match in DOC_DRAFT_MARKERS.finditer(document):
                line = document.count('\n', 0, match.start()) + 1
                # A public release may describe planned work under a TODO heading.
                if match.group().casefold() == 'todo' and re.fullmatch(
                    r'#{1,6}\s+[^\w\n]*TODO\s*', lines[line - 1], re.I
                ):
                    continue
                findings.append(f'{name}:{line}: unfinished documentation')
            links = re.findall(r'\]\(([^)]+)\)', document)
            links.extend(re.findall(
                r'<(?:img|a)\b[^>]*\b(?:src|href)=[\x22\x27]([^\x22\x27]+)[\x22\x27]',
                document, re.I
            ))
            for link in links:
                if '://' in link or link.startswith('#'):
                    continue
                target = link.split('#', 1)[0]
                if target and not (path.parent / target).exists():
                    findings.append(f'{name}: broken local documentation link: {target}')
        contents[name] = raw
    if findings:
        raise ValueError('\n'.join(findings))
    return contents


def audit_checkpoint(path: Path, forbidden=()) -> None:
    """Inspect non-tensor metadata after validating the release checksum/schema."""
    import torch
    from vital.weights import load_bundle, release_manifest, verify_file

    verify_file(path, release_manifest())
    bundle = load_bundle(path)
    findings = []

    def walk(value, label):
        if isinstance(value, dict):
            for key, child in value.items():
                findings.extend(audit_text(str(key), label, forbidden))
                walk(child, label + '/' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, label + '/' + str(index))
        elif isinstance(value, str):
            findings.extend(audit_text(value, label, forbidden))
        elif not isinstance(value, (torch.Tensor, int, float, bool, type(None))):
            findings.append(f'{label}: unsupported metadata type')

    walk(bundle, 'checkpoint')
    with zipfile.ZipFile(path) as archive:
        findings.extend(audit_text('\n'.join(archive.namelist()), 'checkpoint archive', forbidden))
    if findings:
        raise ValueError('\n'.join(findings))


def source_archive(contents: dict[str, bytes]) -> bytes:
    """Build deterministic ZIP bytes without user names or filesystem timestamps."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(contents.items()):
            entry = zipfile.ZipInfo('vital/' + name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
    return output.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write a reviewed source ZIP; omit to audit only')
    parser.add_argument('--weights', type=Path, help='Also verify and audit checkpoint metadata')
    parser.add_argument('--forbid', action='append', default=[], help='Additional identity string to reject')
    args = parser.parse_args(argv)
    contents = audited_sources(ROOT, args.forbid)
    if args.weights:
        audit_checkpoint(args.weights, args.forbid)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            handle.write(source_archive(contents))
    print(f'PASS: audited {len(contents)} source files' + (' and checkpoint metadata' if args.weights else ''))
    if args.output:
        print(f'Created {args.output}')


if __name__ == '__main__':
    main()
