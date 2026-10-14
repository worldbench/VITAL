"""Regression coverage for the installed package and public release contents."""
from importlib import import_module
from importlib.metadata import distribution
from importlib.resources import files
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import tarfile
import unittest
import zipfile

from vital.weights import release_manifest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release', ROOT / 'tools/release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class PackagingTest(unittest.TestCase):
    def test_package_resources(self):
        self.assertEqual(release_manifest()['filename'], 'vital.pt')
        self.assertTrue(files('vital.data').joinpath('navtest.yaml').is_file())

    def test_all_installed_commands_work_outside_repository(self):
        package = distribution('vital')
        entries = [entry for entry in package.entry_points if entry.group == 'console_scripts']
        self.assertEqual(len(entries), 8)
        with tempfile.TemporaryDirectory() as directory:
            for entry in entries:
                with self.subTest(command=entry.name):
                    # Exercise the installed script, not a source import or PATH shim.
                    import sysconfig
                    executable = Path(sysconfig.get_path('scripts')) / entry.name
                    result = subprocess.run([str(executable), '--help'], cwd=directory,
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_help_does_not_import_gpu_or_scoring_dependencies(self):
        import sys
        code = '''
import sys
from vital.cli.infer import build_parser
from vital.cli.score import build_parser as score_parser
build_parser()
score_parser()
assert not {'torch', 'torchvision', 'navsim', 'nuplan'} & sys.modules.keys()
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_archive_is_deterministic_and_excludes_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('# VITAL\n')
            for name in ('checkpoints/private.pt', 'data/private.csv', '.git/config', '.env'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('private')
            contents = release.audited_sources(root)
            first = release.source_archive(contents)
            self.assertEqual(first, release.source_archive(contents))
            with zipfile.ZipFile(io.BytesIO(first)) as archive:
                self.assertEqual(archive.namelist(), ['vital/README.md'])
                entry = archive.infolist()[0]
                self.assertEqual(entry.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(entry.extra, b'')

    def test_private_path_and_custom_identity_are_rejected(self):
        private = '/' + 'home' + '/researcher/private'
        self.assertTrue(release.audit_text(private, 'sample'))
        self.assertTrue(release.audit_text('Example Lab', 'sample', ['example lab']))

    def test_public_readme_figures_and_license_are_in_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('<img src="docs/assets/teaser.png"/>\n')
            (root / 'LICENSE').write_text('Project license\n')
            assets = root / 'docs/assets'
            assets.mkdir(parents=True)
            for name in ('teaser.png', 'method.png'):
                (assets / name).write_bytes(b'\x89PNG\r\n\x1a\n')
            contents = release.audited_sources(root)
            with zipfile.ZipFile(io.BytesIO(release.source_archive(contents))) as archive:
                self.assertIn('vital/LICENSE', archive.namelist())
                for name in ('teaser.png', 'method.png'):
                    self.assertEqual(archive.read(f'vital/docs/assets/{name}'),
                                     (assets / name).read_bytes())

    def test_public_roadmap_heading_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('## 🚧 TODO\n\n- [ ] Release training code.\n')
            self.assertIn('README.md', release.audited_sources(root))

    def test_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').symlink_to(root / 'private.txt')
            with self.assertRaises(ValueError):
                release.source_files(root)

    def test_unfinished_documentation_blocks_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for draft in ('Download URL: TBD\n', 'TODO: add installation instructions.\n'):
                with self.subTest(draft=draft):
                    (root / 'README.md').write_text(draft)
                    with self.assertRaisesRegex(ValueError, 'unfinished documentation'):
                        release.audited_sources(root)

    def test_broken_documentation_link_blocks_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('[Weights](docs/missing.md)\n')
            with self.assertRaisesRegex(ValueError, 'broken local documentation link'):
                release.audited_sources(root)

    def test_broken_readme_image_blocks_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('<img src="docs/assets/missing.png"/>\n')
            with self.assertRaisesRegex(ValueError, 'broken local documentation link'):
                release.audited_sources(root)

    def test_cli_dispatch(self):
        from unittest.mock import patch
        for module, service, function, arguments in (
            ('infer', 'inference', 'run_inference', ['--data-root', 'data/navsim', '--limit', '4']),
            ('prepare_metrics', 'metric_cache', 'prepare_metrics', ['--data-root', 'data/navsim']),
            ('score', 'scoring', 'score_predictions', ['--predictions', 'sample.npz', '--cache', 'cache']),
        ):
            with self.subTest(module=module):
                with patch(f'vital.services.{service}.{function}') as workflow:
                    import_module(f'vital.cli.{module}').main(arguments)
                    workflow.assert_called_once()

    def test_source_distribution_owner_metadata_is_removed(self):
        spec = importlib.util.spec_from_file_location('release_build', ROOT / 'tools/build.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                entry = tarfile.TarInfo('vital/README.md')
                entry.uname = entry.gname = 'local-account'
                entry.uid = entry.gid = 1000
                entry.mtime = 1700000000
                entry.pax_headers = {'mtime': '1700000000.5'}
                entry.size = 5
                archive.addfile(entry, io.BytesIO(b'VITAL'))
            builder.normalize_sdist(path)
            with tarfile.open(path) as archive:
                entry = archive.getmembers()[0]
                self.assertEqual((entry.uid, entry.gid, entry.mtime), (0, 0, 0))
                self.assertEqual((entry.uname, entry.gname, entry.pax_headers), ('', '', {}))
                self.assertEqual(archive.extractfile(entry).read(), b'VITAL')


if __name__ == '__main__':
    unittest.main()
