from importlib.resources import files
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np
import torch
import yaml

from vital.models.decoder import TrajectoryDecoder
from vital.models.residual import ResidualPolicy
from vital.models.residual_math import compose_trajectory
from vital.infrastructure import download
from vital.evaluation.pdms import pdms_components
from vital.weights import load_bundle, verify_file

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTest(unittest.TestCase):
    def test_public_split(self):
        config = yaml.safe_load(files('vital.data').joinpath('navtest.yaml').read_text())
        self.assertEqual(len(set(config['tokens'])), 12146)
        self.assertEqual(len(config['log_names']), 136)

    def test_zero_residual_keeps_yaw_bits(self):
        base = np.arange(120, dtype=np.float32).reshape(40, 3) / 100
        np.testing.assert_array_equal(compose_trajectory(base, np.zeros_like(base)), base)

    def test_v1_score_does_not_use_v2_components(self):
        row = dict(NC=1., DAC=1., EP=.5, TTC=.8, C=.9)
        baseline = pdms_components(row)['PDMS']
        row.update(LK=0., two_frame_extended_comfort=0.)
        self.assertEqual(pdms_components(row)['PDMS'], baseline)

    def test_reject_zip_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'a.zip'
            with zipfile.ZipFile(archive, 'w') as f:
                f.writestr('../escape', 'bad')
            with self.assertRaises(RuntimeError):
                download.extract_archive(archive, Path(tmp) / 'out')

    def test_reject_tar_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'a.tar'
            with tarfile.open(archive, 'w') as f:
                member = tarfile.TarInfo('link')
                member.type = tarfile.SYMTYPE
                member.linkname = '/tmp'
                f.addfile(member)
            with self.assertRaises(RuntimeError):
                download.extract_archive(archive, Path(tmp) / 'out')

    def test_weights_decoder_and_policy(self):
        path = ROOT / 'checkpoints/vital.pt'
        if not path.is_file():
            self.skipTest('Download weights to run checkpoint-dependent checks')
        bundle = load_bundle(path)
        self.assertNotIn('visual_backbone', bundle)
        self.assertNotIn('image_vae', bundle)
        decoder = TrajectoryDecoder().eval()
        decoder.load_state_dict(bundle['trajectory_decoder'], strict=True)
        with torch.no_grad():
            result = decoder(torch.zeros(2, 4, 76))
        self.assertEqual(tuple(result.shape), (2, 40, 3))
        self.assertTrue(torch.isfinite(result).all())
        payload = bundle['residual_policy']
        policy = ResidualPolicy(payload['state_dict'], payload['config'])
        with torch.no_grad():
            correction = policy.correction(torch.zeros(2, 4528))
        self.assertEqual(tuple(correction.shape), (2, 4, 12))
        self.assertTrue(torch.isfinite(correction).all())

    def test_reject_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.pt'
            torch.save({'format': 'vital_weights_v1'}, path)
            with self.assertRaises(ValueError):
                load_bundle(path)

    def test_reject_wrong_upstream_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            path.write_bytes(b'not the pinned public model')
            with self.assertRaises(ValueError):
                verify_file(path, {'bytes': path.stat().st_size, 'sha256': '0' * 64})


if __name__ == '__main__':
    unittest.main()
