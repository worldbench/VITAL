"""Exercise workflow boundaries with deterministic synthetic model outputs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vital.services.inference import run_inference


class InferenceWorkflowTest(unittest.TestCase):
    def test_predictions_are_sorted_and_preserve_the_output_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'predictions.npz'
            with patch('vital.services.inference.VITALPolicy') as policy, \
                    patch('vital.services.inference.scenes', return_value=[('b', []), ('a', [])]), \
                    patch('vital.services.inference.observation', return_value=(None, None)):
                policy.return_value.predict.side_effect = [np.ones((40, 3)), np.zeros((40, 3))]
                run_inference(data_root=directory, weights='weights.pt', simwam_weights='base.pt',
                              wan_vae='vae.pt', output=output, device='cpu', limit=2)
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result['tokens'].tolist(), ['a', 'b'])
                np.testing.assert_array_equal(result['poses'][0], np.zeros((40, 3)))
                np.testing.assert_array_equal(result['poses'][1], np.ones((40, 3)))
                self.assertEqual(result['nfe'].item(), 1)
                self.assertEqual(result['sampling_interval_s'].item(), .1)

    def test_incomplete_full_benchmark_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'predictions.npz'
            with patch('vital.services.inference.VITALPolicy') as policy, \
                    patch('vital.services.inference.scenes', return_value=[('a', [])]), \
                    patch('vital.services.inference.observation', return_value=(None, None)):
                policy.return_value.predict.return_value = np.zeros((40, 3))
                with self.assertRaisesRegex(RuntimeError, '12,146'):
                    run_inference(data_root=directory, weights='weights.pt', simwam_weights='base.pt',
                                  wan_vae='vae.pt', output=output, device='cpu')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
