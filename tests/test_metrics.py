import unittest

import numpy as np

from vital.evaluation.metrics import trajectory_metrics


class TrajectoryMetricTest(unittest.TestCase):
    def test_zero_error(self):
        trajectory = np.stack([np.arange(1, 41) * 0.1, np.zeros(40), np.zeros(40)], axis=1)
        recorded = np.stack([np.arange(1, 9) * 0.5, np.zeros(8), np.zeros(8)], axis=1)
        result = trajectory_metrics(trajectory, recorded)
        self.assertAlmostEqual(result["ADE_m"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
