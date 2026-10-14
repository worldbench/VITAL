import unittest

from vital.evaluation.pdms import pdms_components


class PDMSComponentTest(unittest.TestCase):
    def test_compact_v1_formula(self):
        result = pdms_components({"nc": 0.9, "dac": 0.8, "ep": 0.7, "ttc": 0.6, "comfort": 0.5})
        expected = 100 * 0.9 * 0.8 * ((5 * 0.7 + 5 * 0.6 + 2 * 0.5) / 12)
        self.assertAlmostEqual(result["PDMS"], expected)


if __name__ == "__main__":
    unittest.main()
