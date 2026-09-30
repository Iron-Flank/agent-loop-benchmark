"""Reporting boundaries: no failure-to-wrong-answer conversion or artifact overwrite."""
import unittest

from almm_report.metrics import distribution
from almm_report.writer import footprint_warning


class ReportingBoundaries(unittest.TestCase):
    def test_nearest_rank_percentile_and_empty_distribution(self):
        self.assertEqual(distribution([10, 20, 30, 40])['p95'], 40)
        self.assertIsNone(distribution([])['mean'])

    def test_disk_limit_warns_only_above_five_decimal_gigabytes(self):
        self.assertEqual(footprint_warning(5_000_000_000), [])
        self.assertEqual(len(footprint_warning(5_000_000_001)), 1)


if __name__ == '__main__':
    unittest.main()
