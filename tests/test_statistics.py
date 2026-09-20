"""Known-answer checks for paired clustered inference."""

import unittest

import numpy as np

from mosaic_rec.statistics import holm_adjust, paired_cluster_interval


class StatisticsTests(unittest.TestCase):
    def test_constant_paired_effect_and_item_clusters(self) -> None:
        model = np.array([0.3, 0.4, 0.5, 0.6])
        comparator = model - 0.1
        for kind, ids in (("user", [1, 2, 3, 4]), ("item", [9, 9, 10, 10])):
            result = paired_cluster_interval(model, comparator, np.array(ids),
                                             cluster_kind=kind, draws=1000, seed=42)
            self.assertAlmostEqual(result.point, 0.1)
            self.assertAlmostEqual(result.lower, 0.1)
            self.assertAlmostEqual(result.upper, 0.1)

    def test_invalid_metric_or_pairing_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_cluster_interval(np.array([0.2, 1.2]), np.array([0.1, 0.1]),
                                    np.array([1, 2]), cluster_kind="user")
        with self.assertRaises(ValueError):
            paired_cluster_interval(np.array([0.2]), np.array([0.1, 0.1]),
                                    np.array([1]), cluster_kind="user")

    def test_holm_known_answer(self) -> None:
        np.testing.assert_allclose(holm_adjust(np.array([0.04, 0.01, 0.03])),
                                   np.array([0.06, 0.03, 0.06]))


if __name__ == "__main__":
    unittest.main()
