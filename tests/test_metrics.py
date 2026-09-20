import math
import unittest

from mosaic_rec.metrics import single_positive_metrics


class SinglePositiveMetricTests(unittest.TestCase):
    def test_known_ranks_and_misses(self):
        actual = single_positive_metrics([1, 2, 21, None], k=20)
        self.assertEqual(actual["n"], 4)
        self.assertAlmostEqual(actual["ndcg_at_k"], (1 + 1 / math.log2(3)) / 4)
        self.assertAlmostEqual(actual["recall_at_k"], 0.5)
        self.assertAlmostEqual(actual["mrr_at_k"], 0.375)
        self.assertAlmostEqual(actual["hit_rate_at_k"], 0.5)

    def test_cutoff_and_invalid_input(self):
        self.assertEqual(single_positive_metrics([3], k=2)["ndcg_at_k"], 0)
        with self.assertRaises(ValueError):
            single_positive_metrics([], k=20)
        with self.assertRaises(ValueError):
            single_positive_metrics([0], k=20)


if __name__ == "__main__":
    unittest.main()
