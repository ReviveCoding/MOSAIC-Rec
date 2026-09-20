"""Known-answer temporal negative sampling contract for the BPR pilot."""

import unittest

from mosaic_rec.pilot_bpr_split import time_safe_negatives


class BPRNegativeTimeContract(unittest.TestCase):
    def test_future_items_and_future_user_history_are_unavailable(self) -> None:
        fit = [(1, 1, 1), (2, 2, 2), (3, 1, 3), (4, 1, 2)]
        users, positives, negatives, seen, warm = time_safe_negatives(fit, 17)
        self.assertEqual(users, [2, 1])
        self.assertEqual(positives, [2, 3])
        self.assertEqual(negatives, [1, 2])
        self.assertEqual(seen[1], {1, 2, 3})
        self.assertEqual(warm, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
