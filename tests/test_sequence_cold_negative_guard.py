"""Sequence training candidates must exclude FIT-hidden cold items."""

import unittest

from mosaic_rec.pilot_sasrec import training_availability


class SequenceColdNegativeGuard(unittest.TestCase):
    def test_training_pool_uses_fit_items_only(self) -> None:
        order, times = training_availability([(2, 1, 7), (1, 2, 3)], {9})
        self.assertEqual(order, [3, 7])
        self.assertEqual(times, [1, 2])
        self.assertNotIn(9, order)

    def test_visible_cold_support_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Controlled-cold"):
            training_availability([(1, 1, 9)], {9})


if __name__ == "__main__":
    unittest.main()
