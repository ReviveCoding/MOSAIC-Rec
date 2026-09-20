"""Known-answer intervention constraints independent of model accuracy."""

import unittest

import numpy as np

from mosaic_rec.interventions import apply_mismatched_images, mismatched_image_index


class InterventionTests(unittest.TestCase):
    def test_mismatch_is_deterministic_derangement(self) -> None:
        first = mismatched_image_index(11, 42)
        second = mismatched_image_index(11, 42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.arange(11))
        self.assertFalse(np.any(first == np.arange(11)))

    def test_mismatch_uses_other_existing_vectors(self) -> None:
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        corrupted, mapping = apply_mismatched_images(features, 7)
        np.testing.assert_array_equal(corrupted, features[mapping])
        self.assertFalse(np.shares_memory(corrupted, features))
        with self.assertRaises(ValueError):
            mismatched_image_index(1, 7)


if __name__ == "__main__":
    unittest.main()
