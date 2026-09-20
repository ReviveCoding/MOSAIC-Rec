"""CPU-safe shape smoke for content dimensions across dataset tracks."""

import unittest
from importlib.util import find_spec

if find_spec("torch"):
    import torch
    from mosaic_rec.pilot_mosaic import MosaicSequence
else:
    torch = None
    MosaicSequence = None


@unittest.skipIf(MosaicSequence is None, "Optional torch runtime not installed")
class ModelShapeTests(unittest.TestCase):
    def test_non_microlens_content_widths(self) -> None:
        model = MosaicSequence(
            1, torch.ones(3, 384), torch.ones(3, 1024),
            torch.tensor([0, 1, 0]), torch.tensor([True, True, False]), 2,
        )
        model.eval()
        fused, weights = model.fuse(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(fused.shape), (3, 64))
        self.assertEqual(tuple(weights.shape), (3, 4))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
