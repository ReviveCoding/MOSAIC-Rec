import torch

from mosaic_rec.p2.qualification import _ensemble


def test_ensemble_averages_rank_aligned_seed_scores() -> None:
    scores = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [3.0, 2.0, 1.0]]])
    result = _ensemble(scores)
    assert result.shape == (1, 3)
    assert torch.allclose(result, torch.tensor([[1 / 3, 0.5, 2 / 3]]))
