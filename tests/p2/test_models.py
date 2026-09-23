from __future__ import annotations

import torch

from mosaic_rec.p2.models import BERT4RecExpert, ContentTwoTower, cloze_batch


def test_bert4rec_cloze_and_query_shapes() -> None:
    torch.manual_seed(1)
    model = BERT4RecExpert(20, width=8, length=5, layers=1, heads=2, dropout=0)
    sequence = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
    lengths = torch.tensor([3, 4])
    masked, selected, labels = cloze_batch(sequence, lengths, model.mask_id, 0)
    assert selected.sum().item() == 2
    assert labels.tolist() == [3, 7]
    assert model(masked).shape == (2, 5, 8)
    assert model.query(sequence, lengths).shape == (2, 8)


def test_content_tower_padding_is_zero_and_outputs_are_finite() -> None:
    image = torch.randn(4, 6)
    text = torch.randn(4, 3)
    tags = torch.tensor([1, 2, 0, 1])
    model = ContentTwoTower(image, text, tags, width=4, tag_count=2)
    items = model.item_vectors(torch.tensor([[0, 1, 2]]))
    assert torch.equal(items[0, 0], torch.zeros(4))
    users = model.user_vectors(torch.tensor([[1, 2, 0], [3, 4, 1]]))
    assert users.shape == (2, 4)
    assert torch.isfinite(users).all()
