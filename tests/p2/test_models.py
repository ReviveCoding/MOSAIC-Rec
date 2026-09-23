from __future__ import annotations

import torch

from mosaic_rec.p2.models import (
    BERT4RecExpert,
    ContentTwoTower,
    ContextRouterGate,
    MonotonicSupportGate,
    cloze_batch,
)
from mosaic_rec.p2.router import hard_zero_combine


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


def test_hard_zero_router_uses_content_only_at_zero_support() -> None:
    expert_scores = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]], dtype=torch.float64
    )
    support = torch.tensor([0, 1, 0])
    combined = hard_zero_combine(expert_scores, support)
    assert combined.shape == (1, 3)
    sequence_only = hard_zero_combine(
        torch.stack([expert_scores[:, 0], expert_scores[:, 0]], dim=1), support
    )
    assert combined[0, 1] == sequence_only[0, 1]
    assert combined[0, 0] != sequence_only[0, 0]


def test_router_gates_enforce_zero_support_content_only() -> None:
    features = torch.zeros(2, 3, 8)
    features[:, 1:, 0] = torch.tensor([0.2, 0.8])
    monotonic = MonotonicSupportGate()
    context = ContextRouterGate(hidden=4)
    for gate in (monotonic, context):
        weights = gate(features)
        assert torch.equal(weights[:, 0], torch.zeros(2))
        assert torch.all((weights[:, 1:] > 0) & (weights[:, 1:] < 1))
    monotonic_weights = monotonic(features)
    assert torch.all(monotonic_weights[:, 2] >= monotonic_weights[:, 1])
