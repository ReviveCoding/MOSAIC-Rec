from __future__ import annotations

import torch

from mosaic_rec.p2.models import (
    BPRMF,
    VBPR,
    BERT4RecExpert,
    ContentTwoTower,
    ContextRouterGate,
    LegacyM1,
    LightGCN,
    MonotonicSupportGate,
    SASRecExpert,
    cloze_batch,
)
from mosaic_rec.p2.router import align_expert_scores, hard_zero_combine


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


def test_declared_alignment_methods_preserve_shape_and_finite_order() -> None:
    scores = torch.tensor([[[1.0, 3.0, -torch.inf], [2.0, 4.0, -torch.inf]]])
    for method in (
        "per_query_standardized_logits",
        "temperature_scaling",
        "rank_percentile_normalization",
    ):
        aligned = align_expert_scores(scores, method, temperatures=(0.5, 2.0))
        assert aligned.shape == scores.shape
        assert aligned[0, 0, 1] > aligned[0, 0, 0]
        assert torch.isneginf(aligned[0, :, 2]).all()


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


def test_additional_baseline_shapes() -> None:
    mf = BPRMF(3, 5, width=4)
    users = torch.tensor([0, 1])
    items = torch.tensor([[0, 1], [2, 3]])
    assert mf.scores(users, items).shape == (2, 2)
    sas = SASRecExpert(5, width=4, length=3, layers=1, heads=2, dropout=0)
    sequence = torch.tensor([[1, 2, 0], [2, 3, 4]])
    assert sas.query(sequence, torch.tensor([2, 3])).shape == (2, 4)
    vbpr = VBPR(3, 5, torch.randn(5, 6), width=4)
    assert vbpr.scores(users, items).shape == (2, 2)
    graph = LightGCN(2, 3, width=4, layers=1)
    indices = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]])
    adjacency = torch.sparse_coo_tensor(indices, torch.ones(4), (5, 5)).coalesce()
    user_vectors, item_vectors = graph(adjacency)
    assert user_vectors.shape == (2, 4)
    assert item_vectors.shape == (3, 4)


def test_legacy_m1_removes_collaborative_branch_at_zero_support() -> None:
    model = LegacyM1(
        torch.randn(3, 6),
        torch.randn(3, 5),
        torch.tensor([1, 2, 1]),
        torch.tensor([True, True, True]),
        torch.tensor([True, True, True]),
        width=4,
        length=3,
        tag_count=2,
        layers=1,
        heads=2,
        dropout=0,
    )
    items = torch.tensor([[1, 2, 0]])
    fused = model.fuse(items, torch.tensor([[0, 3, 0]]))
    assert fused.shape == (1, 3, 4)
    assert torch.equal(fused[0, 2], torch.zeros(4))
    state = model.user_state(items, torch.tensor([[0, 3, 0]]), torch.tensor([2]))
    assert state.shape == (1, 4)
