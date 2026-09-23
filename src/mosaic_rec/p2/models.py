"""p2 sequence and multimodal content experts."""

from __future__ import annotations

import torch
from torch import nn


class BERT4RecExpert(nn.Module):
    def __init__(
        self,
        catalog_size: int,
        *,
        width: int = 64,
        length: int = 50,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.catalog_size = catalog_size
        self.mask_id = catalog_size + 1
        self.length = length
        self.item = nn.Embedding(catalog_size + 2, width, padding_idx=0)
        self.position = nn.Embedding(length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        position = torch.arange(sequence.shape[1], device=sequence.device)
        values = self.item(sequence) + self.position(position)[None, :, :]
        encoded = self.encoder(values, src_key_padding_mask=sequence.eq(0))
        return self.norm(encoded)

    def query(self, histories: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        masked = histories.clone()
        positions = lengths.clamp_min(1) - 1
        masked[torch.arange(len(masked), device=masked.device), positions] = (
            self.mask_id
        )
        encoded = self(masked)
        return encoded[torch.arange(len(masked), device=masked.device), positions]


def cloze_batch(
    sequence: torch.Tensor,
    lengths: torch.Tensor,
    mask_id: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observed = sequence.ne(0)
    selected = (
        torch.rand(sequence.shape, device=sequence.device) < probability
    ) & observed
    selected[torch.arange(len(sequence), device=sequence.device), lengths - 1] = True
    labels = sequence[selected]
    masked = sequence.clone()
    masked[selected] = mask_id
    return masked, selected, labels


class ContentTwoTower(nn.Module):
    def __init__(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        tag_ids: torch.Tensor,
        *,
        width: int = 64,
        tag_count: int = 108,
    ) -> None:
        super().__init__()
        self.register_buffer("image_features", image, persistent=False)
        self.register_buffer("text_features", text, persistent=False)
        self.register_buffer("tag_ids", tag_ids, persistent=False)
        self.image_projection = nn.Linear(image.shape[1], width, bias=False)
        self.text_projection = nn.Linear(text.shape[1], width, bias=False)
        self.tag_embedding = nn.Embedding(tag_count + 1, width, padding_idx=0)
        self.item_norm = nn.LayerNorm(width)
        self.user_norm = nn.LayerNorm(width)

    def item_vectors(self, model_item_ids: torch.Tensor) -> torch.Tensor:
        padding = model_item_ids.eq(0)
        indices = (model_item_ids - 1).clamp_min(0)
        image = self.image_features[indices]
        text = self.text_features[indices]
        tags = self.tag_ids[indices]
        values = (
            self.image_projection(image)
            + self.text_projection(text)
            + self.tag_embedding(tags)
        ) / 3.0
        values = self.item_norm(values)
        values = torch.nn.functional.normalize(values, dim=-1)
        return values.masked_fill(padding[..., None], 0)

    def user_vectors(self, histories: torch.Tensor) -> torch.Tensor:
        values = self.item_vectors(histories)
        mask = histories.ne(0)
        summed = (values * mask[..., None]).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.nn.functional.normalize(self.user_norm(summed / counts), dim=-1)


class MonotonicSupportGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intercept = nn.Parameter(torch.tensor(0.0))
        self.raw_slope = nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        support_log = features[..., 0]
        weight = torch.sigmoid(
            self.intercept + torch.nn.functional.softplus(self.raw_slope) * support_log
        )
        return torch.where(support_log.eq(0), torch.zeros_like(weight), weight)


class ContextRouterGate(nn.Module):
    def __init__(self, feature_count: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.network(features).squeeze(-1))
        return torch.where(features[..., 0].eq(0), torch.zeros_like(weight), weight)
