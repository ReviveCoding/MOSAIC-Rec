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


class BPRMF(nn.Module):
    def __init__(self, users: int, items: int, width: int = 64) -> None:
        super().__init__()
        self.user = nn.Embedding(users, width)
        self.item = nn.Embedding(items, width)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)

    def scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users)[..., None, :] * self.item(items)).sum(dim=-1)


class SASRecExpert(nn.Module):
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
        self.length = length
        self.item = nn.Embedding(catalog_size + 1, width, padding_idx=0)
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
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        values = self.item(sequence) + self.position(positions)[None, :, :]
        causal = torch.triu(
            torch.ones(
                sequence.shape[1],
                sequence.shape[1],
                device=sequence.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        encoded = self.encoder(values, mask=causal, src_key_padding_mask=sequence.eq(0))
        return self.norm(encoded)

    def query(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self(sequence)
        positions = lengths.clamp_min(1) - 1
        return encoded[torch.arange(len(sequence), device=sequence.device), positions]


class VBPR(nn.Module):
    def __init__(
        self, users: int, items: int, image_features: torch.Tensor, width: int = 64
    ) -> None:
        super().__init__()
        self.register_buffer("image_features", image_features, persistent=False)
        self.user_id = nn.Embedding(users, width)
        self.item_id = nn.Embedding(items, width)
        self.user_visual = nn.Embedding(users, width)
        self.image_projection = nn.Linear(image_features.shape[1], width, bias=False)
        for embedding in (self.user_id, self.item_id, self.user_visual):
            nn.init.normal_(embedding.weight, std=0.02)

    def scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        id_score = (self.user_id(users)[..., None, :] * self.item_id(items)).sum(dim=-1)
        visual = self.image_projection(self.image_features[items])
        visual_score = (self.user_visual(users)[..., None, :] * visual).sum(dim=-1)
        return id_score + visual_score


class LightGCN(nn.Module):
    def __init__(
        self, users: int, items: int, width: int = 64, layers: int = 2
    ) -> None:
        super().__init__()
        self.users = users
        self.items = items
        self.layers = layers
        self.embedding = nn.Embedding(users + items, width)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.embedding.weight
        levels = [values]
        for _ in range(self.layers):
            values = torch.sparse.mm(adjacency, values)
            levels.append(values)
        combined = torch.stack(levels).mean(dim=0)
        return combined[: self.users], combined[self.users :]


class LegacyM1(nn.Module):
    """p1 M1 fixed-fusion architecture adapted to the p2 catalog."""

    def __init__(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        tag_ids: torch.Tensor,
        image_present: torch.Tensor,
        text_present: torch.Tensor,
        *,
        width: int = 64,
        length: int = 50,
        tag_count: int = 108,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.length = length
        self.item = nn.Embedding(len(image) + 1, width, padding_idx=0)
        self.image_projection = nn.Linear(image.shape[1], width, bias=False)
        self.text_projection = nn.Linear(text.shape[1], width, bias=False)
        self.tag_embedding = nn.Embedding(tag_count + 1, width, padding_idx=0)
        self.position = nn.Embedding(length, width)
        self.register_buffer("image_features", image, persistent=False)
        self.register_buffer("text_features", text, persistent=False)
        self.register_buffer("tag_ids", tag_ids, persistent=False)
        self.register_buffer("image_present", image_present, persistent=False)
        self.register_buffer("text_present", text_present, persistent=False)
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

    def fuse(self, model_item_ids: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        active = model_item_ids.ne(0)
        indices = (model_item_ids - 1).clamp_min(0)
        branches = torch.stack(
            [
                self.item(model_item_ids),
                self.text_projection(self.text_features[indices]),
                self.image_projection(self.image_features[indices]),
                self.tag_embedding(self.tag_ids[indices]),
            ],
            dim=-2,
        )
        available = torch.stack(
            [
                active & support.gt(0),
                active & self.text_present[indices],
                active & self.image_present[indices],
                active,
            ],
            dim=-1,
        )
        weights = available.float() / available.sum(dim=-1, keepdim=True).clamp_min(1)
        return (
            (branches * weights[..., None])
            .sum(dim=-2)
            .masked_fill(~active[..., None], 0)
        )

    def user_state(
        self, sequence: torch.Tensor, support: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        values = self.fuse(sequence, support) + self.position(positions)[None, :, :]
        causal = torch.triu(
            torch.ones(
                sequence.shape[1],
                sequence.shape[1],
                dtype=torch.bool,
                device=sequence.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(values, mask=causal, src_key_padding_mask=sequence.eq(0))
        last = encoded[torch.arange(len(sequence), device=sequence.device), lengths - 1]
        return self.norm(last)
