import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BasicBlock(nn.Module):
    """Residual block used inside the per-move CNN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class PerMoveCNN(nn.Module):
    """Encode board planes for each move independently."""

    def __init__(
        self,
        in_ch: int,
        width: int = 128,
        depth: int = 4,
        d_move: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[BasicBlock(width) for _ in range(depth)])
        head: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, d_move)]
        if dropout > 0.0:
            head.append(nn.Dropout(dropout))
        head.append(nn.ReLU(inplace=True))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class LSTMEncoder(nn.Module):
    """Sequence encoder built on bidirectional LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            lengths = mask.long().sum(dim=-1)
            lengths = torch.clamp(lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            seq_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            seq_out, _ = self.lstm(x)
        pooled = masked_mean(seq_out, mask)
        return seq_out, pooled


class TransformerEncoder(nn.Module):
    """Optional Transformer-based encoder for experimentation."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~(mask.bool())
        x = self.input_proj(x)
        seq_out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        seq_out = self.norm(seq_out)
        pooled = masked_mean(seq_out, mask)
        return seq_out, pooled


class ProjectionHead(nn.Module):
    """Project pooled representation and enforce unit-norm embeddings."""

    def __init__(self, input_dim: int, hidden: int = 256, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(torch.tanh(x), p=2, dim=-1)


class RankHead(nn.Module):
    """Linear head for supervised objectives (e.g., player classification)."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Model(nn.Module):
    """Full style encoder combining per-move CNN, sequence model, and projection head."""

    def __init__(
        self,
        in_ch: int,
        d_move: int = 128,
        d_seq: int = 256,
        d_vec: int = 128,
        cnn_depth: int = 4,
        seq_type: str = "lstm",
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        per_move_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.per_move = PerMoveCNN(in_ch, width=d_seq, depth=cnn_depth, d_move=d_move, dropout=per_move_dropout)
        if seq_type == "transformer":
            self.seq_encoder = TransformerEncoder(d_move, d_model=d_seq, dropout=dropout)
        else:
            self.seq_encoder = LSTMEncoder(d_move, hidden_dim=d_seq // 2, dropout=dropout)
        self.proj = ProjectionHead(self.seq_encoder.out_dim, hidden=d_seq, output_dim=d_vec, dropout=dropout)
        self.rank_head = RankHead(d_vec, num_classes) if num_classes is not None else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 5:
            raise ValueError(f"Expected input of shape (B, T, C, H, W), got {tuple(x.shape)}")
        b, t, _, _, _ = x.shape
        per_move = self.per_move(rearrange(x, "b t c h w -> (b t) c h w"))
        per_move = rearrange(per_move, "(b t) d -> b t d", b=b, t=t)
        if mask is not None:
            mask = mask.to(per_move.dtype)
        _, pooled = self.seq_encoder(per_move, mask)
        vector = self.proj(pooled)
        logits = self.rank_head(vector) if self.rank_head is not None else None
        return vector, logits


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class GE2ELoss(nn.Module):
    """Generalized end-to-end loss for metric learning across players."""

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings.new_tensor(0.0)
        labels = labels.long()
        unique_labels, inverse = torch.unique(labels, return_inverse=True)
        if unique_labels.numel() <= 1:
            return embeddings.new_tensor(0.0)

        emb = F.normalize(embeddings, dim=-1)
        counts = torch.bincount(inverse, minlength=unique_labels.numel()).unsqueeze(1).to(emb.dtype)
        sums = torch.zeros(unique_labels.numel(), emb.size(1), device=emb.device, dtype=emb.dtype)
        sums.index_add_(0, inverse, emb)
        centroids = sums / counts.clamp_min(1.0)

        sims = emb @ centroids.t()

        for idx in range(emb.size(0)):
            label_idx = inverse[idx]
            if counts[label_idx] <= 1:
                continue
            excl = (sums[label_idx] - emb[idx]) / (counts[label_idx] - 1.0)
            excl = F.normalize(excl, dim=-1)
            sims[idx, label_idx] = (emb[idx] * excl).sum()

        sims = self.w.clamp(min=1e-6) * sims + self.b
        return F.cross_entropy(sims, inverse)
