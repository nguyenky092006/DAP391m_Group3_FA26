"""M2-M4 architectures for the approved Pitch-Fusion comparison."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SharedEmotionHead(nn.Module):
    """The classifier contract shared by M2, M3, and M4."""

    def __init__(
        self,
        input_size: int,
        projection_size: int = 256,
        classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del dropout
        self.projector = nn.Linear(input_size, projection_size)
        self.classifier = nn.Linear(projection_size, classes)

    def forward(
        self, sequence: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        projected = self.projector(sequence)
        if padding_mask is None:
            pooled = projected.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1).to(projected.dtype)
            pooled = (projected * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


class Wav2Vec2EmotionClassifier(nn.Module):
    """M2: raw waveform to Wav2Vec2 to the shared emotion classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        projection_size: int = 256,
        classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.head = SharedEmotionHead(hidden_size, projection_size, classes, dropout)

    def forward(
        self, audio: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        acoustic = self.backbone(
            audio, attention_mask=attention_mask
        ).last_hidden_state
        frame_valid = self.backbone._get_feature_vector_attention_mask(
            acoustic.shape[1], attention_mask
        )
        return self.head(acoustic, ~frame_valid)


class CrossAttentionBlock(nn.Module):
    """Residual attention and feed-forward block used by M3 and M4."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            target,
            source,
            source,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        output = self.norm1(target + self.dropout(attended))
        return self.norm2(output + self.dropout(self.feed_forward(output)))


class ProsodySequenceEncoder(nn.Module):
    """Paper-aligned temporal Conv1D encoder, generalized only by channel count.

    M3 uses one input channel and M4 uses five. Every later operation and all
    convolutional hyperparameters are identical, so M3 to M4 isolates feature
    gain rather than architecture gain.
    """

    def __init__(self, input_channels: int, hidden_size: int = 768) -> None:
        super().__init__()
        kernels = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        layers: list[nn.Module] = []
        source_channels = input_channels
        for kernel, stride in zip(kernels, strides):
            layers.extend(
                (
                    nn.Conv1d(source_channels, 512, kernel, stride=stride),
                    nn.GELU(),
                )
            )
            source_channels = 512
        self.convolution = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, hidden_size)

    def forward(self, prosody: torch.Tensor) -> torch.Tensor:
        encoded = self.convolution(prosody).transpose(1, 2)
        return self.projection(self.norm(encoded))


class PitchFusionClassifier(nn.Module):
    """M3/M4 with an identical encoder, fusion mechanism, and classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        prosody_channels: int,
        projection_size: int = 256,
        classes: int = 4,
        attention_heads: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.prosody_channels = prosody_channels
        self.prosody_encoder = ProsodySequenceEncoder(prosody_channels, hidden_size)
        self.audio_to_prosody = CrossAttentionBlock(
            hidden_size, attention_heads, dropout
        )
        self.prosody_to_audio = CrossAttentionBlock(
            hidden_size, attention_heads, dropout
        )
        self.self_attention = CrossAttentionBlock(
            hidden_size * 2, attention_heads, dropout
        )
        self.head = SharedEmotionHead(
            hidden_size * 2, projection_size, classes, dropout
        )

    def forward(
        self,
        audio: torch.Tensor,
        prosody: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prosody.shape[1] != self.prosody_channels:
            raise ValueError(
                f"Expected {self.prosody_channels} prosody channels, "
                f"received {prosody.shape[1]}"
            )
        acoustic = self.backbone(
            audio, attention_mask=attention_mask
        ).last_hidden_state
        frame_valid = self.backbone._get_feature_vector_attention_mask(
            acoustic.shape[1], attention_mask
        )
        pitch = self.prosody_encoder(prosody)
        width = min(acoustic.shape[1], pitch.shape[1])
        acoustic = acoustic[:, :width]
        pitch = pitch[:, :width]
        padding_mask = (~frame_valid)[:, :width]
        acoustic_attended = self.audio_to_prosody(
            acoustic, pitch, padding_mask
        )
        prosody_attended = self.prosody_to_audio(
            pitch, acoustic, padding_mask
        )
        joined = torch.cat((acoustic_attended, prosody_attended), dim=-1)
        fused = self.self_attention(joined, joined, padding_mask)
        return self.head(fused, padding_mask)


def freeze_feature_encoder(backbone: Any) -> None:
    """Freeze only Wav2Vec2's convolutional waveform feature extractor."""
    if hasattr(backbone, "freeze_feature_encoder"):
        backbone.freeze_feature_encoder()
    elif hasattr(backbone, "feature_extractor"):
        backbone.feature_extractor._freeze_parameters()
    else:
        raise AttributeError("Backbone does not expose a feature encoder")
