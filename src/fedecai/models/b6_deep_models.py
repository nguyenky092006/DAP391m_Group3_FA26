"""Small CUDA-ready neural classifiers used by the B6 deep runner."""

from __future__ import annotations

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def gradient_reverse(values: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReversal.apply(values, strength)


class LogMelCNN2D(nn.Module):
    def __init__(self, classes: int = 4, dropout: float = 0.35) -> None:
        super().__init__()
        blocks = []
        channels = (1, 16, 32, 64)
        for source, target in zip(channels, channels[1:]):
            blocks.extend([
                nn.Conv2d(source, target, 3, padding=1),
                nn.BatchNorm2d(target), nn.ReLU(), nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, classes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


class AccentInvariantCNN(nn.Module):
    """CNN emotion model with optional (conditional) GRL accent supervision."""

    def __init__(
        self, classes: int = 4, accents: int = 3, dropout: float = 0.35,
        adversarial: bool = False, conditional: bool = False,
    ) -> None:
        super().__init__()
        blocks = []
        channels = (1, 16, 32, 64)
        for source, target in zip(channels, channels[1:]):
            blocks.extend([
                nn.Conv2d(source, target, 3, padding=1),
                nn.BatchNorm2d(target), nn.ReLU(), nn.MaxPool2d(2),
            ])
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.emotion_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, classes))
        self.adversarial = adversarial
        self.conditional = conditional
        discriminator_input = 64 + (classes if conditional else 0)
        self.accent_head = nn.Sequential(
            nn.Linear(discriminator_input, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, accents),
        ) if adversarial else None

    def forward(
        self, values: torch.Tensor, grl_strength: float = 0.0,
        emotion_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = self.encoder(values)
        emotion_logits = self.emotion_head(embedding)
        if not self.adversarial or self.accent_head is None:
            return emotion_logits, None
        reversed_embedding = gradient_reverse(embedding, grl_strength)
        if self.conditional:
            if emotion_condition is None:
                emotion_condition = torch.softmax(emotion_logits.detach(), dim=1)
            reversed_embedding = torch.cat((reversed_embedding, emotion_condition), dim=1)
        return emotion_logits, self.accent_head(reversed_embedding)


class Wav2Vec2EmbeddingMLP(nn.Module):
    def __init__(self, classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class FrozenPitchFusion(nn.Module):
    """Gated fusion of frozen Wav2Vec2 utterance embeddings and pitch summaries."""

    def __init__(self, classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.audio = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU())
        self.pitch = nn.Sequential(nn.Linear(6, 32), nn.LayerNorm(32), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(160, 128), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(160, 64), nn.GELU(), nn.Linear(64, classes))

    def forward(self, audio: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        audio_hidden = self.audio(audio)
        pitch_hidden = self.pitch(pitch)
        joined = torch.cat((audio_hidden, pitch_hidden), dim=1)
        fused = torch.cat((audio_hidden * self.gate(joined), pitch_hidden), dim=1)
        return self.classifier(fused)


class CrossAttentionBlock(nn.Module):
    """Paper-aligned cross/self-attention block with residual feed-forward layers."""

    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, target: torch.Tensor, source: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            target, source, source, key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        output = self.norm1(target + self.dropout(attended))
        return self.norm2(output + self.dropout(self.feed_forward(output)))


class PitchSequenceEncoder(nn.Module):
    """Wav2Vec2-style Conv1D encoder for a waveform-length pitch sequence."""

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        kernels = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        layers: list[nn.Module] = []
        source = 1
        for kernel, stride in zip(kernels, strides):
            layers.extend((nn.Conv1d(source, 512, kernel, stride=stride), nn.GELU()))
            source = 512
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden_size))

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(pitch[:, None, :]).transpose(1, 2))


class PaperAlignedPitchFusionHead(nn.Module):
    """Bidirectional cross-attention head described by the official ViSEC code."""

    def __init__(
        self, hidden_size: int = 768, projection_size: int = 256,
        classes: int = 4, heads: int = 16, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pitch_encoder = PitchSequenceEncoder(hidden_size)
        self.audio_to_pitch = CrossAttentionBlock(hidden_size, heads, dropout)
        self.pitch_to_audio = CrossAttentionBlock(hidden_size, heads, dropout)
        self.self_attention = CrossAttentionBlock(hidden_size * 2, heads, dropout)
        self.projector = nn.Linear(hidden_size * 2, projection_size)
        self.classifier = nn.Linear(projection_size, classes)

    def forward(
        self, acoustic: torch.Tensor, pitch_waveform: torch.Tensor,
        frame_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pitch = self.pitch_encoder(pitch_waveform)
        width = min(acoustic.shape[1], pitch.shape[1])
        acoustic, pitch = acoustic[:, :width], pitch[:, :width]
        mask = frame_padding_mask[:, :width] if frame_padding_mask is not None else None
        acoustic_attended = self.audio_to_pitch(acoustic, pitch, mask)
        pitch_attended = self.pitch_to_audio(pitch, acoustic, mask)
        fused = self.self_attention(
            torch.cat((acoustic_attended, pitch_attended), dim=-1),
            torch.cat((acoustic_attended, pitch_attended), dim=-1), mask,
        )
        projected = self.projector(fused)
        if mask is None:
            pooled = projected.mean(dim=1)
        else:
            valid = (~mask).unsqueeze(-1).to(projected.dtype)
            pooled = (projected * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)
