"""Small CUDA-ready neural classifiers used by the B6 deep runner."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _torchvision_models():
    try:
        from torchvision import models
    except ImportError as error:
        raise RuntimeError(
            "A working torchvision runtime is required for the B6 ResNet-18, MobileNetV3, and ViT models"
        ) from error
    return models


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


class MelImageClassifier(nn.Module):
    """Torchvision image backbone consuming the shared B4 three-channel Mel tensor."""

    SUPPORTED = ("resnet18", "mobilenet_v3_small", "vit_b_16")

    def __init__(self, backbone: str, classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        models = _torchvision_models()
        self.backbone_name = backbone
        if backbone == "resnet18":
            self.network = models.resnet18(weights=None, num_classes=classes)
            self.network.fc = nn.Sequential(nn.Dropout(dropout), self.network.fc)
            self.feature_dim = int(self.network.fc[1].in_features)
        elif backbone == "mobilenet_v3_small":
            self.network = models.mobilenet_v3_small(weights=None, num_classes=classes)
            self.network.classifier[2] = nn.Dropout(p=dropout, inplace=True)
            self.feature_dim = int(self.network.classifier[-1].in_features)
        elif backbone == "vit_b_16":
            self.network = models.vit_b_16(weights=None, image_size=224, num_classes=classes)
            self.network.heads.head = nn.Sequential(
                nn.Dropout(dropout), self.network.heads.head
            )
            self.feature_dim = int(self.network.heads.head[1].in_features)
        else:
            raise ValueError(f"Unsupported image backbone: {backbone}")

    def _validate_input(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != 3 or values.shape[2] != 80:
            raise ValueError(
                f"Expected B4 Mel tensor shape (batch, 3, 80, frames), got {tuple(values.shape)}"
            )
        if self.backbone_name == "vit_b_16":
            values = F.interpolate(
                values, size=(224, 224), mode="bilinear", align_corners=False
            )
        return values

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        """Return the stable pooled representation immediately before the task FC."""
        values = self._validate_input(values)
        if self.backbone_name != "resnet18":
            raise RuntimeError("The formal forward_features API is active only for resnet18")
        network = self.network
        values = network.conv1(values)
        values = network.bn1(values)
        values = network.relu(values)
        values = network.maxpool(values)
        values = network.layer1(values)
        values = network.layer2(values)
        values = network.layer3(values)
        values = network.layer4(values)
        values = network.avgpool(values)
        return torch.flatten(values, 1)

    def forward_emotion(self, representation: torch.Tensor) -> torch.Tensor:
        """Classify a pooled ResNet18 representation."""
        if self.backbone_name != "resnet18":
            raise RuntimeError("The formal forward_emotion API is active only for resnet18")
        if representation.ndim != 2 or representation.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected pooled feature shape (batch, {self.feature_dim}), "
                f"got {tuple(representation.shape)}"
            )
        return self.network.fc(representation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "resnet18":
            return self.forward_emotion(self.forward_features(values))
        return self.network(self._validate_input(values))


class ResNet18EmotionAccentModel(nn.Module):
    """Only active E1-E7 model: ResNet18 with optional compact accent adversary."""

    MODES = ("none", "unconditional", "conditional")

    def __init__(
        self, *, adversary: str = "none", classes: int = 4, accents: int = 3,
        dropout: float = 0.2, adversary_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if adversary not in self.MODES:
            raise ValueError(f"Unsupported adversary mode: {adversary}")
        self.emotion_model = MelImageClassifier("resnet18", classes=classes, dropout=dropout)
        self.adversary = adversary
        self.feature_dim = self.emotion_model.feature_dim
        adversary_input = self.feature_dim + (classes if adversary == "conditional" else 0)
        self.accent_head = None if adversary == "none" else nn.Sequential(
            nn.Linear(adversary_input, adversary_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adversary_hidden_dim, accents),
        )

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        return self.emotion_model.forward_features(values)

    def forward_emotion(self, representation: torch.Tensor) -> torch.Tensor:
        return self.emotion_model.forward_emotion(representation)

    def forward(
        self, values: torch.Tensor, lambda_adv: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        representation = self.forward_features(values)
        emotion_logits = self.forward_emotion(representation)
        if self.accent_head is None:
            return emotion_logits, None
        if self.adversary == "conditional":
            # The conditioning branch is deliberately detached. GRL supplies the
            # only reversal; the positive accent CE must never be negated again.
            posterior = torch.softmax(emotion_logits, dim=1).detach()
            representation = torch.cat((representation, posterior), dim=1)
        accent_logits = self.accent_head(gradient_reverse(representation, lambda_adv))
        return emotion_logits, accent_logits


class Wav2Vec2EmbeddingMLP(nn.Module):
    def __init__(self, classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class FrozenEmbeddingHead(nn.Module):
    """Small train-only head for a frozen encoder's cached utterance vectors."""

    def __init__(self, dimension: int = 768, classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension), nn.Dropout(dropout), nn.Linear(dimension, classes)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.network[0].normalized_shape[0]:
            raise ValueError(f"Unexpected frozen embedding shape: {tuple(values.shape)}")
        return self.network(values)


class FrozenWav2Vec2Classifier(nn.Module):
    """Official raw-waveform Wav2Vec2 model with a fully frozen encoder."""

    def __init__(
        self, encoder: nn.Module, hidden_size: int = 768,
        classes: int = 4, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, classes)
        )

    def forward(
        self, waveform: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # The encoder remains frozen and deterministic while the head trains.
        self.encoder.eval()
        with torch.no_grad():
            hidden = self.encoder(
                waveform, attention_mask=attention_mask
            ).last_hidden_state
            valid = self.encoder._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


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
