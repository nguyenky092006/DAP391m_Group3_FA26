from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from pitchfusion.models.architectures import (
    PitchFusionClassifier,
    Wav2Vec2EmotionClassifier,
)


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        kernels = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        layers: list[nn.Module] = []
        source = 1
        for kernel, stride in zip(kernels, strides):
            layers.append(nn.Conv1d(source, hidden_size, kernel, stride=stride))
            source = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        sequence = self.encoder(audio[:, None, :]).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=sequence)

    def _get_feature_vector_attention_mask(
        self, width: int, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones(
            attention_mask.shape[0], width, dtype=torch.bool, device=attention_mask.device
        )


def test_m2_output_shape() -> None:
    model = Wav2Vec2EmotionClassifier(DummyBackbone())
    audio = torch.randn(2, 4_000)
    mask = torch.ones(2, 4_000, dtype=torch.long)
    assert model(audio, mask).shape == (2, 4)


def test_m3_and_m4_share_output_contract() -> None:
    audio = torch.randn(2, 4_000)
    mask = torch.ones(2, 4_000, dtype=torch.long)
    m3 = PitchFusionClassifier(DummyBackbone(), prosody_channels=1)
    m4 = PitchFusionClassifier(DummyBackbone(), prosody_channels=5)
    assert m3(audio, torch.randn(2, 1, 4_000), mask).shape == (2, 4)
    assert m4(audio, torch.randn(2, 5, 4_000), mask).shape == (2, 4)


def test_m3_rejects_multi_prosody_input() -> None:
    model = PitchFusionClassifier(DummyBackbone(), prosody_channels=1)
    audio = torch.randn(1, 4_000)
    mask = torch.ones(1, 4_000, dtype=torch.long)
    try:
        model(audio, torch.randn(1, 5, 4_000), mask)
    except ValueError as error:
        assert "Expected 1 prosody channels" in str(error)
    else:
        raise AssertionError("M3 accepted M4 input")
