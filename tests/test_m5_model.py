import json
from pathlib import Path

import torch

from pitchfusion.models.run_m5_clean import LightweightResidualDilatedBlock


ROOT = Path(__file__).resolve().parents[1]


def test_residual_block_preserves_shape_and_has_finite_gradients() -> None:
    block = LightweightResidualDilatedBlock(8, 4, dilation=2, dropout=0.0)
    values = torch.randn(2, 8, 25, requires_grad=True)
    output = block(values)
    assert output.shape == values.shape
    output.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_m5_contract_changes_only_prosody_cnn_architecture() -> None:
    config = json.loads(
        (ROOT / "configs/models/m5_residual_dilated_clean_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = config["comparison_contract"]
    assert contract["baseline"] == "m4_multi_prosody_clean"
    assert contract["same_multi_prosody_input"] is True
    assert contract["same_fusion"] is True
    assert contract["same_classifier"] is True
    assert contract["same_training_configuration"] is True
    assert contract["initialization_from_m4_checkpoint"] is False


def test_m5_config_has_lightweight_dilated_schedule() -> None:
    config = json.loads(
        (ROOT / "configs/models/m5_residual_dilated_clean_v1.json").read_text(
            encoding="utf-8"
        )
    )
    architecture = config["m5_architecture"]
    assert architecture["input_channels"] == 5
    assert architecture["dilations"] == [1, 2, 4]
    assert architecture["bottleneck_channels"] < 512
