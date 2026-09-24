from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_m1_is_fixed_svm_rbf_without_search_grid() -> None:
    config = json.loads(
        (ROOT / "configs/models/m1_svm_rbf_v1.json").read_text(encoding="utf-8")
    )
    model = config["models"]["m1_svm_rbf"]
    assert model["kernel"] == "rbf"
    assert isinstance(model["C"], float)
    assert model["gamma"] == "scale"
    assert config["evaluation"]["selection_partition"].startswith("none;")


def test_m3_m4_difference_is_feature_channels_only() -> None:
    config = json.loads(
        (ROOT / "configs/models/m2_m4_pitch_fusion_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["models"] == [
        "m2_wav2vec2",
        "m3_pitch_fusion",
        "m4_multi_prosody",
    ]
    assert config["architecture"]["m3_m4_difference"] == "input channel count only"
    assert len(config["architecture"]["m3_prosody_channels"]) == 1
    assert len(config["architecture"]["m4_prosody_channels"]) == 5


def test_m5_is_human_approved_architecture_only_comparison() -> None:
    config = json.loads(
        (ROOT / "configs/models/m5_residual_dilated_clean_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract = config["comparison_contract"]
    assert contract["baseline"] == "m4_multi_prosody_clean"
    assert contract["same_multi_prosody_input"] is True
    assert contract["same_training_configuration"] is True
    assert contract["only_intended_change"].startswith("simple_cnn1d_to_")
