from __future__ import annotations

import json
import wave
from pathlib import Path

import pandas as pd

from pitchfusion.models.run_paper_reproduction import (
    build_clean_stratified_split,
    interpolate_like_author,
    parse_kaldi_text_archive,
    write_kaldi_pcm16_wav,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paper_reproduction_config_is_intentionally_not_speaker_disjoint() -> None:
    config = json.loads(
        (ROOT / "configs/models/m3_paper_reproduction_clean_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["data"]["expected_eligible_rows"] == 4992
    assert config["data"]["speaker_disjoint"] is False
    assert config["training"]["batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["learning_rate"] == 1.5e-5
    assert config["training"]["scheduler"] == "linear"
    assert config["training"]["loss"] == "cross_entropy"
    assert config["pitch"]["extractor"] == "kaldi.compute-kaldi-pitch-feats"
    assert config["pitch"]["kaldi_binary_output_index"] == 1
    assert config["emotion_order"] == ["happy", "neutral", "sad", "angry"]


def test_clean_split_is_deterministic_and_stratified(tmp_path: Path) -> None:
    rows = []
    emotions = ["angry", "happy", "neutral", "sad"]
    for index in range(80):
        rows.append(
            {
                "sample_id": f"s{index:03d}",
                "speaker_id": str(index % 8),
                "emotion": emotions[index % 4],
                "audio_sha256": f"hash-{index}",
                "source_row_index": index,
                "is_eligible_for_split": True,
            }
        )
    clean = tmp_path / "clean.csv"
    pd.DataFrame(rows).to_csv(clean, index=False)
    config = {
        "schema_version": "test",
        "seed": 391,
        "split_protocol": "test_split",
        "emotion_order": emotions,
        "data": {
            "eligible_column": "is_eligible_for_split",
            "expected_eligible_rows": 80,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
    }
    first = build_clean_stratified_split(
        clean, tmp_path / "split.csv", tmp_path / "report.json", config, force=False
    )
    second = build_clean_stratified_split(
        clean, tmp_path / "split.csv", tmp_path / "report.json", config, force=False
    )
    assert first[["sample_id", "split"]].equals(second[["sample_id", "split"]])
    assert first["split"].value_counts().to_dict() == {
        "train": 64,
        "validation": 8,
        "test": 8,
    }
    assert (first.groupby(["split", "emotion"]).size() > 0).all()


def test_author_interpolation_keeps_waveform_length() -> None:
    result = interpolate_like_author(
        pd.Series([100.0, 120.0, 140.0]).to_numpy(dtype="float32"), 16
    )
    assert result.shape == (16,)
    assert result.dtype.name == "float32"


def test_parse_kaldi_text_archive() -> None:
    matrices = parse_kaldi_text_archive(
        "utt1  [\n  0.1 100\n  0.2 120 ]\nutt2  [ 0.3 140 ]\n"
    )
    assert matrices["utt1"].shape == (2, 2)
    assert matrices["utt1"][:, 1].tolist() == [100.0, 120.0]
    assert matrices["utt2"].shape == (1, 2)


def test_kaldi_wav_copy_is_pcm16(tmp_path: Path) -> None:
    output = tmp_path / "kaldi.wav"
    write_kaldi_pcm16_wav(
        pd.Series([-1.0, -0.25, 0.0, 0.25, 0.999]).to_numpy(dtype="float32"),
        output,
    )
    with wave.open(str(output), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 16000
        assert stream.getnframes() == 5
