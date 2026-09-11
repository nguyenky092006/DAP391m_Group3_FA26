"""Validate the B4 ViSEC feature specification before audio extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FAMILIES = {
    "mfcc40_summary",
    "egemaps_v02",
    "log_mel_80",
    "pitch_contour",
    "wav2vec2_embedding",
}
ALLOWED_STATUSES = {
    "locked_for_pilot",
    "deferred",
    "specified_not_downloaded",
    "pilot_validated",
}
TRAIN_ONLY = "train_only_after_b5"
WAV2VEC2_MODEL_ID = "nguyenvulebinh/wav2vec2-base-vi"
WAV2VEC2_MODEL_REVISION = "86bde51fa76dd7f2c4c1bb28d7475d622639f869"
PITCH_FUSION_REPOSITORY_REVISION = "7fbd5d88f3fbe864bc12aa7a7b71c6f52e5f5953"


class FeatureSpecError(ValueError):
    """Raised when a feature specification violates the B4 contract."""


def load_feature_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_feature_spec(spec: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != "b4.features.v1":
        errors.append("schema_version must be b4.features.v1")

    dataset = spec.get("dataset", {})
    for field in (
        "manifest",
        "eligibility_field",
        "sample_id_field",
        "speaker_id_field",
        "emotion_field",
        "accent_field",
    ):
        if not dataset.get(field):
            errors.append(f"dataset.{field} is required")

    audio = spec.get("audio_contract", {})
    if audio.get("sample_rate_hz") != 16_000:
        errors.append("audio_contract.sample_rate_hz must be 16000")
    if audio.get("mono") is not True:
        errors.append("audio_contract.mono must be true")
    if audio.get("trim_silence") is not False:
        errors.append("B4 pilot must not silently trim flagged silence")
    if audio.get("preserve_full_duration_during_extraction") is not True:
        errors.append("B4 extraction must preserve full duration before the B5 decision")

    leakage = spec.get("leakage_policy", {})
    for field in (
        "fixed_duration_selection_scope",
        "augmentation_scope",
        "normalization_fit_scope",
        "feature_selection_fit_scope",
        "dimensionality_reduction_fit_scope",
    ):
        if leakage.get(field) != TRAIN_ONLY:
            errors.append(f"leakage_policy.{field} must be {TRAIN_ONLY}")
    if leakage.get("test_usage") != "final_evaluation_only":
        errors.append("test data must be reserved for final evaluation")

    families = spec.get("feature_families", [])
    names = [family.get("name") for family in families]
    if len(names) != len(set(names)):
        errors.append("feature family names must be unique")
    missing = sorted(REQUIRED_FAMILIES - set(names))
    if missing:
        errors.append(f"missing required feature families: {missing}")
    for family in families:
        name = family.get("name", "<unnamed>")
        if family.get("status") not in ALLOWED_STATUSES:
            errors.append(f"{name}: unsupported status")
        if family.get("fit_required") is not False:
            errors.append(f"{name}: per-sample extraction must not fit on the dataset")
        if not family.get("extractor"):
            errors.append(f"{name}: extractor is required")
        if family.get("status") == "deferred" and not family.get("decision_required"):
            errors.append(f"{name}: deferred feature requires a documented decision")

    family_by_name = {family.get("name"): family for family in families}
    mfcc = family_by_name.get("mfcc40_summary", {})
    if mfcc.get("parameters", {}).get("n_mfcc") != 40 or mfcc.get("output_dimension") != 80:
        errors.append("mfcc40_summary must contain 40 mean and 40 standard-deviation values")
    egemaps = family_by_name.get("egemaps_v02", {})
    if egemaps.get("output_dimension") != 88:
        errors.append("eGeMAPSv02 Functionals must declare 88 output values")
    log_mel = family_by_name.get("log_mel_80", {})
    if log_mel.get("parameters", {}).get("n_mels") != 80:
        errors.append("log_mel_80 must use 80 Mel bands")
    pitch = family_by_name.get("pitch_contour", {})
    if (
        pitch.get("parameters", {}).get("no_voiced_summary_policy")
        != "voiced_fraction_and_f0_statistics_zero_sentinel"
    ):
        errors.append("pitch_contour must define the deterministic all-unvoiced sentinel")
    wav2vec = family_by_name.get("wav2vec2_embedding", {})
    if wav2vec.get("status") == "deferred" and wav2vec.get("parameters", {}).get("model_id") is not None:
        errors.append("deferred wav2vec2 model_id must remain null until verified")
    if wav2vec.get("status") in {"specified_not_downloaded", "pilot_validated"}:
        parameters = wav2vec.get("parameters", {})
        baseline = wav2vec.get("baseline_reproduction", {})
        if parameters.get("model_id") != WAV2VEC2_MODEL_ID:
            errors.append(f"wav2vec2 model_id must be {WAV2VEC2_MODEL_ID}")
        if parameters.get("model_revision") != WAV2VEC2_MODEL_REVISION:
            errors.append("wav2vec2 model_revision must pin the verified checkpoint")
        if parameters.get("sample_rate_hz") != 16_000:
            errors.append("wav2vec2 sample_rate_hz must be 16000")
        if parameters.get("waveform_normalization") is not True:
            errors.append("wav2vec2 waveform normalization must be enabled")
        if parameters.get("layer_selection") != "final_transformer_hidden_state":
            errors.append("wav2vec2 must use the final transformer hidden state")
        if parameters.get("pooling") != "attention_mask_aware_mean_over_valid_frames":
            errors.append("wav2vec2 embedding pooling must exclude padded frames")
        if parameters.get("hidden_dimension") != 768 or parameters.get("output_dimension") != 768:
            errors.append("wav2vec2 base hidden and pooled dimensions must both be 768")
        if baseline.get("repository_revision") != PITCH_FUSION_REPOSITORY_REVISION:
            errors.append("Pitch-Fusion repository revision must be pinned")
        if baseline.get("model_class") != "transformers.Wav2Vec2ForSequenceClassification":
            errors.append("baseline reproduction must use Wav2Vec2ForSequenceClassification")
        if wav2vec.get("status") == "pilot_validated":
            evidence = wav2vec.get("pilot_evidence", {})
            if evidence.get("scope") != "one_reviewed_utterance_cpu_only":
                errors.append("wav2vec2 pilot evidence must retain its one-sample scope")
            if evidence.get("sample_id") != "visec_hf_000111":
                errors.append("wav2vec2 pilot evidence must identify the reviewed sample")
            if evidence.get("report") != "reports/tables/b4/b4_wav2vec2_one_sample_report.json":
                errors.append("wav2vec2 pilot evidence must identify the tracked report")
            if len(evidence.get("artifact_sha256", "")) != 64:
                errors.append("wav2vec2 pilot artifact SHA-256 must be recorded")

    for transform in spec.get("post_split_transforms", []):
        if transform.get("fit_scope") != TRAIN_ONLY:
            errors.append(f"{transform.get('name', '<unnamed>')}: fit_scope must be {TRAIN_ONLY}")

    required_index_fields = set(
        spec.get("output_contract", {}).get("required_index_fields", [])
    )
    required_lineage = {
        "sample_id",
        "speaker_id",
        "emotion",
        "accent",
        "feature_family",
        "feature_version",
        "artifact_path",
        "artifact_sha256",
        "status",
        "error",
    }
    missing_lineage = sorted(required_lineage - required_index_fields)
    if missing_lineage:
        errors.append(f"output index is missing lineage fields: {missing_lineage}")

    if errors:
        raise FeatureSpecError("\n".join(f"- {error}" for error in errors))
    return {
        "status": "PASS",
        "schema_version": spec["schema_version"],
        "feature_family_count": len(families),
        "pilot_ready": sorted(
            family["name"] for family in families if family["status"] == "locked_for_pilot"
        ),
        "deferred": sorted(
            family["name"] for family in families if family["status"] == "deferred"
        ),
        "specified_not_downloaded": sorted(
            family["name"]
            for family in families
            if family["status"] == "specified_not_downloaded"
        ),
        "pilot_validated": sorted(
            family["name"]
            for family in families
            if family["status"] == "pilot_validated"
        ),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=repo_root / "configs/features/visec_features_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_feature_spec(load_feature_spec(args.spec.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
