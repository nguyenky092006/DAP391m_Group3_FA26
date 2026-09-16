"""Create the active deterministic sample-level emotion-stratified split.

Speaker identity is retained as metadata and is never used to group samples.
Exact waveform duplication across splits remains a hard failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[3]
SPLIT_VERSION = "emotion_stratified_v1"
DATASET_VERSION = "visec_clean_eligible_v1"
DEFAULT_SEED = 391
SPLIT_COUNTS = {"train": 3681, "validation": 646, "test": 665}
EMOTIONS = ("angry", "happy", "neutral", "sad")
ACCENTS = ("central", "north", "south")


def _sha256_lines(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cross_split_duplicate_count(frame: pd.DataFrame, column: str) -> int:
    usable = frame[column].notna() & frame[column].astype(str).ne("")
    return int(frame.loc[usable].groupby(column)["split"].nunique().gt(1).sum())


def build_split(
    source: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    train_count: int = SPLIT_COUNTS["train"],
    validation_count: int = SPLIT_COUNTS["validation"],
    test_count: int = SPLIT_COUNTS["test"],
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Return all source rows with deterministic split and sample-level CV labels."""
    required = {
        "sample_id", "speaker_id", "emotion", "accent_clean", "audio_sha256",
        "audio_locator", "is_eligible_for_split",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Clean manifest missing columns: {sorted(missing)}")
    if source["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique before splitting")

    output = source.copy()
    eligible = output[output["is_eligible_for_split"].astype(bool)].copy()
    eligible = eligible.sort_values(["sample_id", "source_row_index"], kind="stable")
    expected_total = train_count + validation_count + test_count
    if len(eligible) != expected_total:
        raise ValueError(f"Expected {expected_total:,} eligible samples, found {len(eligible):,}")
    if set(eligible["emotion"]) != set(EMOTIONS):
        raise ValueError(f"Expected emotions {EMOTIONS}, found {sorted(eligible['emotion'].unique())}")
    if set(eligible["accent_clean"]) != set(ACCENTS):
        raise ValueError(f"Expected accents {ACCENTS}, found {sorted(eligible['accent_clean'].unique())}")
    if eligible["audio_sha256"].duplicated().any():
        raise ValueError("Eligible population contains duplicate waveform hashes; resolve in B2 first")

    # Emotion is the scientific target. Accent is included as a secondary
    # balancing stratum so that rare accent/emotion cells remain evaluable.
    primary_strata = eligible["emotion"].astype(str) + "|" + eligible["accent_clean"].astype(str)
    train_ids, held_out_ids = train_test_split(
        eligible["sample_id"],
        train_size=train_count,
        test_size=validation_count + test_count,
        stratify=primary_strata,
        random_state=seed,
    )
    held_out = eligible.set_index("sample_id").loc[held_out_ids]
    held_out_strata = held_out["emotion"].astype(str) + "|" + held_out["accent_clean"].astype(str)
    validation_ids, test_ids = train_test_split(
        held_out.index.to_series(),
        train_size=validation_count,
        test_size=test_count,
        stratify=held_out_strata,
        random_state=seed + 1,
    )

    split_by_id = {
        **{str(value): "train" for value in train_ids},
        **{str(value): "validation" for value in validation_ids},
        **{str(value): "test" for value in test_ids},
    }
    output["split"] = output["sample_id"].astype(str).map(split_by_id).fillna("excluded")
    output["dataset_version"] = DATASET_VERSION
    output["split_version"] = SPLIT_VERSION
    output["split_seed"] = seed
    output["split_protocol"] = "sample_level_emotion_stratified"
    output["test_membership_frozen"] = output["split"].eq("test")
    output["cv_fold"] = pd.NA

    train = output[output["split"].eq("train")].sort_values("sample_id", kind="stable")
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for fold_index, (_, fold_indices) in enumerate(
        splitter.split(train["sample_id"], train["emotion"]), start=1
    ):
        sample_ids = train.iloc[fold_indices]["sample_id"]
        output.loc[output["sample_id"].isin(sample_ids), "cv_fold"] = f"fold_{fold_index}"

    return output


def validate_split(frame: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    eligible = frame[frame["split"].isin(("train", "validation", "test"))].copy()
    split_counts = {name: int((eligible["split"] == name).sum()) for name in SPLIT_COUNTS}
    speakers = {
        name: set(eligible.loc[eligible["split"].eq(name), "speaker_id"].astype(str))
        for name in SPLIT_COUNTS
    }
    overlaps = {
        "train_validation": len(speakers["train"] & speakers["validation"]),
        "train_test": len(speakers["train"] & speakers["test"]),
        "validation_test": len(speakers["validation"] & speakers["test"]),
    }
    duplicate_hashes = _cross_split_duplicate_count(eligible, "audio_sha256")
    duplicate_identities = _cross_split_duplicate_count(eligible, "audio_locator")
    emotion_distribution = {
        split: {emotion: int(((eligible["split"] == split) & (eligible["emotion"] == emotion)).sum()) for emotion in EMOTIONS}
        for split in SPLIT_COUNTS
    }
    accent_distribution = {
        split: {accent: int(((eligible["split"] == split) & (eligible["accent_clean"] == accent)).sum()) for accent in ACCENTS}
        for split in SPLIT_COUNTS
    }
    accent_emotion_distribution = {
        split: {
            accent: {
                emotion: int(((eligible["split"] == split) & (eligible["accent_clean"] == accent) & (eligible["emotion"] == emotion)).sum())
                for emotion in EMOTIONS
            }
            for accent in ACCENTS
        }
        for split in SPLIT_COUNTS
    }
    assigned_once = len(eligible) == eligible["sample_id"].nunique()
    cv_train = eligible[eligible["split"].eq("train")]
    cv_valid = (
        cv_train["cv_fold"].notna().all()
        and set(cv_train["cv_fold"].astype(str)) == {f"fold_{index}" for index in range(1, 6)}
        and eligible.loc[~eligible["split"].eq("train"), "cv_fold"].isna().all()
    )
    validation = {
        "all_samples_assigned_exactly_once": bool(assigned_once),
        "split_counts_match_target": split_counts == SPLIT_COUNTS,
        "split_labels_valid": set(frame["split"]) <= {"train", "validation", "test", "excluded"},
        "split_version_valid": set(frame["split_version"]) == {SPLIT_VERSION},
        "fixed_seed_valid": set(frame["split_seed"]) == {seed},
        "all_emotions_present_per_split": all(all(value > 0 for value in counts.values()) for counts in emotion_distribution.values()),
        "all_accent_emotion_cells_present": all(
            value > 0
            for split in accent_emotion_distribution.values()
            for accent in split.values()
            for value in accent.values()
        ),
        "sample_stratified_cv_valid": bool(cv_valid),
        "duplicate_waveform_cross_split_zero": duplicate_hashes == 0,
        "duplicate_file_identity_cross_split_zero": duplicate_identities == 0,
    }
    status = "PASS" if all(validation.values()) else "FAIL"
    report = {
        "schema_version": "b5.emotion_stratified_split.v1",
        "status": status,
        "dataset_version": DATASET_VERSION,
        "split_version": SPLIT_VERSION,
        "protocol": "deterministic sample-level emotion-stratified train/validation/test",
        "stratification": {
            "primary_target": "emotion",
            "secondary_balance": "accent_clean",
            "implementation": "joint emotion|accent strata to preserve rare accent/emotion support",
        },
        "seed": seed,
        "stage_seeds": {"train_vs_held_out": seed, "validation_vs_test": seed + 1, "cv": seed},
        "target_counts": SPLIT_COUNTS,
        "split_counts": split_counts,
        "ratio": {name: value / len(eligible) for name, value in split_counts.items()},
        "total_source_rows": int(len(frame)),
        "total_eligible_samples": int(len(eligible)),
        "total_emotions": int(eligible["emotion"].nunique()),
        "total_accents": int(eligible["accent_clean"].nunique()),
        "total_unique_speakers": int(eligible["speaker_id"].nunique()),
        "emotion_distribution": emotion_distribution,
        "accent_distribution": accent_distribution,
        "accent_emotion_distribution": accent_emotion_distribution,
        "unique_speakers_by_split": {name: len(value) for name, value in speakers.items()},
        "speaker_overlap": overlaps,
        "speaker_overlap_status": "INFO",
        "speaker_policy": "speaker_id is metadata, not a split grouping constraint",
        "duplicate_waveform_cross_split": duplicate_hashes,
        "duplicate_file_identity_cross_split": duplicate_identities,
        "test_sample_ids_sha256": _sha256_lines(eligible.loc[eligible["split"].eq("test"), "sample_id"].astype(str).tolist()),
        "assignment_sha256": _sha256_lines((eligible["sample_id"].astype(str) + "=" + eligible["split"]).tolist()),
        "validation": validation,
    }
    if status != "PASS":
        raise ValueError(f"Emotion-stratified split validation failed: {validation}")
    return report


def write_outputs(frame: pd.DataFrame, report: dict[str, Any], output: Path, report_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, output)
    temporary_json = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_json, report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/manifests/clean_manifest.csv")
    parser.add_argument("--output", default="data/manifests/emotion_stratified_v1.csv")
    parser.add_argument("--report", default="data/manifests/emotion_stratified_v1_report.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = ROOT / args.input
    output_path = ROOT / args.output
    report_path = ROOT / args.report
    source = pd.read_csv(source_path, dtype={"sample_id": "string", "speaker_id": "string"})
    frame = build_split(source, seed=args.seed)
    report = validate_split(frame, seed=args.seed)
    report.update({
        "source_manifest": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "output_manifest": str(output_path.relative_to(ROOT)).replace("\\", "/"),
    })
    write_outputs(frame, report, output_path, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
