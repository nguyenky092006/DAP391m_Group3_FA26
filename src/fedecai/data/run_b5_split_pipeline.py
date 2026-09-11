"""Run all B5 split-v2, grouped-CV, and federated-partition deliverables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from .build_b5_partitions import (
        DEFAULT_CV_ATTEMPTS,
        DEFAULT_CV_FOLDS,
        assign_federated_clients,
        assign_grouped_cv,
        build_partition_report,
    )
    from .create_speaker_split import DEFAULT_SEED, SPEAKER_COUNTS, build_speaker_profiles, read_manifest
    from .create_speaker_split_v2 import (
        DEFAULT_RESTARTS,
        DEFAULT_STEPS,
        apply_assignment_v2,
        build_report_v2,
        optimize_assignment_v2,
    )
except ImportError:  # Direct execution from repository root.
    from build_b5_partitions import (  # type: ignore
        DEFAULT_CV_ATTEMPTS,
        DEFAULT_CV_FOLDS,
        assign_federated_clients,
        assign_grouped_cv,
        build_partition_report,
    )
    from create_speaker_split import (  # type: ignore
        DEFAULT_SEED,
        SPEAKER_COUNTS,
        build_speaker_profiles,
        read_manifest,
    )
    from create_speaker_split_v2 import (  # type: ignore
        DEFAULT_RESTARTS,
        DEFAULT_STEPS,
        apply_assignment_v2,
        build_report_v2,
        optimize_assignment_v2,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "data/manifests/clean_manifest.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--cv-attempts", type=int, default=DEFAULT_CV_ATTEMPTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    v1_paths = (
        repo_root / "data/manifests/split_manifest.csv",
        repo_root / "data/manifests/split_report.json",
    )
    for path in v1_paths:
        if not path.exists():
            raise FileNotFoundError(f"Protected split-v1 artifact is missing: {path}")
    v1_before = {str(path.relative_to(repo_root)): file_sha256(path) for path in v1_paths}

    print("[1/4] Optimizing coverage-complete speaker-disjoint split v2")
    rows = read_manifest(args.input.resolve())
    profiles = build_speaker_profiles(rows)
    forced_train_speaker = max(profiles, key=lambda speaker_id: profiles[speaker_id]["samples"])
    assignment, objective_score = optimize_assignment_v2(
        profiles, SPEAKER_COUNTS, args.seed, args.restarts, args.steps, forced_train_speaker
    )

    print("[2/4] Repeating split optimization to verify deterministic output")
    repeated_assignment, repeated_score = optimize_assignment_v2(
        profiles, SPEAKER_COUNTS, args.seed, args.restarts, args.steps, forced_train_speaker
    )
    if assignment != repeated_assignment or objective_score != repeated_score:
        raise RuntimeError("Split-v2 reproducibility check failed; no outputs were written")

    split_rows = apply_assignment_v2(rows, assignment, args.seed)
    split_report = build_report_v2(
        split_rows,
        profiles,
        assignment,
        objective_score,
        args.seed,
        forced_train_speaker,
        args.restarts,
        args.steps,
    )

    print("[3/4] Building grouped CV folds and accent-domain federated clients")
    train_profiles = {
        speaker_id: profile
        for speaker_id, profile in profiles.items()
        if assignment[speaker_id] == "train"
    }
    cv_assignment, cv_score = assign_grouped_cv(
        train_profiles, args.seed, args.cv_folds, args.cv_attempts
    )
    client_assignment = assign_federated_clients(profiles, assignment)
    partition_report = build_partition_report(
        split_rows,
        cv_assignment,
        client_assignment,
        cv_score,
        args.seed,
        args.cv_folds,
    )
    enriched_rows = []
    for source in split_rows:
        row = dict(source)
        speaker_id = row["speaker_id"]
        row["cv_fold"] = cv_assignment.get(speaker_id, "") if row["split"] == "train" else ""
        row["federated_client_id"] = (
            client_assignment.get(speaker_id, "") if row["split"] == "train" else ""
        )
        enriched_rows.append(row)

    v1_after = {str(path.relative_to(repo_root)): file_sha256(path) for path in v1_paths}
    split_report["pipeline_validation"] = {
        "same_seed_repeat_identical": True,
        "protected_v1_hashes_unchanged": v1_before == v1_after,
        "protected_v1_sha256": v1_after,
        **partition_report["validation"],
    }
    if not all(
        value
        for key, value in split_report["pipeline_validation"].items()
        if key != "protected_v1_sha256"
    ):
        raise RuntimeError(f"B5 final validation failed: {split_report['pipeline_validation']}")

    speaker_rows = [
        {
            "speaker_id": speaker_id,
            "split": assignment[speaker_id],
            "cv_fold": cv_assignment.get(speaker_id, ""),
            "federated_client_id": client_assignment.get(speaker_id, ""),
            "accent": profiles[speaker_id]["accent"],
            "gender": profiles[speaker_id]["gender"],
            "samples": profiles[speaker_id]["samples"],
        }
        for speaker_id in sorted(profiles, key=int)
    ]
    cv_summary = [
        {
            "fold": item["fold"],
            "samples": item["samples"],
            "speakers": item["speakers"],
            **{f"accent_{key}": value for key, value in item["accent_samples"].items()},
            **{f"emotion_{key}": value for key, value in item["emotion_samples"].items()},
        }
        for item in partition_report["cv"]["folds"]
    ]
    client_summary = [
        {
            "client_id": item["client_id"],
            "samples": item["samples"],
            "speakers": item["speakers"],
            **{f"emotion_{key}": value for key, value in item["emotion_samples"].items()},
        }
        for item in partition_report["federated_clients"]["clients"]
    ]
    pipeline_summary = {
        "status": "PASS",
        "split_version": split_report["split_version"],
        "assignment_sha256": split_report["assignment_sha256"],
        "eligible_rows": sum(row["split"] != "excluded" for row in split_rows),
        "excluded_rows": sum(row["split"] == "excluded" for row in split_rows),
        "split_distributions": split_report["distributions"],
        "partitioning": partition_report,
        "validation": split_report["pipeline_validation"],
    }

    print("[4/4] Publishing versioned manifests, evidence tables, and B5 report")
    manifest_dir = repo_root / "data/manifests"
    report_dir = repo_root / "reports/tables/b5"
    write_csv_atomic(split_rows, manifest_dir / "split_manifest_v2.csv")
    write_csv_atomic(enriched_rows, manifest_dir / "split_manifest_v2_enriched.csv")
    write_text_atomic(json_text(split_report), manifest_dir / "split_report_v2.json")
    write_csv_atomic(split_report["cross_strata"], report_dir / "b5_split_v2_cross_strata.csv")
    write_csv_atomic(speaker_rows, report_dir / "b5_speaker_assignments_v2.csv")
    write_csv_atomic(cv_summary, report_dir / "b5_grouped_cv_summary.csv")
    write_csv_atomic(client_summary, report_dir / "b5_federated_client_summary.csv")
    write_text_atomic(json_text(pipeline_summary), report_dir / "b5_pipeline_summary.json")

    documentation = f"""# B5 Split v2 Finalization Report

Generated by `run_b5_split_pipeline.py` with seed `{args.seed}`.

## Result

- Status: **PASS**
- Split version: `{split_report['split_version']}`
- Speaker assignment SHA-256: `{split_report['assignment_sha256']}`
- Same-seed repeat: identical
- Protected v1 files: unchanged
- Split x accent x emotion coverage: complete (36/36 cells)
- Grouped CV: {args.cv_folds} folds over training speakers only, with no speaker leakage
- Federated partition: three non-IID accent-domain clients over training speakers only

The frozen split contains {split_report['distributions']['train']['samples']:,}
train, {split_report['distributions']['validation']['samples']:,} validation,
and {split_report['distributions']['test']['samples']:,} test samples, with
exactly 103/22/22 speakers. All 288 B2-excluded rows remain excluded.

## Grouped-CV interpretation

Every CV fold contains every accent and every emotion at the marginal level.
Full accent-emotion cross coverage is not required when a training cross-stratum
has fewer speakers than the number of folds. Missing cells are recorded in
`b5_pipeline_summary.json`.

Speaker {forced_train_speaker} contributes
{profiles[forced_train_speaker]['samples']:,} samples and remains a single group.
Consequently, its fold is necessarily much larger than the other folds. Model
selection should report macro-average and per-fold scores, using the frozen
external validation/test sets as the primary comparison evidence.

The accent-domain federated clients are intentionally non-IID and imbalanced.
Federated evaluation should report macro-client and sample-weighted metrics.

## Canonical artifacts

- `data/manifests/split_manifest_v2.csv`: frozen train/validation/test split.
- `data/manifests/split_manifest_v2_enriched.csv`: same split plus `cv_fold` and
  `federated_client_id` for training rows.
- `data/manifests/split_report_v2.json`: validation and frozen assignment evidence.
- `reports/tables/b5/`: cross-stratum, speaker, CV, client, and pipeline summaries.

The released ViSEC metadata does not establish whether speaker ID 0 represents one
verified person or a catch-all ID. Therefore, speaker-disjoint claims remain limited
to the released `speaker_id` metadata. The dominant ID is intentionally kept in train.
"""
    write_text_atomic(documentation, repo_root / "docs/research/B5_split_v2_finalization.md")
    print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
