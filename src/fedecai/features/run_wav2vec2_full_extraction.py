"""Run, resume, validate, and aggregate the planned full Wav2Vec2 extraction."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .plan_wav2vec2_full_run import (
        PLAN_FIELDS,
        build_chunk_plan,
        plan_sha256,
        validate_chunk_plan,
    )
    from .wav2vec2_chunk_contract import (
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
    )
except ImportError:
    from plan_wav2vec2_full_run import (  # type: ignore
        PLAN_FIELDS,
        build_chunk_plan,
        plan_sha256,
        validate_chunk_plan,
    )
    from wav2vec2_chunk_contract import (  # type: ignore
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
    )


AGGREGATE_INDEX_FIELDS = (
    "sample_id",
    "source_row_index",
    "speaker_id",
    "emotion",
    "accent",
    "audio_sha256",
    "feature_family",
    "feature_version",
    "model_revision",
    "artifact_path",
    "artifact_sha256",
    "shape",
    "dtype",
    "finite_fraction",
    "status",
    "error",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_write_csv(
    rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_locked_plan(
    repo_root: Path,
    manifest_path: Path,
    plan_csv_path: Path,
    plan_report_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    eligible = eligible_manifest_rows(read_manifest_csv(manifest_path))
    plan_report = json.loads(plan_report_path.read_text(encoding="utf-8"))
    if plan_report.get("status") != "PASS" or plan_report.get("mode") != "plan_only":
        raise ValueError("Full-run plan report is not a passing plan-only artifact")
    if file_sha256(manifest_path) != plan_report.get("manifest_sha256"):
        raise ValueError("Current B2 manifest SHA-256 differs from the locked plan")
    chunk_size = int(plan_report["chunk_size"])
    expected_plan = build_chunk_plan(eligible, chunk_size)
    validate_chunk_plan(eligible, expected_plan)
    if plan_sha256(expected_plan) != plan_report.get("plan_sha256"):
        raise ValueError("Regenerated plan SHA-256 differs from the locked plan")

    tracked_plan = read_csv(plan_csv_path)
    if len(tracked_plan) != len(expected_plan):
        raise ValueError("Tracked plan CSV has an unexpected chunk count")
    for expected, tracked in zip(expected_plan, tracked_plan):
        for field in PLAN_FIELDS:
            if str(expected[field]) != tracked[field]:
                raise ValueError(
                    f"Tracked plan differs at chunk {expected['chunk_id']} field {field}"
                )
    return eligible, expected_plan, plan_report


def validate_chunk_evidence(
    repo_root: Path,
    chunk: dict[str, Any],
    eligible_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    report_path = repo_root / chunk["report_path"]
    index_path = repo_root / chunk["evidence_index_path"]
    if not report_path.is_file() or not index_path.is_file():
        raise ValueError(f"Chunk {chunk['chunk_id']} evidence files are missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index = read_csv(index_path)
    offset = int(chunk["eligible_offset"])
    actual_size = int(chunk["actual_size"])
    selected = eligible_rows[offset : offset + actual_size]
    selected_ids = [row["sample_id"] for row in selected]

    if report.get("status") != "PASS" or report.get("failures") != 0:
        raise ValueError(f"Chunk {chunk['chunk_id']} report did not pass")
    if report.get("selected_samples") != selected_ids:
        raise ValueError(f"Chunk {chunk['chunk_id']} selected-sample order changed")
    selection = report.get("selection", {})
    if selection.get("selection_sha256") != chunk["selection_sha256"]:
        raise ValueError(f"Chunk {chunk['chunk_id']} selection hash changed")
    if int(selection.get("requested_offset", -1)) != offset:
        raise ValueError(f"Chunk {chunk['chunk_id']} offset changed")
    if int(selection.get("actual_size", -1)) != actual_size:
        raise ValueError(f"Chunk {chunk['chunk_id']} actual size changed")
    if len(index) != actual_size:
        raise ValueError(f"Chunk {chunk['chunk_id']} index row count changed")
    if [row["sample_id"] for row in index] != selected_ids:
        raise ValueError(f"Chunk {chunk['chunk_id']} index ordering changed")

    report_hashes = report.get("artifact_hashes", {})
    for manifest_row, index_row in zip(selected, index):
        for field in (
            "sample_id",
            "source_row_index",
            "speaker_id",
            "emotion",
            "accent",
            "audio_sha256",
        ):
            if index_row[field] != manifest_row[field]:
                raise ValueError(
                    f"Chunk {chunk['chunk_id']} lineage mismatch for {field}"
                )
        if index_row["status"] != "ok" or index_row["error"]:
            raise ValueError(f"Chunk {chunk['chunk_id']} contains a failed index row")
        if index_row["shape"] != "768" or index_row["dtype"] != "float32":
            raise ValueError(f"Chunk {chunk['chunk_id']} contains an invalid shape/dtype")
        if index_row["finite_fraction"] != "1.00000000":
            raise ValueError(f"Chunk {chunk['chunk_id']} contains non-finite values")
        if len(index_row["artifact_sha256"]) != 64:
            raise ValueError(f"Chunk {chunk['chunk_id']} has an invalid artifact hash")
        if report_hashes.get(index_row["sample_id"]) != index_row["artifact_sha256"]:
            raise ValueError(f"Chunk {chunk['chunk_id']} report/index hash mismatch")
    return report, index


def run_chunk(repo_root: Path, chunk: dict[str, Any]) -> None:
    command = [
        sys.executable,
        str(repo_root / "src/fedecai/features/extract_wav2vec2_chunk.py"),
        "--chunk-offset",
        str(chunk["eligible_offset"]),
        "--chunk-size",
        str(chunk["requested_size"]),
        "--report-path",
        str(repo_root / chunk["report_path"]),
        "--evidence-index",
        str(repo_root / chunk["evidence_index_path"]),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            f"Chunk {chunk['chunk_id']} failed with exit code "
            f"{completed.returncode}:\n{detail}"
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data/manifests/clean_manifest.csv",
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_full_chunk_plan.csv",
    )
    parser.add_argument(
        "--plan-report",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_full_chunk_plan.json",
    )
    parser.add_argument(
        "--aggregate-index",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_full_feature_index.csv",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_full_extraction_report.json",
    )
    args = parser.parse_args()
    if args.start_chunk < 0:
        parser.error("--start-chunk must be non-negative")
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    eligible, plan, plan_report = validate_locked_plan(
        repo_root,
        args.manifest.resolve(),
        args.plan_csv.resolve(),
        args.plan_report.resolve(),
    )
    if args.start_chunk >= len(plan):
        raise ValueError(f"--start-chunk must be below {len(plan)}")
    stop = (
        len(plan)
        if args.max_chunks is None
        else min(len(plan), args.start_chunk + args.max_chunks)
    )
    selected_plan = plan[args.start_chunk : stop]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "dry_run",
                    "planned_chunks": len(selected_plan),
                    "first_chunk": selected_plan[0]["chunk_id"],
                    "last_chunk": selected_plan[-1]["chunk_id"],
                    "inference_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    processed = resumed = invalidated = 0
    for position, chunk in enumerate(selected_plan, start=1):
        print(
            f"[{position}/{len(selected_plan)}] chunk={chunk['chunk_id']} "
            f"offset={chunk['eligible_offset']} size={chunk['actual_size']}",
            flush=True,
        )
        run_chunk(repo_root, chunk)
        report, _ = validate_chunk_evidence(repo_root, chunk, eligible)
        processed += int(report["processed"])
        resumed += int(report["resumed"])
        invalidated += int(report["invalidated"])
        print(
            f"  PASS processed={report['processed']} resumed={report['resumed']} "
            f"invalidated={report['invalidated']}",
            flush=True,
        )

    if args.start_chunk != 0 or stop != len(plan):
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "partial_execution",
                    "executed_chunks": len(selected_plan),
                    "processed": processed,
                    "resumed": resumed,
                    "invalidated": invalidated,
                    "next_chunk": stop if stop < len(plan) else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    aggregate_rows: list[dict[str, str]] = []
    chunk_summaries = []
    for chunk in plan:
        report, index = validate_chunk_evidence(repo_root, chunk, eligible)
        aggregate_rows.extend(index)
        chunk_summaries.append(
            {
                "chunk_id": chunk["chunk_id"],
                "processed": report["processed"],
                "resumed": report["resumed"],
                "invalidated": report["invalidated"],
                "failures": report["failures"],
            }
        )
    if len(aggregate_rows) != len(eligible):
        raise ValueError("Aggregate index does not cover every eligible row")
    if [row["sample_id"] for row in aggregate_rows] != [
        row["sample_id"] for row in eligible
    ]:
        raise ValueError("Aggregate index order differs from eligible manifest order")
    if len({row["sample_id"] for row in aggregate_rows}) != len(eligible):
        raise ValueError("Aggregate index contains duplicate samples")

    aggregate_index_path = args.aggregate_index.resolve()
    atomic_write_csv(aggregate_rows, aggregate_index_path, AGGREGATE_INDEX_FIELDS)
    aggregate_report = {
        "status": "PASS",
        "feature_family": "wav2vec2_embedding",
        "feature_version": "b4_features_v1",
        "eligible_rows": len(eligible),
        "indexed_artifacts": len(aggregate_rows),
        "unique_samples": len({row["sample_id"] for row in aggregate_rows}),
        "chunk_count": len(plan),
        "failures": 0,
        "processed_this_invocation": processed,
        "resumed_this_invocation": resumed,
        "invalidated_this_invocation": invalidated,
        "manifest_sha256": plan_report["manifest_sha256"],
        "plan_sha256": plan_report["plan_sha256"],
        "aggregate_index_path": aggregate_index_path.relative_to(repo_root).as_posix(),
        "aggregate_index_sha256": file_sha256(aggregate_index_path),
        "chunks": chunk_summaries,
    }
    atomic_write_json(aggregate_report, args.aggregate_report.resolve())
    print(json.dumps(aggregate_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
