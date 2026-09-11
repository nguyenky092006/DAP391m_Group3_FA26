"""Plan complete Wav2Vec2 chunk coverage without loading a model or audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .wav2vec2_chunk_contract import (
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
        selection_sha256,
    )
except ImportError:
    from wav2vec2_chunk_contract import (  # type: ignore
        eligible_manifest_rows,
        file_sha256,
        read_manifest_csv,
        selection_sha256,
    )


PLAN_FIELDS = (
    "chunk_id",
    "eligible_offset",
    "requested_size",
    "actual_size",
    "eligible_end_exclusive",
    "first_source_row_index",
    "last_source_row_index",
    "first_sample_id",
    "last_sample_id",
    "selection_sha256",
    "run_name",
    "report_path",
    "evidence_index_path",
    "command",
)


def build_chunk_plan(
    eligible_rows: list[dict[str, str]], chunk_size: int
) -> list[dict[str, Any]]:
    if not eligible_rows:
        raise ValueError("Cannot plan an empty eligible dataset")
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1")
    plan: list[dict[str, Any]] = []
    for chunk_id, offset in enumerate(range(0, len(eligible_rows), chunk_size)):
        selected = eligible_rows[offset : offset + chunk_size]
        actual_size = len(selected)
        run_name = f"wav2vec2_chunk_o{offset:06d}_n{actual_size:06d}"
        evidence_stem = f"b4_wav2vec2_chunk_o{offset:06d}_n{actual_size:06d}"
        plan.append(
            {
                "chunk_id": chunk_id,
                "eligible_offset": offset,
                "requested_size": chunk_size,
                "actual_size": actual_size,
                "eligible_end_exclusive": offset + actual_size,
                "first_source_row_index": int(selected[0]["source_row_index"]),
                "last_source_row_index": int(selected[-1]["source_row_index"]),
                "first_sample_id": selected[0]["sample_id"],
                "last_sample_id": selected[-1]["sample_id"],
                "selection_sha256": selection_sha256(selected),
                "run_name": run_name,
                "report_path": f"reports/tables/b4/{evidence_stem}_run.json",
                "evidence_index_path": (
                    f"reports/tables/b4/{evidence_stem}_feature_index.csv"
                ),
                "command": (
                    ".\\.venv\\Scripts\\python.exe "
                    "src\\fedecai\\features\\extract_wav2vec2_chunk.py "
                    f"--chunk-offset {offset} --chunk-size {chunk_size}"
                ),
            }
        )
    return plan


def validate_chunk_plan(
    eligible_rows: list[dict[str, str]], plan: list[dict[str, Any]]
) -> dict[str, Any]:
    if not plan:
        raise ValueError("Chunk plan is empty")
    cursor = 0
    covered_ids: list[str] = []
    for expected_chunk_id, chunk in enumerate(plan):
        if int(chunk["chunk_id"]) != expected_chunk_id:
            raise ValueError("Chunk IDs are not consecutive")
        offset = int(chunk["eligible_offset"])
        actual_size = int(chunk["actual_size"])
        if offset != cursor:
            raise ValueError(f"Chunk coverage gap or overlap at eligible offset {cursor}")
        if actual_size < 1:
            raise ValueError("Every planned chunk must contain at least one row")
        selected = eligible_rows[offset : offset + actual_size]
        if len(selected) != actual_size:
            raise ValueError("Chunk extends beyond eligible rows")
        if int(chunk["eligible_end_exclusive"]) != offset + actual_size:
            raise ValueError("Chunk end does not match offset plus actual size")
        if chunk["first_sample_id"] != selected[0]["sample_id"]:
            raise ValueError("Chunk first sample does not match eligible ordering")
        if chunk["last_sample_id"] != selected[-1]["sample_id"]:
            raise ValueError("Chunk last sample does not match eligible ordering")
        if chunk["selection_sha256"] != selection_sha256(selected):
            raise ValueError("Chunk selection SHA-256 does not match its rows")
        covered_ids.extend(row["sample_id"] for row in selected)
        cursor += actual_size
    expected_ids = [row["sample_id"] for row in eligible_rows]
    if cursor != len(eligible_rows):
        raise ValueError(f"Plan covers {cursor} of {len(eligible_rows)} eligible rows")
    if covered_ids != expected_ids:
        raise ValueError("Planned sample order differs from eligible manifest order")
    if len(covered_ids) != len(set(covered_ids)):
        raise ValueError("Plan contains duplicate samples")
    return {
        "status": "PASS",
        "eligible_rows": len(eligible_rows),
        "covered_rows": len(covered_ids),
        "unique_covered_samples": len(set(covered_ids)),
        "gaps": 0,
        "overlaps": 0,
        "first_sample_id": covered_ids[0],
        "last_sample_id": covered_ids[-1],
    }


def plan_sha256(plan: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=256)
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
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 1000:
        parser.error("--chunk-size must be between 1 and 1000")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = args.manifest.resolve()
    eligible = eligible_manifest_rows(read_manifest_csv(manifest_path))
    plan = build_chunk_plan(eligible, args.chunk_size)
    coverage = validate_chunk_plan(eligible, plan)
    report = {
        "status": "PASS",
        "mode": "plan_only",
        "inference_started": False,
        "model_loaded": False,
        "audio_loaded": False,
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
        "eligible_selection_sha256": selection_sha256(eligible),
        "chunk_size": args.chunk_size,
        "chunk_count": len(plan),
        "full_chunks": sum(chunk["actual_size"] == args.chunk_size for chunk in plan),
        "final_chunk_size": plan[-1]["actual_size"],
        "plan_sha256": plan_sha256(plan),
        "coverage": coverage,
        "storage_estimate": {
            "embedding_payload_bytes": len(eligible) * 768 * 4,
            "npy_bytes_using_observed_3200_per_artifact": len(eligible) * 3200,
            "excludes_checkpoint_and_csv_json_evidence": True,
        },
        "plan_csv": args.plan_csv.resolve().relative_to(repo_root).as_posix(),
        "execution_policy": (
            "Sequential CPU chunks only; validate each report before advancing; "
            "reruns must reuse only lineage-and-hash-verified artifacts."
        ),
    }
    atomic_write_csv(plan, args.plan_csv.resolve())
    atomic_write_json(report, args.plan_report.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
