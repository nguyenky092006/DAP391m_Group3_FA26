"""Extract one deterministic, resumable Wav2Vec2 chunk from the B2 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .extract_wav2vec2_resume_pilot import run_resume_pilot
    from .wav2vec2_chunk_contract import (
        file_sha256,
        read_manifest_csv,
        select_eligible_chunk,
        selection_sha256,
    )
except ImportError:
    from extract_wav2vec2_resume_pilot import run_resume_pilot  # type: ignore
    from wav2vec2_chunk_contract import (  # type: ignore
        file_sha256,
        read_manifest_csv,
        select_eligible_chunk,
        selection_sha256,
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-name")
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--evidence-index", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data/manifests/clean_manifest.csv",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=repo_root / "data/raw/visec_hf/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--spec", type=Path, default=repo_root / "configs/features/visec_features_v1.json"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=repo_root / "data/interim/models/huggingface"
    )
    args = parser.parse_args()
    if args.chunk_offset < 0:
        parser.error("--chunk-offset must be non-negative")
    if not 1 <= args.chunk_size <= 1000:
        parser.error("--chunk-size must be between 1 and 1000")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = args.manifest.resolve()
    selected, eligible_count = select_eligible_chunk(
        read_manifest_csv(manifest_path), args.chunk_offset, args.chunk_size
    )
    actual_count = len(selected)
    run_name = args.run_name or (
        f"wav2vec2_chunk_o{args.chunk_offset:06d}_n{actual_count:06d}"
    )
    run_root = repo_root / "data/interim/features/b4_features_v1" / run_name
    default_stem = f"b4_wav2vec2_chunk_o{args.chunk_offset:06d}_n{actual_count:06d}"
    report_path = (
        args.report_path.resolve()
        if args.report_path is not None
        else repo_root / "reports/tables/b4" / f"{default_stem}_run.json"
    )
    evidence_index = (
        args.evidence_index.resolve()
        if args.evidence_index is not None
        else repo_root / "reports/tables/b4" / f"{default_stem}_feature_index.csv"
    )
    report = run_resume_pilot(
        repo_root=repo_root,
        spec_path=args.spec.resolve(),
        pilot_manifest_path=manifest_path,
        parquet_path=args.parquet.resolve(),
        cache_dir=args.cache_dir.resolve(),
        run_root=run_root.resolve(),
        report_path=report_path,
        evidence_index_path=evidence_index,
        sample_ids=tuple(row["sample_id"] for row in selected),
        threads=args.threads,
        selected_rows=selected,
        scope=(
            "One deterministic B2-eligible manifest chunk; per-sample CPU "
            "extraction from the local pinned checkpoint; no fitted transforms."
        ),
        selection_metadata={
            "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
            "manifest_sha256": file_sha256(manifest_path),
            "eligible_rows": eligible_count,
            "requested_offset": args.chunk_offset,
            "requested_size": args.chunk_size,
            "actual_size": actual_count,
            "first_source_row_index": int(selected[0]["source_row_index"]),
            "last_source_row_index": int(selected[-1]["source_row_index"]),
            "selection_sha256": selection_sha256(selected),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
