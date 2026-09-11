"""Run all remaining B4 extraction, feature-table, figure, and dashboard work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .build_b4_deliverables import build_deliverables
    from .run_b4_handcrafted_full import run_full_handcrafted
except ImportError:
    from build_b4_deliverables import build_deliverables  # type: ignore
    from run_b4_handcrafted_full import run_full_handcrafted  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    tables = repo_root / "reports/tables/b4"
    print("[1/2] Full resumable MFCC/eGeMAPS/log-Mel/pitch extraction", flush=True)
    handcrafted = run_full_handcrafted(
        repo_root,
        repo_root / "data/manifests/clean_manifest.csv",
        repo_root / "data/raw/visec_hf/train-00000-of-00001.parquet",
        repo_root / "data/interim/features/b4_features_v1/handcrafted_full",
        tables / "b4_handcrafted_full_feature_index.csv",
        tables / "b4_handcrafted_full_extraction_report.json",
        args.batch_size,
    )
    print("[2/2] Combined feature table, three figures, and Plotly dashboard", flush=True)
    deliverables = build_deliverables(repo_root)
    print(
        json.dumps(
            {"status": "PASS", "handcrafted": handcrafted, "deliverables": deliverables},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
