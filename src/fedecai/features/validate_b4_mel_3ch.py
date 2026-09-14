"""Validate the B4 three-channel Mel adapter against existing log-Mel files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from fedecai.features.mel_3ch import MEL_3CH_MODE, build_mel_3ch
from fedecai.features.validate_feature_spec import load_feature_spec, validate_feature_spec


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_all(repo_root: Path, config_path: Path, index_path: Path) -> dict[str, Any]:
    spec = load_feature_spec(config_path)
    validate_feature_spec(spec)
    family = next(item for item in spec["feature_families"] if item["name"] == "mel_3ch")
    mode = family["parameters"]["mel_3ch_mode"]
    with index_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    log_mel_rows = [
        row for row in rows if row["feature_family"] == "log_mel_80" and row["status"] == "ok"
    ]
    if not log_mel_rows:
        raise ValueError("No valid log_mel_80 rows found")
    min_frames: int | None = None
    max_frames = 0
    for row in log_mel_rows:
        source = repo_root / row["artifact_path"]
        value = np.load(source, allow_pickle=False)
        converted = build_mel_3ch(value, mode=mode)
        if converted.shape != (3, 80, value.shape[1]) or not np.isfinite(converted).all():
            raise ValueError(f"Invalid three-channel conversion: {source}")
        min_frames = value.shape[1] if min_frames is None else min(min_frames, value.shape[1])
        max_frames = max(max_frames, value.shape[1])
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "status": "PASS",
        "contract_revision": spec["contract_revision"],
        "mel_3ch_mode": MEL_3CH_MODE,
        "normalization": "fixed_db_range_to_unit_interval",
        "output_layout": "channels_first",
        "output_channels": 3,
        "validated_artifacts": len(log_mel_rows),
        "min_source_frames": min_frames,
        "max_source_frames": max_frames,
        "materialization": family["materialization"],
        "target_frames_selection_scope": family["padding_or_crop"]["target_frames_selection_scope"],
        "config_path": config_path.relative_to(repo_root).as_posix(),
        "config_sha256": file_sha256(config_path),
        "source_index_path": index_path.relative_to(repo_root).as_posix(),
        "source_index_sha256": file_sha256(index_path),
        "git_commit": commit,
        "worktree_dirty": dirty,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/features/visec_features_v1.json")
    parser.add_argument(
        "--index",
        type=Path,
        default=root / "reports/tables/b4/b4_handcrafted_full_feature_index.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports/tables/b4/b4_mel_3ch_contract_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    report = validate_all(root, args.config.resolve(), args.index.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
