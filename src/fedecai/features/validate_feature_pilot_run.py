"""Reopen and validate a completed B4 feature-pilot run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .extract_feature_pilot import PILOT_FAMILIES, validate_feature_value
except ImportError:
    from extract_feature_pilot import PILOT_FAMILIES, validate_feature_value  # type: ignore


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_feature(family: str, path: Path) -> Any:
    if family == "pitch_contour":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    return np.load(path, allow_pickle=False)


def validate_run(repo_root: Path, run_root: Path) -> dict[str, Any]:
    report = json.loads((run_root / "pilot_report.json").read_text(encoding="utf-8"))
    pilot_manifest = read_csv(run_root / "pilot_manifest.csv")
    feature_index = read_csv(run_root / "feature_index.csv")
    errors: list[str] = []

    selected_samples = int(report["selected_samples"])
    if len(pilot_manifest) != selected_samples:
        errors.append("pilot_manifest row count does not match selected_samples")
    if report.get("failures") != 0:
        errors.append("pilot report contains feature failures")

    families = tuple(report.get("families", {}))
    unknown = sorted(set(families) - set(PILOT_FAMILIES))
    if unknown:
        errors.append(f"unknown feature families: {unknown}")
    expected_records = selected_samples * len(families)
    if len(feature_index) != expected_records:
        errors.append(
            f"feature_index has {len(feature_index)} rows; expected {expected_records}"
        )

    manifest_ids = {row["sample_id"] for row in pilot_manifest}
    seen_pairs: set[tuple[str, str]] = set()
    verified_hashes = 0
    verified_artifacts = 0
    for row in feature_index:
        pair = (row["sample_id"], row["feature_family"])
        if pair in seen_pairs:
            errors.append(f"duplicate feature record: {pair}")
            continue
        seen_pairs.add(pair)
        if row["sample_id"] not in manifest_ids:
            errors.append(f"feature sample is absent from pilot manifest: {row['sample_id']}")
        if row["status"] != "ok" or row["error"]:
            errors.append(f"feature record is not clean: {pair}")
            continue
        artifact_path = (repo_root / row["artifact_path"]).resolve()
        try:
            artifact_path.relative_to(repo_root.resolve())
        except ValueError:
            errors.append(f"artifact is outside the repository: {artifact_path}")
            continue
        if not artifact_path.is_file():
            errors.append(f"artifact does not exist: {artifact_path}")
            continue
        actual_hash = file_sha256(artifact_path)
        if actual_hash != row["artifact_sha256"]:
            errors.append(f"artifact hash mismatch: {artifact_path}")
            continue
        verified_hashes += 1
        try:
            value = _load_feature(row["feature_family"], artifact_path)
            validate_feature_value(row["feature_family"], value)
            verified_artifacts += 1
        except Exception as exc:
            errors.append(f"cannot validate {artifact_path}: {type(exc).__name__}: {exc}")

    expected_pairs = {
        (sample_id, family) for sample_id in manifest_ids for family in families
    }
    missing_pairs = sorted(expected_pairs - seen_pairs)
    if missing_pairs:
        errors.append(f"missing sample-family pairs: {missing_pairs}")

    names_path = run_root / "egemaps_feature_names.json"
    if "egemaps_v02" in families:
        if not names_path.is_file():
            errors.append("eGeMAPS feature-name file is missing")
        else:
            names = json.loads(names_path.read_text(encoding="utf-8"))
            if len(names) != 88 or len(names) != len(set(names)):
                errors.append("eGeMAPS must contain 88 unique feature names")

    if errors:
        raise ValueError("Pilot validation failed:\n" + "\n".join(f"- {e}" for e in errors))
    return {
        "status": "PASS",
        "run_root": run_root.resolve().relative_to(repo_root.resolve()).as_posix(),
        "selected_samples": selected_samples,
        "families": list(families),
        "verified_artifacts": verified_artifacts,
        "verified_hashes": verified_hashes,
        "expected_records": expected_records,
    }


def publish_evidence(run_root: Path, evidence_dir: Path, run_name: str) -> list[str]:
    """Copy only small manifests and reports; feature arrays remain ignored."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    published = []
    for source_name in (
        "pilot_manifest.csv",
        "feature_index.csv",
        "pilot_report.json",
        "egemaps_feature_names.json",
    ):
        source = run_root / source_name
        if not source.is_file():
            continue
        destination = evidence_dir / f"{run_name}_{source_name}"
        shutil.copyfile(source, destination)
        published.append(destination.name)
    return published


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data/interim/features/b4_features_v1",
    )
    parser.add_argument("--evidence-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.root.resolve() / args.run_name
    result = validate_run(repo_root, run_root)
    if args.evidence_dir is not None:
        result["published_evidence"] = publish_evidence(
            run_root, args.evidence_dir.resolve(), args.run_name
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
