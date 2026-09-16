"""Snapshot superseded active FedECAI implementation before E2-E7 replacement."""

from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/spec_superseded/20260915_resnet18_e2_e7_v1"
SOURCES = (
    ("src/fedecai/models/run_b6_fedecai_pipeline.py", "legacy_runner"),
    ("src/fedecai/models/b6_deep_models.py", "legacy_model_source"),
    ("configs/models/b6_fedecai_v1.json", "superseded_plan_config"),
    ("tests/test_b6_fedecai_pipeline.py", "legacy_test"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if ARCHIVE.exists():
        raise RuntimeError(f"Refusing to overwrite archive: {ARCHIVE}")
    archived_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for relative, file_type in SOURCES:
        source = ROOT / relative
        destination = ARCHIVE / "snapshots" / relative
        source_hash = sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Archive hash mismatch: {relative}")
        rows.append({
            "original_path": relative,
            "archived_path": destination.relative_to(ROOT).as_posix(),
            "sha256": source_hash,
            "file_type": file_type,
            "archive_reason": "SUPERSEDED_BY_RESNET18_E2_E7_IMPLEMENTATION",
            "archived_at": archived_at,
        })
    manifest = ARCHIVE / "ARCHIVE_MANIFEST.csv"
    fields = ["original_path", "archived_path", "sha256", "file_type", "archive_reason", "archived_at"]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"ARCHIVED={len(rows)}")
    print(manifest.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
