"""Snapshot superseded active ladder specifications before reconciliation."""

from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/spec_superseded/20260915_resnet18_ladder_v1"
SOURCES = (
    "PLAN_FEDECAI.md",
    "PROJECT_RULES.md",
    "docs/research/B1_SPEC.md",
    "docs/research/B6_SPEC.md",
    "docs/research/B7_SPEC.md",
    "docs/research/B8_SPEC.md",
    "docs/research/B10_SPEC.md",
    "configs/models/b6_fedecai_v1.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = ARCHIVE / "ARCHIVE_MANIFEST.csv"
    if ARCHIVE.exists():
        raise RuntimeError(f"Refusing to overwrite existing archive: {ARCHIVE}")
    archived_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for relative in SOURCES:
        source = ROOT / relative
        destination = ARCHIVE / "snapshots" / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Hash mismatch after snapshot: {relative}")
        rows.append({
            "original_path": relative,
            "archived_path": destination.relative_to(ROOT).as_posix(),
            "sha256": source_hash,
            "archive_reason": "SUPERSEDED_BY_RESNET18_FEDECAI_LADDER",
            "archived_at": archived_at,
        })
    columns = ["original_path", "archived_path", "sha256", "archive_reason", "archived_at"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"archived_specs={len(rows)}")
    print(manifest.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
