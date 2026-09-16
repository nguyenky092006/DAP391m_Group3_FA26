"""Lock the approved B6 benchmark and safely archive superseded evidence.

The apply path copies each file to a versioned archive, verifies SHA-256, and
only then removes the source. It never overwrites an archive destination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_CONFIG = ROOT / "configs/models/b6_final_lock_v1.json"
FINAL_DIR = ROOT / "reports/tables/b6_final"
TABLE_COLUMNS = [
    "experiment_id", "model", "training_policy", "dataset_version",
    "split_version", "seed", "overall_macro_f1", "uar", "weighted_f1",
    "accuracy", "north_macro_f1", "central_macro_f1", "south_macro_f1",
    "worst_accent_macro_f1", "best_accent_macro_f1", "accent_gap",
    "accent_std", "total_parameters", "trainable_parameters", "model_size_mb",
    "training_time_sec", "feature_extraction_time_sec", "model_training_time_sec",
    "end_to_end_training_pipeline_time_sec", "latency_mean_ms",
    "latency_median_ms", "latency_std_ms", "latency_p95_ms", "checkpoint_path",
    "checkpoint_sha256", "prediction_path", "prediction_sha256", "artifact_status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_final_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    expected_ids: set[str] | None = None
    resnet_expected = {
        "overall_macro_f1": 0.6218351697860597,
        "uar": 0.6216308745720511,
        "weighted_f1": 0.6367225088173246,
        "accuracy": 0.6439628482972136,
        "north_macro_f1": 0.5651249841390686,
        "central_macro_f1": 0.5184027777777778,
        "south_macro_f1": 0.6469934829083388,
        "worst_accent_macro_f1": 0.5184027777777778,
        "best_accent_macro_f1": 0.6469934829083388,
        "accent_gap": 0.12859070513056103,
        "accent_std": 0.05314653234635135,
        "total_parameters": 11178564,
        "model_size_mb": 42.71912670135498,
        "training_time_sec": 158.98527500004275,
        "latency_median_ms": 3.513,
    }
    for item in config["final_models"]:
        experiment_id = item["experiment_id"]
        metrics_path = ROOT / item["metrics"]
        prediction_path = ROOT / item["predictions"]
        checkpoint_path = ROOT / item["checkpoint"]
        report_path = ROOT / item["report"]
        required = (metrics_path, prediction_path, checkpoint_path, report_path)
        if not all(path.is_file() for path in required):
            missing = [rel(path) for path in required if not path.is_file()]
            raise RuntimeError(f"Missing FINAL_ACTIVE evidence for {experiment_id}: {missing}")
        metrics = read_json(metrics_path)
        report = read_json(report_path)
        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        sample_ids = [row["sample_id"] for row in predictions]
        splits = {row["split"] for row in predictions}
        current_ids = set(sample_ids)
        if len(predictions) != config["validation_sample_count"] or len(current_ids) != len(predictions):
            raise RuntimeError(f"Invalid validation prediction support for {experiment_id}")
        if splits != {"validation"}:
            raise RuntimeError(f"Non-validation prediction found for {experiment_id}: {splits}")
        if expected_ids is None:
            expected_ids = current_ids
        elif current_ids != expected_ids:
            raise RuntimeError(f"Validation sample IDs differ for {experiment_id}")
        checkpoint_hash = sha256(checkpoint_path)
        report_hash = (report.get("lineage") or {}).get("checkpoint_sha256")
        hash_contract = report_hash == checkpoint_hash if report_hash else True
        if not hash_contract:
            raise RuntimeError(f"Checkpoint/report hash mismatch for {experiment_id}")
        if report.get("test_samples_loaded", 0) != 0:
            raise RuntimeError(f"Test samples were loaded by {experiment_id}")
        row = {column: metrics.get(column) for column in TABLE_COLUMNS}
        row.update({
            "experiment_id": experiment_id,
            "model": item["model"],
            "training_policy": item["training_policy"],
            "dataset_version": config["dataset_version"],
            "split_version": config["split_version"],
            "seed": config["seed"],
            "checkpoint_path": item["checkpoint"],
            "checkpoint_sha256": checkpoint_hash,
            "prediction_path": item["predictions"],
            "prediction_sha256": sha256(prediction_path),
            "artifact_status": "FINAL_ACTIVE",
        })
        if experiment_id == "b6_resnet18_mel3_finetune_seed391":
            for key, expected in resnet_expected.items():
                if row[key] != expected:
                    raise RuntimeError(f"Locked ResNet18 value mismatch: {key}={row[key]!r}, expected {expected!r}")
        rows.append(row)
        gates.append({
            "experiment_id": experiment_id,
            "prediction_count": len(predictions),
            "unique_sample_ids": len(current_ids),
            "split": "validation",
            "checkpoint_exists": True,
            "checkpoint_hash_matches_source_report": hash_contract,
            "source_report_carries_checkpoint_hash": report_hash is not None,
            "test_samples_loaded": 0,
        })
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 FINAL_ACTIVE rows, found {len(rows)}")
    return rows, gates


def reason_for(path: Path) -> tuple[str, str, str, str]:
    text = rel(path).lower()
    name = path.name.lower()
    experiment_id = ""
    model = ""
    if "pitchfusion_frozen_stable_controlled" in text:
        experiment_id, model = "b6_pitchfusion_frozen_stable_controlled_seed391", "pitch_fusion_paper_aligned"
        reason = "CONTROLLED_EXPERIMENT_NOT_SELECTED"
    elif "wav2vec2_raw_finetune12_controlled" in text:
        experiment_id, model = "b6_wav2vec2_raw_finetune12_controlled_seed391", "wav2vec2_raw"
        reason = "CONTROLLED_EXPERIMENT_NOT_SELECTED"
    elif "pitchfusion_raw_pitch_finetune" in text:
        experiment_id, model = "b6_pitchfusion_raw_pitch_finetune_seed391", "pitch_fusion_paper_aligned"
        reason = "INVALIDATED_CHECKPOINT_LINEAGE" if "checkpoints/" in text else "FAILED_REGRESSION_RUN"
    elif "wav2vec2_raw_finetune_seed391" in text:
        experiment_id, model = "b6_wav2vec2_raw_finetune_seed391", "wav2vec2_raw"
        reason = "INVALIDATED_CHECKPOINT_LINEAGE" if "checkpoints/" in text else "FAILED_REGRESSION_RUN"
    elif "cnn2d_logmel_performance" in text:
        experiment_id, model, reason = "b6_cnn2d_logmel_performance_seed391", "cnn2d_logmel", "SUPERSEDED_BY_FINAL_B6"
    elif "resnet18_mel3_seed391" in text and "finetune" not in text:
        experiment_id, model, reason = "b6_resnet18_mel3_seed391", "resnet18_mel3", "SUPERSEDED_BY_FINAL_B6"
    elif "mobilenet_v3_mel3_seed391" in text and "finetune" not in text:
        experiment_id, model, reason = "b6_mobilenet_v3_mel3_seed391", "mobilenet_v3_mel3", "SUPERSEDED_BY_FINAL_B6"
    elif "ranking" in name or "comparison" in name or "candidate" in name or "figure" in text:
        reason = "OBSOLETE_RANKING"
    elif "audit" in text or "preflight" in text:
        reason = "OBSOLETE_REPORT"
    else:
        reason = "OBSOLETE_REPORT"
    if "smoke" in text:
        reason = "SMOKE_ARTIFACT"
    artifact_type = "log" if text.startswith("logs/") else path.suffix.lower().lstrip(".") or "file"
    if "checkpoints/" in text:
        artifact_type = "checkpoint"
    elif "predictions/" in text:
        artifact_type = "predictions"
    elif "metrics/" in text:
        artifact_type = "metrics"
    elif "audit" in text:
        artifact_type = "audit"
    return experiment_id, model, artifact_type, reason


def archive_candidates(config: dict[str, Any]) -> list[Path]:
    active = {str((ROOT / item[key]).resolve()).lower() for item in config["final_models"] for key in ("checkpoint", "metrics", "predictions", "report")}
    keep_extra = {
        str((ROOT / "reports/tables/b6/svm_rbf_cv_results.csv").resolve()).lower(),
        str((ROOT / "reports/tables/b6/xgboost_cv_results.csv").resolve()).lower(),
        str((ROOT / "reports/figures/b6/final_centralized_b6_ranking.png").resolve()).lower(),
    }
    keep_runs = {
        "20260914_214300_centralized_infrastructure_seed391",
        "20260915_001624_b6_mobilenet_v3_mel3_finetune_seed391_train",
        "20260915_002014_b6_resnet18_mel3_finetune_seed391_train",
    }
    roots = [
        ROOT / "artifacts/checkpoints", ROOT / "artifacts/metrics",
        ROOT / "artifacts/predictions", ROOT / "artifacts/audit", ROOT / "artifacts/b6",
        ROOT / "reports/tables/b6", ROOT / "reports/tables/b6_performance",
        ROOT / "reports/tables/b6_regression_controls", ROOT / "reports/figures/b6",
        ROOT / "reports/tables/b7",
    ]
    candidates: set[Path] = set()
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and str(path.resolve()).lower() not in active | keep_extra:
                candidates.add(path)
    runs_root = ROOT / "logs/runs"
    if runs_root.is_dir():
        for run_dir in runs_root.iterdir():
            if run_dir.is_dir() and run_dir.name not in keep_runs:
                candidates.update(path for path in run_dir.rglob("*") if path.is_file())
    return sorted(candidates, key=lambda value: rel(value))


def archive_files(paths: list[Path], archive_root: Path, apply: bool) -> list[dict[str, Any]]:
    archived_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for source in paths:
        destination = archive_root / rel(source)
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite archive destination: {destination}")
        source_hash = sha256(source)
        experiment_id, model, artifact_type, reason = reason_for(source)
        status = "SMOKE_TEST_ONLY" if reason == "SMOKE_ARTIFACT" else "SUPERSEDED"
        rows.append({
            "original_path": rel(source),
            "archived_path": destination.relative_to(ROOT).as_posix(),
            "experiment_id": experiment_id,
            "model": model,
            "artifact_type": artifact_type,
            "sha256": source_hash,
            "original_status": status,
            "archive_reason": reason,
            "archived_at": archived_at,
        })
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != source_hash:
                raise RuntimeError(f"Archive hash verification failed: {source}")
            source.unlink()
    return rows


def remove_empty_directories() -> None:
    roots = [
        ROOT / "artifacts/b6", ROOT / "reports/tables/b6_regression_controls",
        ROOT / "reports/figures/b6", ROOT / "logs/runs",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
            if not any(directory.iterdir()):
                directory.rmdir()
        if root != ROOT / "logs/runs" and root.is_dir() and not any(root.iterdir()):
            root.rmdir()


def finalize_registries(config: dict[str, Any], archive_root: Path) -> None:
    """Archive the mixed historical registries and retain only active B6 rows."""
    manifest_path = archive_root / "ARCHIVE_MANIFEST.csv"
    if not manifest_path.is_file():
        raise RuntimeError("Archive manifest is required before registry finalization")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    archived_at = datetime.now(timezone.utc).isoformat()
    final_checkpoints = {item["checkpoint"] for item in config["final_models"]}
    keep_run_ids = {
        "20260914_214300_centralized_infrastructure_seed391",
        "20260915_001624_b6_mobilenet_v3_mel3_finetune_seed391_train",
        "20260915_002014_b6_resnet18_mel3_finetune_seed391_train",
    }
    registry_specs = (
        (ROOT / "logs/run_index.csv", lambda row: row.get("step") != "B6" or row.get("run_id") in keep_run_ids),
        (ROOT / "logs/experiment_registry.csv", lambda row: row.get("step") != "B6" or row.get("checkpoint_path") in final_checkpoints or "20260914_214300" in row.get("experiment_id", "")),
    )
    for source, keep in registry_specs:
        destination = archive_root / rel(source)
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite archived registry: {destination}")
        source_hash = sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"Archived registry hash verification failed: {source}")
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        retained = [row for row in rows if keep(row)]
        write_csv(source, retained, columns)
        manifest_rows.append({
            "original_path": rel(source),
            "archived_path": destination.relative_to(ROOT).as_posix(),
            "experiment_id": "",
            "model": "",
            "artifact_type": "registry_snapshot",
            "sha256": source_hash,
            "original_status": "HISTORICAL_MIXED",
            "archive_reason": "OBSOLETE_REPORT",
            "archived_at": archived_at,
        })
    manifest_columns = [
        "original_path", "archived_path", "experiment_id", "model",
        "artifact_type", "sha256", "original_status", "archive_reason", "archived_at",
    ]
    write_csv(manifest_path, manifest_rows, manifest_columns)
    archived_run_rows = [row for row in manifest_rows if row["experiment_id"] or row["artifact_type"] in ("log", "registry_snapshot")]
    write_csv(FINAL_DIR / "archived_b6_runs.csv", archived_run_rows, manifest_columns)
    validation_path = FINAL_DIR / "b6_final_lock_validation.json"
    validation = read_json(validation_path)
    validation["archived_files"] = len(manifest_rows)
    validation["active_run_index_b6_rows"] = sum(1 for row in csv.DictReader((ROOT / "logs/run_index.csv").open(encoding="utf-8")) if row.get("step") == "B6")
    validation["active_experiment_registry_b6_rows"] = sum(1 for row in csv.DictReader((ROOT / "logs/experiment_registry.csv").open(encoding="utf-8")) if row.get("step") == "B6")
    validation["registry_snapshots_archived"] = True
    write_json(validation_path, validation)


def verify_final_lock(config: dict[str, Any], archive_root: Path) -> None:
    final_rows, gates = load_final_rows(config)
    manifest_path = archive_root / "ARCHIVE_MANIFEST.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    hash_failures: list[str] = []
    source_failures: list[str] = []
    removed_sources = 0
    registry_snapshots = 0
    for row in manifest_rows:
        destination = ROOT / row["archived_path"]
        if not destination.is_file() or sha256(destination) != row["sha256"]:
            hash_failures.append(row["archived_path"])
        source = ROOT / row["original_path"]
        if row["artifact_type"] == "registry_snapshot":
            registry_snapshots += 1
        elif source.exists():
            source_failures.append(row["original_path"])
        else:
            removed_sources += 1
    remaining = [rel(path) for path in archive_candidates(config)]
    figure_path = ROOT / "reports/figures/b6/final_centralized_b6_ranking.png"
    if not figure_path.is_file():
        raise RuntimeError(f"Final report figure is missing: {rel(figure_path)}")
    if hash_failures or source_failures or remaining:
        raise RuntimeError(
            f"Final archive verification failed: hash={hash_failures}, source={source_failures}, remaining={remaining}"
        )
    validation_path = FINAL_DIR / "b6_final_lock_validation.json"
    validation = read_json(validation_path)
    validation.update({
        "status": "PASS",
        "gates": gates,
        "archive_hashes_verified": len(manifest_rows),
        "archived_sources_removed_after_copy_verification": removed_sources,
        "registry_snapshots_verified": registry_snapshots,
        "active_final_source_evidence_files": 34,
        "final_lock_output_files": 5,
        "total_active_final_files": 39,
        "final_table_figure": rel(figure_path),
        "final_table_figure_sha256": sha256(figure_path),
        "unarchived_failures": [],
    })
    validation.pop("active_final_evidence_files", None)
    write_json(validation_path, validation)
    print(json.dumps({
        "status": "PASS",
        "final_active_models": len(final_rows),
        "archive_hashes_verified": len(manifest_rows),
        "archived_sources_removed": removed_sources,
        "registry_snapshots_verified": registry_snapshots,
        "remaining_superseded_active_files": 0,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Copy, verify, and remove superseded active files")
    parser.add_argument("--finalize-registries", action="store_true", help="Archive mixed registries and retain only active B6 rows")
    parser.add_argument("--verify", action="store_true", help="Re-hash the archive and validate the locked active namespace")
    args = parser.parse_args()
    config = read_json(LOCK_CONFIG)
    archive_root = ROOT / "archive/b6_superseded" / config["lock_version"]
    if args.verify:
        verify_final_lock(config, archive_root)
        return
    if args.finalize_registries:
        finalize_registries(config, archive_root)
        print(json.dumps({"status": "PASS", "registry_snapshots_archived": True}, indent=2))
        return
    if args.apply and archive_root.exists():
        raise RuntimeError(f"Archive lock version already exists; refusing overwrite: {archive_root}")
    final_rows, gates = load_final_rows(config)
    candidates = archive_candidates(config)
    archive_rows = archive_files(candidates, archive_root, args.apply)
    preview = {
        "mode": "APPLY" if args.apply else "DRY_RUN",
        "lock_version": config["lock_version"],
        "final_active_models": len(final_rows),
        "archive_file_count": len(archive_rows),
        "archive_size_bytes": sum((ROOT / row["original_path"]).stat().st_size for row in archive_rows) if not args.apply else sum((ROOT / row["archived_path"]).stat().st_size for row in archive_rows),
        "archive_root": archive_root.relative_to(ROOT).as_posix(),
    }
    if not args.apply:
        print(json.dumps(preview, indent=2))
        for row in archive_rows:
            print(f"{row['archive_reason']}: {row['original_path']}")
        return

    manifest_columns = [
        "original_path", "archived_path", "experiment_id", "model",
        "artifact_type", "sha256", "original_status", "archive_reason", "archived_at",
    ]
    write_csv(archive_root / "ARCHIVE_MANIFEST.csv", archive_rows, manifest_columns)
    remove_empty_directories()
    write_csv(FINAL_DIR / "centralized_b6_final_locked.csv", final_rows, TABLE_COLUMNS)
    archived_run_rows = [row for row in archive_rows if row["experiment_id"] or row["artifact_type"] == "log"]
    write_csv(FINAL_DIR / "archived_b6_runs.csv", archived_run_rows, manifest_columns)
    ranking = sorted(final_rows, key=lambda row: float(row["overall_macro_f1"]), reverse=True)
    runner_sources = list((ROOT / "src/fedecai/models").glob("run_b6*.py"))
    archive_scan_matches = [
        rel(path) for path in runner_sources
        if any(token in path.read_text(encoding="utf-8").lower() for token in ('root / "archive', "root / 'archive", 'glob("archive', "glob('archive"))
    ]
    smoke_guard_sources = {
        name: (ROOT / "src/fedecai/models" / name).read_text(encoding="utf-8")
        for name in ("run_b6_wav2vec2_finetune.py", "run_b6_pitch_fusion_finetune.py")
    }
    smoke_guard_pass = all(
        "if not args.smoke_test:\n                    checkpoint.parent.mkdir" in source
        for source in smoke_guard_sources.values()
    )
    if archive_scan_matches or not smoke_guard_pass:
        raise RuntimeError(f"Final source gates failed: archive_scans={archive_scan_matches}, smoke_guard={smoke_guard_pass}")
    remaining_superseded = archive_candidates(config)
    if remaining_superseded:
        raise RuntimeError(f"Superseded files remain active: {[rel(path) for path in remaining_superseded]}")
    validation = {
        "schema_version": "b6.final_lock.validation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "lock_version": config["lock_version"],
        "final_active_rows": len(final_rows),
        "active_final_evidence_files": len(final_rows) * 4 + 2,
        "archived_files": len(archive_rows),
        "archive_manifest": rel(archive_root / "ARCHIVE_MANIFEST.csv"),
        "gates": gates,
        "no_final_path_points_to_archive": all("archive/" not in str(value).lower() for row in final_rows for value in row.values()),
        "active_runner_archive_scan_matches": len(archive_scan_matches),
        "active_runner_archive_scan_files": archive_scan_matches,
        "smoke_checkpoint_overwrite_guard": smoke_guard_pass,
        "test_samples_loaded": 0,
        "centralized_retraining_triggered": False,
        "federated_training_triggered": False,
        "federated_backbone": config["federated_backbone"],
        "efficiency_fallback": config["efficiency_fallback"],
        "unarchived_failures": [],
    }
    write_json(FINAL_DIR / "b6_final_lock_validation.json", validation)
    report_lines = [
        "# Final locked B6 centralized benchmark",
        "",
        "All values are validation-only saved evidence. The fixed 665-sample test set remains untouched.",
        "",
        "| Rank | Model | Experiment ID | Macro-F1 | UAR | Worst-accent F1 | Latency median ms | Size MB |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranking, 1):
        report_lines.append(
            f"| {index} | {row['model']} | `{row['experiment_id']}` | {float(row['overall_macro_f1']):.6f} | "
            f"{float(row['uar']):.6f} | {float(row['worst_accent_macro_f1']):.6f} | "
            f"{float(row['latency_median_ms']):.3f} | {float(row['model_size_mb']):.3f} |"
        )
    report_lines.extend([
        "", "## Human decision", "",
        "- `FEDERATED_BACKBONE = resnet18_mel3`",
        "- `BACKBONE_TRAINING_POLICY = full_finetune`",
        "- `EFFICIENCY_FALLBACK = mobilenet_v3_mel3`",
        "- ResNet18 is the strongest finalized neural model, retains measurable accent disparity, "
        "and is materially smaller/faster than Pitch-Fusion while remaining practical for simulated FL.",
        "", "## Lock status", "",
        "```text",
        "B6_STATUS = LOCKED",
        "B6_ARCHIVE_STATUS = COMPLETE",
        "CENTRALIZED_HYPERPARAMETER_SEARCH = CLOSED",
        "CENTRALIZED_RETRAINING = DISABLED_BY_HUMAN_DECISION",
        "FEDERATED_BACKBONE_STATUS = SELECTED_RESNET18",
        "EFFICIENCY_FALLBACK_STATUS = SELECTED_MOBILENETV3",
        "TEST_SET_STATUS = UNTOUCHED",
        "FEDERATED_STAGE = READY_TO_START",
        "FEDECAI_STATUS = NOT_STARTED",
        "```",
    ])
    (FINAL_DIR / "FINAL_B6_LOCK_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(preview | {"status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
