"""Read-only final integrity audit for the ResNet18 E1-E7 ladder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from fedecai.models.b6_common import sha256_file, write_json_atomic
from fedecai.models.run_fedecai_ladder import CONFIGS, E1_CHECKPOINT, ROOT, load_config


ROOTS = {
    "E2": ROOT / "artifacts/fedecai/e2_resnet18_unconditional_grl",
    "E3": ROOT / "artifacts/fedecai/e3_resnet18_conditional_grl",
    "E4": ROOT / "artifacts/fedecai/e4_fedavg_resnet18",
    "E5": ROOT / "artifacts/fedecai/e5_fedprox_resnet18",
    "E6": ROOT / "artifacts/fedecai/e6_fedavg_unconditional_grl",
    "E7": ROOT / "artifacts/fedecai/e7_fedecai_resnet18",
}
REQUIRED_REPORTS = [
    "rq4_e1_e2_e3.csv", "rq5_e4_e5_e6_e7.csv", "e1_e7_full_ladder.csv",
    "e1_e7_deltas.csv", "federated_resource_comparison.csv",
    "per_emotion_comparison.csv", "per_accent_comparison.csv",
    "accent_probe_comparison.csv", "FEDECAI_E1_E7_DECISION_PACKET.md",
]


def archive_valid() -> tuple[bool, int]:
    manifest = ROOT / "archive/spec_superseded/20260915_resnet18_e2_e7_v1/ARCHIVE_MANIFEST.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    valid = all(
        row["archive_reason"] == "SUPERSEDED_BY_RESNET18_E2_E7_IMPLEMENTATION"
        and sha256_file(ROOT / row["archived_path"]) == row["sha256"]
        for row in rows
    )
    return valid, len(rows)


def main() -> None:
    checks: dict[str, object] = {}
    e1_compat = json.loads((ROOT / "artifacts/fedecai/e1_checkpoint_compatibility.json").read_text(encoding="utf-8"))
    grl = json.loads((ROOT / "artifacts/fedecai/grl_test_report.json").read_text(encoding="utf-8"))
    clients = json.loads((ROOT / "artifacts/fedecai/federated_client_preflight.json").read_text(encoding="utf-8"))
    checks["resnet_feature_interface_status"] = e1_compat["resnet_feature_interface_status"]
    checks["e1_checkpoint_compatibility_status"] = e1_compat["e1_checkpoint_compatibility_status"]
    checks["grl_test_status"] = grl["status"]
    checks["federated_client_contract_status"] = clients["status"]
    checks["e1_checkpoint_sha256"] = sha256_file(E1_CHECKPOINT)
    reference_ids = set(pd.read_csv(
        ROOT / "artifacts/predictions/b6_resnet18_mel3_finetune_seed391_validation.csv",
        usecols=["sample_id"], dtype={"sample_id": "string"},
    )["sample_id"].astype(str))
    experiment_checks = {}
    test_loaded = int(e1_compat.get("test_samples_loaded", 0)) + int(grl.get("test_samples_loaded", 0)) + int(clients.get("test_samples_loaded", 0))
    for step, artifact_root in ROOTS.items():
        config_path, config = load_config(step)
        required = ["config.json", "run_report.json", "metrics.json", "predictions.csv", "resource_metrics.json", "audit.json", "checkpoint_manifest.json", "accent_probe.json"]
        required.append("training_history.csv" if step in {"E2", "E3"} else "round_history.csv")
        missing = [name for name in required if not (artifact_root / name).is_file()]
        report = json.loads((artifact_root / "run_report.json").read_text(encoding="utf-8"))
        audit = json.loads((artifact_root / "audit.json").read_text(encoding="utf-8"))
        checkpoint_manifest = json.loads((artifact_root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(artifact_root / "predictions.csv", dtype={"sample_id": "string"})
        checkpoint_hashes_valid = True
        if step in {"E2", "E3"}:
            checkpoint_hashes_valid = (
                sha256_file(artifact_root / checkpoint_manifest["best_checkpoint"]) == checkpoint_manifest["best_sha256"]
                and sha256_file(artifact_root / checkpoint_manifest["last_checkpoint"]) == checkpoint_manifest["last_sha256"]
            )
        else:
            checkpoint_hashes_valid = all(sha256_file(ROOT / row["path"]) == row["sha256"] for row in checkpoint_manifest["checkpoints"])
        valid = (
            not missing and config["backbone"] == "resnet18_mel3"
            and config["input"] == "3-channel Mel" and report["status"] == "COMPLETE"
            and audit["status"] == "PASS" and len(predictions) == 646
            and set(predictions["sample_id"].astype(str)) == reference_ids
            and checkpoint_hashes_valid
        )
        test_loaded += int(report.get("test_samples_loaded", 0)) + int(audit.get("test_samples_loaded", 0))
        experiment_checks[step] = {
            "status": "PASS" if valid else "FAIL", "missing": missing,
            "config": config_path.relative_to(ROOT).as_posix(),
            "prediction_rows": len(predictions), "same_validation_membership": set(predictions["sample_id"].astype(str)) == reference_ids,
            "checkpoint_hashes_valid": checkpoint_hashes_valid,
        }
    archive_status, archive_files = archive_valid()
    report_root = ROOT / "reports/tables/fedecai"
    missing_reports = [name for name in REQUIRED_REPORTS if not (report_root / name).is_file()]
    active_model_source = (ROOT / "src/fedecai/models/b6_deep_models.py").read_text(encoding="utf-8")
    active_runner_source = (ROOT / "src/fedecai/models/run_b6_fedecai_pipeline.py").read_text(encoding="utf-8")
    legacy_absent = all(token not in active_model_source + active_runner_source for token in ("AccentInvariantCNN", "one_hot", "NotImplementedError", "E1_cnn", "E4_fedavg_cnn"))
    passed = (
        all(value == "PASS" for key, value in checks.items() if key.endswith("status"))
        and all(value["status"] == "PASS" for value in experiment_checks.values())
        and test_loaded == 0 and archive_status and not missing_reports and legacy_absent
    )
    output = {
        "status": "PASS" if passed else "FAIL", **checks,
        "experiments": experiment_checks, "test_samples_loaded": test_loaded,
        "same_initial_e1_hash_e4_e7": len({
            json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))["initial_global_checkpoint_sha256"]
            for step, root in ROOTS.items() if step >= "E4"
        }) == 1,
        "archive_status": "PASS" if archive_status else "FAIL", "archived_files": archive_files,
        "active_legacy_path_absent": legacy_absent, "missing_reports": missing_reports,
        "rq6_lambda_sweep_status": "NOT_STARTED", "fixed_test_status": "UNTOUCHED",
    }
    write_json_atomic(output, ROOT / "artifacts/fedecai/final_integrity_audit.json")
    if not passed:
        raise RuntimeError("Final FedECAI ladder integrity audit failed")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
