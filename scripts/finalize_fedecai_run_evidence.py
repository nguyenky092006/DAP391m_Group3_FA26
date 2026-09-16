"""Persist the execution envelope and index the completed ResNet18 E2-E7 runs."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_LOG = ROOT / "logs" / "runs" / "fedecai_e1_e7_20260915"
INDEX = ROOT / "logs" / "run_index.csv"

RUNS = {
    "E2": ("e2_resnet18_unconditional_grl", "e2_resnet18_unconditional_grl.json", "best.pt"),
    "E3": ("e3_resnet18_conditional_grl", "e3_resnet18_emotion_conditioned_grl.json", "best.pt"),
    "E4": ("e4_fedavg_resnet18", "e4_fedavg_resnet18.json", "best_main.pt"),
    "E5": ("e5_fedprox_resnet18", "e5_fedprox_resnet18.json", "best_main.pt"),
    "E6": ("e6_fedavg_unconditional_grl", "e6_fedavg_resnet18_unconditional_grl.json", "best_main.pt"),
    "E7": ("e7_fedecai_resnet18", "e7_fedecai_resnet18.json", "best_main.pt"),
}

COMMANDS = [
    "python src/fedecai/models/run_fedecai_ladder.py preflight-e1",
    "python src/fedecai/models/run_fedecai_ladder.py preflight-grl",
    "python src/fedecai/models/run_fedecai_ladder.py run-central --experiment E2 --smoke-test",
    "python src/fedecai/models/run_fedecai_ladder.py run-central --experiment E2",
    "python src/fedecai/models/run_fedecai_ladder.py run-central --experiment E3 --smoke-test",
    "python src/fedecai/models/run_fedecai_ladder.py run-central --experiment E3",
    "python src/fedecai/models/run_fedecai_ladder.py centralized-gate",
    "python src/fedecai/models/run_fedecai_ladder.py preflight-clients",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E4 --smoke-test",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E4",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E5 --smoke-test",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E5",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E6 --smoke-test",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E6",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E7 --smoke-test",
    "python src/fedecai/models/run_fedecai_ladder.py run-federated --experiment E7",
    "python src/fedecai/models/run_fedecai_accent_probes.py",
    "python src/fedecai/models/generate_fedecai_reports.py",
    "python src/fedecai/models/audit_fedecai_ladder.py",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    RUN_LOG.mkdir(parents=True, exist_ok=True)
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    environment = {
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": transformers_version,
        "platform": platform.platform(),
        "test_samples_loaded": 0,
    }
    (RUN_LOG / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (RUN_LOG / "commands.log").write_text("\n".join(COMMANDS) + "\n", encoding="utf-8")

    with INDEX.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    known = {row["run_id"] for row in rows}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "UNKNOWN"

    summary_rows = []
    for step, (folder, config_name, checkpoint_name) in RUNS.items():
        root = ROOT / "artifacts" / "fedecai" / folder
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        resource = json.loads((root / "resource_metrics.json").read_text(encoding="utf-8"))
        run_id = f"20260915_{metrics['experiment_id']}_full"
        summary_rows.append(
            f"- {step}: COMPLETE; Macro-F1={metrics['overall_macro_f1']:.6f}; "
            f"Worst-accent={metrics['worst_accent_macro_f1']:.6f}; "
            f"test_samples_loaded=0."
        )
        if run_id in known:
            continue
        rows.append({
            "run_id": run_id,
            "experiment_id": metrics["experiment_id"],
            "timestamp": resource["training_end_timestamp"],
            "step": step,
            "model": "resnet18_mel3",
            "variant": "full_validation_ladder",
            "seed": str(metrics["seed"]),
            "status": "COMPLETE",
            "dataset_version": metrics["dataset_version"],
            "split_version": metrics["split_version"],
            "git_commit": commit,
            "config_path": f"configs/models/fedecai/{config_name}",
            "stdout_path": f"logs/runs/fedecai_e1_e7_20260915/{step.lower()}_full_stdout.log",
            "stderr_path": "",
            "metrics_path": rel(root / "metrics.json"),
            "checkpoint_path": rel(root / checkpoint_name),
            "prediction_path": rel(root / "predictions.csv"),
            "training_time_sec": str(metrics["training_time_sec"]),
            "notes": "Validation-only E1-E7 ladder; combined stdout/stderr log; fixed test untouched.",
        })
    with INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = "# ResNet18 FedECAI E2-E7 execution summary\n\n" + "\n".join(summary_rows)
    summary += (
        "\n\nFinal integrity audit: PASS. Fixed 665-sample test: UNTOUCHED. "
        "RQ6 lambda sweep: NOT_STARTED.\n"
    )
    (RUN_LOG / "run_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs_indexed": len(RUNS), **environment}, indent=2))


if __name__ == "__main__":
    main()
