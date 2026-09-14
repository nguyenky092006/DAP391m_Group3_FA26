"""Persist one complete DAP301m infrastructure-validation run."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def package_record(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", importlib.metadata.version(name))
        return {"installed": True, "version": str(version), "import_error": None}
    except Exception as error:
        try:
            version = importlib.metadata.version(name)
        except Exception:
            version = None
        return {
            "installed": version is not None, "version": version,
            "import_error": repr(error),
        }


def environment() -> dict[str, Any]:
    pyvenv = ROOT / "DAP301m/pyvenv.cfg"
    pyvenv_text = pyvenv.read_text(encoding="utf-8") if pyvenv.is_file() else ""
    result = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "environment_name": "DAP301m",
        "uv_managed": "uv =" in pyvenv_text,
        "uv_version": next(
            (line.split("=", 1)[1].strip() for line in pyvenv_text.splitlines() if line.startswith("uv =")),
            None,
        ),
        "packages": {
            name: package_record(name)
            for name in ("torch", "torchvision", "torchaudio", "transformers")
        },
    }
    try:
        import torch
        result.update({
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception:
        result.update({"cuda_available": None, "torch_cuda_version": None, "gpu_name": None})
    return result


def append_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    rows.append({name: row.get(name, "") for name in fields})
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    if "DAP301m" not in str(Path(sys.executable).resolve()):
        raise RuntimeError(f"Validation must run in DAP301m: {sys.executable}")
    now = datetime.now().astimezone()
    run_id = now.strftime("%Y%m%d_%H%M%S_centralized_infrastructure_seed391")
    run_dir = ROOT / "logs/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    env = environment()
    (run_dir / "environment.json").write_text(
        json.dumps(env, indent=2) + "\n", encoding="utf-8"
    )
    commands = [
        [sys.executable, "-m", "py_compile",
         "src/fedecai/models/b6_common.py",
         "src/fedecai/models/benchmark_costs.py",
         "src/fedecai/models/frozen_embedding_cache.py",
         "src/fedecai/models/frozen_head_training.py",
         "src/fedecai/models/centralized_evaluator.py",
         "src/fedecai/models/centralized_output_contract.py",
         "src/fedecai/models/generate_centralized_report.py",
         "src/fedecai/models/run_b6_all_models.py",
         "src/fedecai/models/run_b6_classical_pipeline.py",
         "src/fedecai/models/run_b6_deep_pipeline.py",
         "src/fedecai/models/run_b6_pitch_fusion_paper.py",
         "src/fedecai/models/run_b6_vit_cached.py",
         "src/fedecai/models/run_b6_wav2vec2_raw.py",
         "src/fedecai/models/run_centralized_smoke_tests.py",
         "src/fedecai/models/run_centralized_benchmark_preflight.py"],
        [sys.executable, "-m", "unittest",
         "tests.test_b6_common", "tests.test_b6_eight_model_contract",
         "tests.test_centralized_infrastructure", "-v"],
        [sys.executable, "src/fedecai/models/run_centralized_smoke_tests.py",
         "--device", "cuda"],
        [sys.executable, "src/fedecai/models/run_centralized_benchmark_preflight.py",
         "--output", "artifacts/audit/benchmark_preflight.json"],
    ]
    (run_dir / "commands.log").write_text(
        "\n".join(subprocess.list2cmdline(command) for command in commands) + "\n",
        encoding="utf-8",
    )
    stdout_parts, stderr_parts, failures = [], [], []
    started = time.perf_counter()
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = str(ROOT / "src")
    for command in commands:
        label = subprocess.list2cmdline(command)
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            env=child_environment,
        )
        stdout_parts.append(f"$ {label}\n{completed.stdout}")
        stderr_parts.append(f"$ {label}\n{completed.stderr}")
        if completed.returncode:
            failures.append({"command": label, "exit_code": completed.returncode})
    elapsed = time.perf_counter() - started
    (run_dir / "stdout.log").write_text("\n".join(stdout_parts), encoding="utf-8")
    (run_dir / "stderr.log").write_text("\n".join(stderr_parts), encoding="utf-8")
    smoke_path = ROOT / "artifacts/audit/centralized_smoke_tests.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
    preflight_path = ROOT / "artifacts/audit/benchmark_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    ready = not failures and preflight.get("centralized_infrastructure_status") == "READY"
    status = "READY" if ready else "BLOCKED"
    metrics = {
        "experiment_id": "centralized_infrastructure_validation",
        "model": None, "seed": 391,
        "dataset_version": "visec_clean_eligible_v1",
        "split_version": "speaker_disjoint_v2",
        "evidence_class": "SMOKE_TEST_ONLY",
        "centralized_infrastructure_status": status,
        "smoke_status": smoke.get("status"),
        "overall_macro_f1": None, "uar": None, "weighted_f1": None,
        "accuracy": None, "inference_latency_ms": None,
        "model_size_mb": None, "training_time_sec": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    failure_text = "# Failures\n\nNone.\n" if not failures else (
        "# Failures\n\n" + "\n".join(
            f"- `{item['command']}` exited {item['exit_code']}. See stdout/stderr logs."
            for item in failures
        ) + "\n"
    )
    (run_dir / "failures.md").write_text(failure_text, encoding="utf-8")
    summary = f"""# Run Summary

Run ID: `{run_id}`  
Experiment ID: `centralized_infrastructure_validation`  
Evidence class: `SMOKE_TEST_ONLY`  
Status: `{status}`  
Python executable: `{sys.executable}`  
Start: `{now.isoformat()}`  
Elapsed seconds: `{elapsed:.3f}`  

This run compiled and unit-tested the centralized infrastructure, executed one
small implementation smoke path for each of the eight required models, and
reran the hard-gate preflight. It did not perform benchmark training or accept
smoke metrics as research results.
"""
    (run_dir / "run_summary.md").write_text(summary, encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    append_csv(ROOT / "logs/run_index.csv", {
        "run_id": run_id, "experiment_id": "centralized_infrastructure_validation",
        "timestamp": now.isoformat(), "step": "B6", "variant": "SMOKE_TEST_ONLY",
        "seed": 391, "status": status, "dataset_version": "visec_clean_eligible_v1",
        "split_version": "speaker_disjoint_v2", "git_commit": commit,
        "stdout_path": str((run_dir / "stdout.log").relative_to(ROOT)).replace("\\", "/"),
        "stderr_path": str((run_dir / "stderr.log").relative_to(ROOT)).replace("\\", "/"),
        "metrics_path": str((run_dir / "metrics.json").relative_to(ROOT)).replace("\\", "/"),
        "training_time_sec": elapsed,
        "notes": "Infrastructure validation only; no benchmark result accepted.",
    })
    append_csv(ROOT / "logs/experiment_registry.csv", {
        "experiment_id": f"centralized_infrastructure_validation_{run_id}",
        "timestamp": now.isoformat(), "step": "B6", "rq": "RQ3",
        "variant": "SMOKE_TEST_ONLY", "dataset_version": "visec_clean_eligible_v1",
        "split_version": "speaker_disjoint_v2", "seed": 391,
        "git_commit": commit, "hardware": env.get("gpu_name"), "status": status,
        "notes": "Eight-path smoke validation; no benchmark metrics.",
    })
    print(json.dumps({"run_id": run_id, "status": status, "run_dir": str(run_dir)}, indent=2))
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
