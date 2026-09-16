"""Persisted smoke/train orchestration for the two authorized B6 controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fedecai.models.b6_common import enforce_b6_training_lock, load_config, sha256_file, write_json_atomic
from fedecai.models.run_b6_performance_all import run_one


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/b6_regression_controls_v1.json"
MODELS = ("wav2vec2_raw", "pitch_fusion_paper_aligned")


def command(model: str, smoke: bool) -> list[str]:
    script = (
        "src/fedecai/models/run_b6_wav2vec2_finetune.py"
        if model == "wav2vec2_raw"
        else "src/fedecai/models/run_b6_pitch_fusion_finetune.py"
    )
    result = [sys.executable, str(ROOT / script), "--config", str(CONFIG), "--device", "cuda"]
    if smoke:
        result.extend(("--smoke-test", "--max-audio-samples", "160000"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    enforce_b6_training_lock(ROOT, "controlled B6 retraining/smoke")
    if "DAP301m" not in str(Path(sys.executable).resolve()):
        raise RuntimeError(f"DAP301m required: {sys.executable}")
    config = load_config(CONFIG)
    experiment_id = config[args.model]["experiment_id"]
    checkpoint = ROOT / "artifacts/checkpoints" / f"{experiment_id}.pt"
    before_exists = checkpoint.is_file()
    before_hash = sha256_file(checkpoint) if before_exists else None
    result = run_one(command(args.model, args.phase == "smoke"), args.model, experiment_id, args.phase, CONFIG)
    after_exists = checkpoint.is_file()
    after_hash = sha256_file(checkpoint) if after_exists else None
    payload = {
        "phase": args.phase, "model": args.model, "experiment_id": experiment_id,
        "result": result, "checkpoint_before_exists": before_exists,
        "checkpoint_before_sha256": before_hash, "checkpoint_after_exists": after_exists,
        "checkpoint_after_sha256": after_hash,
        "smoke_checkpoint_unchanged": args.phase != "smoke" or (before_exists == after_exists and before_hash == after_hash),
        "test_samples_loaded": 0,
    }
    audit_path = ROOT / "artifacts/audit" / f"b6_regression_control_{args.model}_{args.phase}.json"
    write_json_atomic(payload, audit_path)
    if result["exit_code"]:
        raise RuntimeError(f"{args.phase} failed for {args.model}")
    if args.phase == "smoke" and not payload["smoke_checkpoint_unchanged"]:
        raise RuntimeError(f"Smoke modified checkpoint for {experiment_id}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
