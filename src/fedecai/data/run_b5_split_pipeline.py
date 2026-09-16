"""Run the active B5 emotion-stratified split and federated-client builders."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=391)
    parser.add_argument("--main-clients", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_command = [
        sys.executable,
        str(ROOT / "src/fedecai/data/create_emotion_stratified_split.py"),
        "--seed", str(args.seed),
    ]
    client_command = [
        sys.executable,
        str(ROOT / "src/fedecai/data/build_b5_federated_scenarios.py"),
        "--main-clients", str(args.main_clients),
    ]
    subprocess.run(split_command, cwd=ROOT, check=True)
    subprocess.run(client_command, cwd=ROOT, check=True)
    print(json.dumps({
        "status": "PASS",
        "split_version": "emotion_stratified_v1",
        "split_manifest": "data/manifests/emotion_stratified_v1.csv",
        "federated_assignments": "data/manifests/federated_client_assignments_emotion_stratified_v1.csv",
        "speaker_overlap_interpretation": "INFO_ONLY",
        "duplicate_waveform_cross_split_policy": "HARD_FAILURE",
    }, indent=2))


if __name__ == "__main__":
    main()
