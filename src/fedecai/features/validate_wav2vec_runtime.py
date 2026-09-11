"""Validate the local runtime before downloading the pinned Wav2Vec2 weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Mapping


REQUIRED_DISTRIBUTIONS = {
    "torch": "2.14.0",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
}


def evaluate_versions(installed: Mapping[str, str]) -> dict[str, object]:
    missing = sorted(set(REQUIRED_DISTRIBUTIONS) - set(installed))
    mismatched = {
        name: {"expected": expected, "installed": installed[name]}
        for name, expected in REQUIRED_DISTRIBUTIONS.items()
        if name in installed and installed[name] != expected
    }
    return {
        "status": "PASS" if not missing and not mismatched else "BLOCKED",
        "missing": missing,
        "mismatched": mismatched,
    }


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def validate_runtime() -> dict[str, object]:
    installed = installed_versions()
    result = evaluate_versions(installed)
    import_check: dict[str, object] = {
        "attempted": result["status"] == "PASS",
        "wav2vec2_model_importable": False,
        "torch_cuda_available": None,
        "torch_cuda_device_count": None,
        "error": None,
    }
    if result["status"] == "PASS":
        try:
            import torch
            from transformers import Wav2Vec2Model  # noqa: F401

            import_check["wav2vec2_model_importable"] = True
            import_check["torch_cuda_available"] = torch.cuda.is_available()
            import_check["torch_cuda_device_count"] = torch.cuda.device_count()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            import_check["error"] = f"{type(exc).__name__}: {exc}"
            result["status"] = "BLOCKED"

    return {
        "status": result["status"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "required_distributions": REQUIRED_DISTRIBUTIONS,
        "installed_distributions": installed,
        "missing_distributions": result["missing"],
        "mismatched_distributions": result["mismatched"],
        "import_check": import_check,
        "model_weights_loaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with status 1 when the runtime is not ready.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optionally write the same JSON result to a reproducibility report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_runtime()
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_pass and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
