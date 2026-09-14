"""Shared, machine-readable cost measurements for centralized B6 models."""

from __future__ import annotations

import platform
import statistics
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


MEBIBYTE = 1024 ** 2
PRIMARY_LATENCY_STATISTIC = "median"


def parameter_count(model: Any) -> dict[str, int]:
    """Return total and trainable parameter counts for a torch-style model."""
    parameters = list(model.parameters())
    return {
        "total_parameters": int(sum(value.numel() for value in parameters)),
        "trainable_parameters": int(
            sum(value.numel() for value in parameters if value.requires_grad)
        ),
    }


def classical_parameter_description(model: Any) -> dict[str, Any]:
    """Describe a classical model without inventing a neural parameter count."""
    representation: dict[str, Any] = {"estimator": type(model).__name__}
    if hasattr(model, "support_vectors_"):
        representation["support_vectors"] = int(len(model.support_vectors_))
    if hasattr(model, "n_estimators"):
        representation["trees"] = int(model.n_estimators)
    return {
        "total_parameters": None,
        "trainable_parameters": None,
        "model_specific_representation": representation,
    }


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(path)


def model_size(
    checkpoint: Path,
    *,
    externally_referenced_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    """Measure deployable bytes; external encoder bytes count toward deployment."""
    checkpoint = checkpoint.resolve()
    external = [Path(item).resolve() for item in externally_referenced_artifacts]
    checkpoint_bytes = _path_bytes(checkpoint)
    external_bytes = sum(_path_bytes(item) for item in external)
    effective_bytes = checkpoint_bytes + external_bytes
    return {
        "model_size_definition": "serialized deployable artifacts; bytes / 1024^2",
        "checkpoint_artifact": str(checkpoint),
        "checkpoint_size_mb": checkpoint_bytes / MEBIBYTE,
        "external_artifacts": [str(item) for item in external],
        "external_artifacts_size_mb": external_bytes / MEBIBYTE,
        "effective_deployment_size_mb": effective_bytes / MEBIBYTE,
        "model_size_mb": effective_bytes / MEBIBYTE,
    }


def hardware_description(device: Any) -> dict[str, Any]:
    import torch

    device_value = torch.device(device)
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "device": str(device_value),
        "gpu_name": (
            torch.cuda.get_device_name(device_value)
            if device_value.type == "cuda"
            else None
        ),
    }


def benchmark_inference_latency(
    inference: Callable[[], Any],
    *,
    device: Any,
    batch_size: int = 1,
    precision: str = "float32",
    warmup_runs: int = 10,
    timed_runs: int = 50,
    preprocessing_included: bool = False,
) -> dict[str, Any]:
    """Measure the actual supplied inference path using median as RQ3 value."""
    if warmup_runs < 1 or timed_runs < 2:
        raise ValueError("Latency requires >=1 warm-up and >=2 timed runs")
    import numpy as np
    import torch

    torch_device = torch.device(device)

    def synchronize() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            inference()
        synchronize()
        samples = []
        for _ in range(timed_runs):
            synchronize()
            started = time.perf_counter_ns()
            inference()
            synchronize()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)

    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return {
        **hardware_description(torch_device),
        "batch_size": int(batch_size),
        "precision": precision,
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "preprocessing_included": bool(preprocessing_included),
        "primary_latency_statistic": PRIMARY_LATENCY_STATISTIC,
        "inference_latency_ms": median,
        "latency_mean_ms": float(statistics.fmean(samples)),
        "latency_median_ms": median,
        "latency_std_ms": float(np.std(values, ddof=0)),
        "latency_p95_ms": float(np.percentile(values, 95)),
    }


class TrainingTimer(AbstractContextManager["TrainingTimer"]):
    """Wall-clock timer excluding time outside its explicit context."""

    def __enter__(self) -> "TrainingTimer":
        self.training_start_timestamp = datetime.now(timezone.utc).isoformat()
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.training_time_sec = time.perf_counter() - self._started
        self.training_end_timestamp = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> dict[str, Any]:
        return {
            "training_start_timestamp": self.training_start_timestamp,
            "training_end_timestamp": self.training_end_timestamp,
            "training_time_sec": float(self.training_time_sec),
        }
