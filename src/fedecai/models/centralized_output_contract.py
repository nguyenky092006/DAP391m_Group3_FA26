"""Canonical artifact and run-evidence paths for centralized experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CentralizedRunPaths:
    root: Path
    experiment_id: str
    run_id: str

    @property
    def prediction(self) -> Path:
        return self.root / "artifacts/predictions" / f"{self.experiment_id}.csv"

    @property
    def checkpoint(self) -> Path:
        return self.root / "artifacts/checkpoints" / f"{self.experiment_id}.artifact"

    @property
    def metrics(self) -> Path:
        return self.root / "artifacts/metrics" / f"{self.experiment_id}.json"

    @property
    def run_directory(self) -> Path:
        return self.root / "logs/runs" / self.run_id

    def prepare(self) -> None:
        for directory in (
            self.prediction.parent, self.checkpoint.parent, self.metrics.parent,
            self.run_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def required_run_evidence(self) -> dict[str, Path]:
        return {
            "run_summary": self.run_directory / "run_summary.md",
            "commands": self.run_directory / "commands.log",
            "stdout": self.run_directory / "stdout.log",
            "stderr": self.run_directory / "stderr.log",
            "environment": self.run_directory / "environment.json",
        }
