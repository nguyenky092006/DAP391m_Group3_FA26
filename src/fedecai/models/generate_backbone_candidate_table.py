"""Generate a no-winner ResNet-18 vs MobileNetV3 evidence table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SPLIT = "emotion_stratified_v1"


def read_metrics(path: Path, expected_model: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate metrics: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model") != expected_model:
        raise ValueError(f"Expected model={expected_model} in {path}")
    if value.get("split_version") != EXPECTED_SPLIT:
        raise ValueError(f"Candidate evidence must use {EXPECTED_SPLIT}: {path}")
    return value


def display(value: Any) -> Any:
    return "N/A" if value is None else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet-metrics", type=Path, required=True)
    parser.add_argument("--mobilenet-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tables/b7")
    args = parser.parse_args()

    resnet_path = args.resnet_metrics.resolve()
    mobile_path = args.mobilenet_metrics.resolve()
    resnet = read_metrics(resnet_path, "resnet18_mel3")
    mobile = read_metrics(mobile_path, "mobilenet_v3_mel3")
    fields = [
        ("Overall Macro-F1", "↑", "overall_macro_f1"),
        ("Worst-accent Macro-F1", "↑", "worst_accent_macro_f1"),
        ("Accent gap", "↓", "accent_gap"),
        ("Latency (ms)", "↓", "inference_latency_ms"),
        ("Model size (MB)", "↓", "model_size_mb"),
        ("Params", "↓", "total_parameters"),
        ("Training time (sec)", "↓", "training_time_sec"),
    ]
    rows = [
        {
            "metric": label,
            "direction": direction,
            "resnet18": display(resnet.get(key)),
            "mobilenetv3": display(mobile.get(key)),
            "source_data": f"{resnet_path.as_posix()} | {mobile_path.as_posix()}",
        }
        for label, direction, key in fields
    ]
    rows.extend([
        {"metric": "Training stability", "direction": "context", "resnet18": "N/A", "mobilenetv3": "N/A", "source_data": "Requires recorded multi-seed/run evidence"},
        {"metric": "FL practicality notes", "direction": "context", "resnet18": "Pending human assessment", "mobilenetv3": "Pending human assessment", "source_data": "No federated run authorized before selection"},
    ])

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "resnet18_vs_mobilenetv3_backbone_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output / "resnet18_vs_mobilenetv3_backbone_candidates.md"
    lines = [
        "# ResNet-18 vs MobileNetV3 Backbone Candidate Table",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Split: `{EXPECTED_SPLIT}`",
        "",
        "| Metric | Direction | ResNet-18 | MobileNetV3 |",
        "| --- | :---: | ---: | ---: |",
        *[f"| {row['metric']} | {row['direction']} | {row['resnet18']} | {row['mobilenetv3']} |" for row in rows],
        "",
        "No winner is assigned automatically.",
        "",
        "`FEDECAI_BACKBONE_STATUS = AWAITING_HUMAN_DECISION`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "csv": str(csv_path),
        "markdown": str(md_path),
        "fedecai_backbone_status": "AWAITING_HUMAN_DECISION",
        "winner_selected": False,
    }, indent=2))


if __name__ == "__main__":
    main()
