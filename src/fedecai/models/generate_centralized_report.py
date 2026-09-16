"""Generate report-ready tables/figures only from eight saved metrics JSON files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REQUIRED_MODELS = {
    "svm_rbf", "xgboost", "cnn2d_logmel", "resnet18_mel3",
    "mobilenet_v3_mel3", "vit_b16_mel3", "wav2vec2_raw",
    "pitch_fusion_paper_aligned",
}
DISPLAY = {
    "svm_rbf": "SVM", "xgboost": "XGBoost", "cnn2d_logmel": "CNN2D",
    "resnet18_mel3": "ResNet-18", "mobilenet_v3_mel3": "MobileNetV3",
    "vit_b16_mel3": "ViT-B/16", "wav2vec2_raw": "Wav2Vec2",
    "pitch_fusion_paper_aligned": "Pitch-Fusion",
}


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[[
        "overall_macro_f1", "worst_accent_macro_f1",
        "inference_latency_ms", "model_size_mb",
    ]].to_numpy(float)
    transformed = values * np.array([-1.0, -1.0, 1.0, 1.0])
    keep = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(transformed):
        dominates = np.all(transformed <= point, axis=1) & np.any(transformed < point, axis=1)
        dominates[index] = False
        keep[index] = not dominates.any()
    return keep


def load_metrics(paths: list[Path]) -> pd.DataFrame:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    models = {row.get("model") for row in rows}
    if len(rows) != 8 or models != REQUIRED_MODELS:
        raise ValueError(f"Report requires exact 8-model metrics; got {sorted(models)}")
    required_values = [
        "overall_macro_f1", "uar", "north_macro_f1", "central_macro_f1",
        "south_macro_f1", "worst_accent_macro_f1", "accent_gap",
        "inference_latency_ms", "model_size_mb", "training_time_sec",
    ]
    missing = {
        row["model"]: [name for name in required_values if row.get(name) is None]
        for row in rows
    }
    missing = {model: names for model, names in missing.items() if names}
    if missing:
        raise ValueError(f"Refusing report generation with missing measured values: {missing}")
    frame = pd.DataFrame(rows)
    frame["pareto_front"] = pareto_mask(frame)
    frame["display_model"] = frame["model"].map(DISPLAY)
    return frame


def save_figure(figure, output: Path) -> None:
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs=8, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/centralized_benchmark")
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    frame = load_metrics([path.resolve() for path in args.metrics])
    table = pd.DataFrame({
        "Model": frame["display_model"], "Macro-F1 ↑": frame["overall_macro_f1"],
        "UAR ↑": frame["uar"], "North F1 ↑": frame["north_macro_f1"],
        "Central F1 ↑": frame["central_macro_f1"], "South F1 ↑": frame["south_macro_f1"],
        "Worst-accent F1 ↑": frame["worst_accent_macro_f1"],
        "Accent Gap ↓": frame["accent_gap"], "Latency ms ↓": frame["inference_latency_ms"],
        "Model Size MB ↓": frame["model_size_mb"],
        "Params": frame.apply(lambda row: "N/A" if pd.isna(row.get("total_parameters")) else f"{int(row['total_parameters']):,}", axis=1),
        "Train Time": frame["training_time_sec"],
    })
    table_path = output / "centralized_benchmark_table.csv"
    table.to_csv(table_path, index=False, float_format="%.4f")
    figure, axis = plt.subplots(figsize=(16, 4.8)); axis.axis("off")
    shown = table.copy()
    for column in shown.columns[1:]:
        if column != "Params": shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    artist = axis.table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="center")
    artist.auto_set_font_size(False); artist.set_fontsize(7); artist.scale(1, 1.35)
    lower_is_better = {"Accent Gap ↓", "Latency ms ↓", "Model Size MB ↓", "Train Time"}
    for column_index, column in enumerate(table.columns[1:], start=1):
        if column == "Params":
            continue
        numeric = pd.to_numeric(table[column], errors="coerce")
        best = numeric.min() if column in lower_is_better else numeric.max()
        for row_index, value in enumerate(numeric, start=1):
            if np.isfinite(value) and np.isclose(value, best):
                artist[(row_index, column_index)].set_facecolor("#d9ead3")
    save_figure(figure, output / "centralized_benchmark_table")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    pareto = frame["pareto_front"]
    axes[0].scatter(frame["inference_latency_ms"], frame["overall_macro_f1"], s=np.maximum(frame["model_size_mb"], 1) * 2, c=np.where(pareto, "#d62728", "#4c78a8"), alpha=.8)
    axes[1].scatter(frame["model_size_mb"], frame["worst_accent_macro_f1"], c=np.where(pareto, "#d62728", "#4c78a8"), alpha=.8)
    for _, row in frame.iterrows():
        axes[0].annotate(row["display_model"], (row["inference_latency_ms"], row["overall_macro_f1"]), fontsize=8)
        axes[1].annotate(row["display_model"], (row["model_size_mb"], row["worst_accent_macro_f1"]), fontsize=8)
    axes[0].set(xlabel="Inference latency (ms) ↓", ylabel="Macro-F1 ↑")
    axes[1].set(xlabel="Model size (MB) ↓", ylabel="Worst-accent Macro-F1 ↑")
    save_figure(figure, output / "pareto_performance_cost")

    figure, axis = plt.subplots(figsize=(11, 5)); positions = np.arange(len(frame)); width = .25
    for offset, (column, label) in enumerate((("north_macro_f1", "North"), ("central_macro_f1", "Central"), ("south_macro_f1", "South"))):
        axis.bar(positions + (offset - 1) * width, frame[column], width, label=label)
    axis.set_xticks(positions, frame["display_model"], rotation=25, ha="right"); axis.set_ylabel("Macro-F1 ↑"); axis.legend()
    save_figure(figure, output / "per_accent_model_comparison")

    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, column, label in zip(axes, ("inference_latency_ms", "model_size_mb", "total_parameters"), ("Latency (ms) ↓", "Model size (MB) ↓", "Total parameters ↓")):
        values = pd.to_numeric(frame[column], errors="coerce")
        axis.bar(frame["display_model"], values); axis.set_ylabel(label); axis.tick_params(axis="x", rotation=55)
    save_figure(figure, output / "model_efficiency_comparison")

    (output / "centralized_benchmark_caption.md").write_text(
        "Centralized eight-model comparison on the fixed emotion-stratified test set. "
        "All values and figures are generated from saved prediction-derived metrics; "
        "red markers denote the four-dimensional Pareto front.\n", encoding="utf-8",
    )
    generated = datetime.now(timezone.utc).isoformat()
    source = ";".join(str(path.resolve()) for path in args.metrics)
    script = str(Path(__file__).resolve())
    manifest = pd.DataFrame([
        {"figure_name": name, "source_data": source, "script": script, "generated_at": generated, "description": description}
        for name, description in (
            ("centralized_benchmark_table", "Report-width eight-model metric and cost table."),
            ("pareto_performance_cost", "Complementary four-dimensional performance/cost Pareto views."),
            ("per_accent_model_comparison", "North/Central/South Macro-F1 by model."),
            ("model_efficiency_comparison", "Latency, deployable size, and parameter count."),
        )
    ])
    manifest.to_csv(output / "report_figure_manifest.csv", index=False)


if __name__ == "__main__":
    main()
