"""Render two report-ready B6 development-validation tables from saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = [
    "pitch_fusion_paper_aligned", "svm_rbf", "xgboost", "wav2vec2_raw",
    "vit_b16_mel3", "mobilenet_v3_mel3", "resnet18_mel3", "cnn2d_logmel",
]
DISPLAY = {
    "svm_rbf": "SVM RBF", "xgboost": "XGBoost", "cnn2d_logmel": "CNN2D",
    "resnet18_mel3": "ResNet-18", "mobilenet_v3_mel3": "MobileNetV3",
    "vit_b16_mel3": "ViT-B/16 (frozen)",
    "wav2vec2_raw": "Wav2Vec2 (frozen)",
    "pitch_fusion_paper_aligned": "Pitch-Fusion",
}


def fmt(value: object, decimals: int = 4) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.{decimals}f}"


def fmt_params(value: object) -> str:
    return "N/A" if pd.isna(value) else f"{int(value):,}"


def fmt_time(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    seconds = float(value)
    return f"{seconds:.2f} s" if seconds < 120 else f"{seconds / 60:.2f} min"


def render_table(
    shown: pd.DataFrame, numeric: pd.DataFrame, output: Path,
    title: str, high_columns: set[str], low_columns: set[str],
) -> None:
    width = max(10.5, 1.45 * len(shown.columns))
    figure, axis = plt.subplots(figsize=(width, 4.8))
    axis.axis("off")
    axis.set_title(title, fontsize=12, fontweight="bold", pad=16)
    artist = axis.table(
        cellText=shown.values, colLabels=shown.columns, loc="center",
        cellLoc="center", colLoc="center",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(8.5)
    artist.scale(1, 1.45)
    for column_index in range(len(shown.columns)):
        artist[(0, column_index)].set_facecolor("#1f4e78")
        artist[(0, column_index)].set_text_props(color="white", weight="bold")
    for row_index in range(1, len(shown) + 1):
        if row_index % 2 == 0:
            for column_index in range(len(shown.columns)):
                artist[(row_index, column_index)].set_facecolor("#eef3f8")
    for column in high_columns | low_columns:
        values = pd.to_numeric(numeric[column], errors="coerce")
        best = values.max() if column in high_columns else values.min()
        column_index = shown.columns.get_loc(column)
        for row_index, value in enumerate(values, start=1):
            if np.isfinite(value) and np.isclose(value, best):
                artist[(row_index, column_index)].set_facecolor("#d9ead3")
                artist[(row_index, column_index)].set_text_props(weight="bold")
    figure.text(
        0.5, 0.035,
        "Validation set: 646 utterances, speaker_disjoint_v2. Development evidence only — NOT final-test results.",
        ha="center", fontsize=9, style="italic",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "artifacts/metrics/b6_centralized_validation_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures/b6")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    frame = pd.read_csv(source)
    if set(frame["model"]) != set(MODEL_ORDER) or len(frame) != 8:
        raise ValueError("Validation report tables require exactly the canonical eight models")
    frame = frame.set_index("model").loc[MODEL_ORDER].reset_index()
    frame.insert(1, "Model", frame["model"].map(DISPLAY))
    output.mkdir(parents=True, exist_ok=True)

    performance_numeric = pd.DataFrame({
        "Model": frame["Model"], "Macro-F1 ↑": frame["overall_macro_f1"],
        "UAR ↑": frame["uar"], "Accuracy ↑": frame["accuracy"],
        "Worst-accent F1 ↑": frame["worst_accent_macro_f1"],
        "Accent gap ↓": frame["accent_gap"],
    })
    performance_shown = performance_numeric.copy()
    for column in performance_shown.columns[1:]:
        performance_shown[column] = performance_shown[column].map(fmt)
    render_table(
        performance_shown, performance_numeric,
        output / "centralized_validation_performance_table",
        "Table 1. Centralized model performance on the validation set",
        {"Macro-F1 ↑", "UAR ↑", "Accuracy ↑", "Worst-accent F1 ↑"},
        {"Accent gap ↓"},
    )

    cost_numeric = pd.DataFrame({
        "Model": frame["Model"], "Latency ms ↓": frame["inference_latency_ms"],
        "Model size MB ↓": frame["model_size_mb"],
        "Total params ↓": frame["total_parameters"],
        "Trainable params ↓": frame["trainable_parameters"],
        "Train time ↓": frame["training_time_sec"],
        "Feature extraction": frame["feature_extraction_time_sec"],
    })
    cost_shown = cost_numeric.copy()
    cost_shown["Latency ms ↓"] = cost_shown["Latency ms ↓"].map(lambda x: fmt(x, 2))
    cost_shown["Model size MB ↓"] = cost_shown["Model size MB ↓"].map(lambda x: fmt(x, 2))
    cost_shown["Total params ↓"] = cost_shown["Total params ↓"].map(fmt_params)
    cost_shown["Trainable params ↓"] = cost_shown["Trainable params ↓"].map(fmt_params)
    cost_shown["Train time ↓"] = cost_shown["Train time ↓"].map(fmt_time)
    cost_shown["Feature extraction"] = cost_shown["Feature extraction"].map(fmt_time)
    render_table(
        cost_shown, cost_numeric,
        output / "centralized_validation_cost_table",
        "Table 2. Centralized model computational cost on the validation protocol",
        set(), {"Latency ms ↓", "Model size MB ↓", "Total params ↓", "Trainable params ↓", "Train time ↓"},
    )

    print(f"PASS: generated two validation report tables from {source}")


if __name__ == "__main__":
    main()
