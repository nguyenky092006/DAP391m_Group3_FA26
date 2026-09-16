"""Render the locked eight-model B6 validation ranking as a report-ready PNG."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "reports/tables/b6_final/centralized_b6_final_locked.csv"
OUTPUT = ROOT / "reports/figures/b6/final_centralized_b6_ranking.png"

DISPLAY_NAMES = {
    "resnet18_mel3": "ResNet-18",
    "xgboost": "XGBoost",
    "svm_rbf": "SVM RBF",
    "pitch_fusion_paper_aligned": "Pitch-Fusion",
    "wav2vec2_raw": "Wav2Vec2",
    "mobilenet_v3_mel3": "MobileNetV3",
    "vit_b16_mel3": "ViT-B/16",
    "cnn2d_logmel": "CNN2D",
}


def number(value: str, decimals: int = 3) -> str:
    if value in ("", "None", "null"):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def parameters(value: str) -> str:
    if value in ("", "None", "null"):
        return "N/A"
    count = int(float(value))
    return f"{count / 1_000_000:.2f}M" if count >= 1_000_000 else f"{count / 1_000:.1f}K"


def main() -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8 or {row["artifact_status"] for row in rows} != {"FINAL_ACTIVE"}:
        raise RuntimeError("The final locked B6 table must contain exactly eight FINAL_ACTIVE rows")
    rows.sort(key=lambda row: float(row["overall_macro_f1"]), reverse=True)

    columns = [
        "Rank", "Model", "Macro-F1 ↑", "UAR ↑", "Worst-accent F1 ↑",
        "Accent Gap ↓", "Latency ms ↓", "Size MB ↓", "Params",
    ]
    table_rows = []
    for rank, row in enumerate(rows, 1):
        table_rows.append([
            str(rank), DISPLAY_NAMES[row["model"]], number(row["overall_macro_f1"]),
            number(row["uar"]), number(row["worst_accent_macro_f1"]),
            number(row["accent_gap"]), number(row["latency_median_ms"]),
            number(row["model_size_mb"]), parameters(row["total_parameters"]),
        ])

    fig, ax = plt.subplots(figsize=(20, 6.8))
    ax.axis("off")
    ax.set_title(
        "Final Locked B6 Centralized Benchmark — Validation Set (n=646)",
        fontsize=19, fontweight="bold", pad=22,
    )
    ax.text(
        0.5, 0.965,
        "Ranked by Overall Macro-F1; fixed 665-sample test set remains untouched",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#4b5563",
    )
    widths = [0.05, 0.15, 0.10, 0.085, 0.135, 0.105, 0.105, 0.10, 0.085]
    table = ax.table(
        cellText=table_rows, colLabels=columns, cellLoc="center", colLoc="center",
        colWidths=widths, bbox=[0.01, 0.10, 0.98, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#1f4e78")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f8fafc" if row_index % 2 else "#eef4f8")
            if column_index == 1:
                cell.get_text().set_ha("left")
            if row_index == 1:
                cell.get_text().set_fontweight("bold")

    metric_specs = {
        2: (max, "#d9ead3"), 3: (max, "#d9ead3"), 4: (max, "#d9ead3"),
        5: (min, "#d9ead3"), 6: (min, "#d9ead3"), 7: (min, "#d9ead3"),
    }
    for column_index, (chooser, color) in metric_specs.items():
        values = [(row_index, float(table_rows[row_index - 1][column_index])) for row_index in range(1, 9)]
        best = chooser(value for _, value in values)
        for row_index, value in values:
            if value == best:
                table[(row_index, column_index)].set_facecolor(color)
                table[(row_index, column_index)].get_text().set_fontweight("bold")

    ax.text(
        0.01, 0.045,
        "Source: reports/tables/b6_final/centralized_b6_final_locked.csv  •  All values are measured validation evidence",
        transform=ax.transAxes, ha="left", fontsize=10, color="#374151",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
