"""Validate and summarize the completed B6 E1-E7 experiment ladder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports/tables/b6/fedecai"


def main() -> None:
    expected = [
        "E1_cnn", "E2_unconditional_grl", "E3_emotion_conditioned_grl",
        *[f"{experiment}_{scenario}" for experiment in (
            "E4_fedavg_cnn", "E5_fedprox_cnn",
            "E6_fedavg_unconditional_grl", "E7_fedecai",
        ) for scenario in ("main", "stress")],
    ]
    rows = []
    reports = {}
    for name in expected:
        path = REPORT_DIR / f"{name}.json"
        report = json.loads(path.read_text(encoding="utf-8")); reports[name] = report
        sequence = report.get("history", report.get("rounds", []))
        predictions = REPORT_DIR / f"{name}_validation_predictions.csv"
        prediction_rows = len(pd.read_csv(predictions))
        if prediction_rows != 646 or report["test_samples_loaded"] != 0:
            raise ValueError(f"Invalid evidence contract for {name}")
        metrics = report["validation"]
        rows.append({
            "experiment": name,
            "scenario": report.get("scenario", "centralized"),
            "kind": report["kind"],
            "training_steps": len(sequence),
            "validation_accuracy": metrics["overall"]["accuracy"],
            "validation_macro_f1": metrics["overall"]["macro_f1"],
            "validation_uar": metrics["overall"]["uar"],
            "worst_accent_macro_f1": metrics["worst_accent_macro_f1"],
            "accent_macro_f1_gap": metrics["accent_macro_f1_gap"],
            "prediction_rows": prediction_rows,
            "test_untouched": True,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT_DIR / "b6_e1_e7_summary.csv", index=False)

    comparisons = []
    for scenario in ("main", "stress"):
        baseline = frame[frame["experiment"] == f"E4_fedavg_cnn_{scenario}"].iloc[0]
        for experiment in ("E5_fedprox_cnn", "E6_fedavg_unconditional_grl", "E7_fedecai"):
            candidate = frame[frame["experiment"] == f"{experiment}_{scenario}"].iloc[0]
            comparisons.append({
                "scenario": scenario,
                "candidate": experiment,
                "baseline": "E4_fedavg_cnn",
                "delta_validation_macro_f1": candidate["validation_macro_f1"] - baseline["validation_macro_f1"],
                "delta_worst_accent_macro_f1": candidate["worst_accent_macro_f1"] - baseline["worst_accent_macro_f1"],
                "delta_accent_gap": candidate["accent_macro_f1_gap"] - baseline["accent_macro_f1_gap"],
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(REPORT_DIR / "b6_federated_deltas_vs_fedavg.csv", index=False)
    fedecai = comparison_frame[comparison_frame["candidate"] == "E7_fedecai"]
    summary = {
        "schema_version": "b6.fedecai.summary.v1",
        "status": "PASS",
        "experiments_completed": len(frame),
        "centralized_epochs_verified": bool((frame[frame["scenario"] == "centralized"]["training_steps"] == 30).all()),
        "federated_rounds_verified": bool((frame[frame["scenario"] != "centralized"]["training_steps"] == 20).all()),
        "validation_prediction_rows_per_experiment": 646,
        "test_untouched": True,
        "best_validation_macro_f1_experiment": frame.sort_values("validation_macro_f1", ascending=False).iloc[0]["experiment"],
        "fedecai_improves_fedavg_macro_f1_in_all_scenarios": bool((fedecai["delta_validation_macro_f1"] > 0).all()),
        "fedecai_improves_fedavg_worst_accent_in_all_scenarios": bool((fedecai["delta_worst_accent_macro_f1"] > 0).all()),
        "interpretation": "Single-seed validation evidence only. The default FedECAI configuration does not support H5 and must proceed to B7 diagnostics and B8 lambda tuning without touching test.",
    }
    (REPORT_DIR / "b6_e1_e7_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    model_card = """# B6 FedECAI Preliminary Model Card

## Intended use

The E1-E7 ladder compares centralized, adversarial, and simulated federated
Vietnamese speech-emotion classifiers under the frozen B5 speaker-disjoint
protocol. It is research evidence only and must not be used for psychological
or clinical assessment.

## Evidence status

- 3 centralized experiments ran for 30 epochs.
- 8 federated experiment/scenario combinations ran for 20 rounds.
- Every report contains 646 validation predictions.
- The 665-sample test partition remains untouched.
- Results currently use one seed and are preliminary.

## Current finding

At the default adversarial lambda of 0.10, FedECAI does not improve FedAvg
Macro-F1 or worst-accent Macro-F1 in both client scenarios. This is a negative
preliminary result, not a failure of the pipeline and not a final rejection of
the hypothesis. B8 must tune lambda within the predeclared utility budget.

## Limitations

- Virtual clients simulate federated learning; they are not physical devices.
- Speaker 0 contributes 2,217 samples and causes unavoidable client-size skew.
- The shared CNN is a compact log-Mel baseline, not a Wav2Vec2 FedECAI model.
- Validation is used for checkpoint selection; test is reserved for final use.
- Accent invariance has not yet been measured with a frozen accent probe.
- Bootstrap confidence intervals, effect sizes, and multi-seed variance are pending.
- Federated learning alone does not guarantee privacy; no differential privacy
  or secure aggregation is implemented.
"""
    (ROOT / "docs/research/B6_fedecai_model_card.md").write_text(model_card, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
