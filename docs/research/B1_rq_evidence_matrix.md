# B1 Research Question and Evidence Matrix

## Purpose

This document defines the evidence required to answer RQ1-RQ3. It does not contain experimental results. Every metric listed here must later be generated from saved predictions or measurements by reproducible code.

## Common evaluation protocol

- Use one finalized speaker-disjoint train, validation, and test split for all five models.
- Fit preprocessing, feature normalization, feature selection, and model parameters using training data only.
- Use validation data for model selection and hyperparameter tuning.
- Keep the test set unchanged and use it only after the experimental configuration is finalized.
- Apply augmentation only to training samples.
- Save sample identifiers, speaker identifiers, accent labels, true emotion labels, predictions, and class probabilities for evaluation.
- Fix and record random seeds before final experiments.
- Report bootstrap 95% confidence intervals using a speaker-aware resampling unit where possible.
- Treat accent as an analysis/domain variable. Emotion remains the main prediction target.

## Evidence matrix

| RQ | Data and target | Required experiment | Main metrics | Required evidence files | Decision rule | Status |
| --- | --- | --- | --- | --- | --- | --- |
| RQ1 | Audio representations: MFCC/eGeMAPS, log-Mel, pitch-related features, and Wav2Vec2 embeddings. Secondary target: Northern, Central, and Southern accent. | Freeze each representation. Train the same simple accent-probe protocol on training speakers, tune on validation speakers, and evaluate on unseen test speakers. Compare with majority and permutation-label baselines. | Accent-probe Macro-F1, UAR, confusion matrix, majority baseline, permutation-label baseline, bootstrap 95% CI. | `data/manifests/split_manifest.csv`; `artifacts/rq1/accent_probe_metrics.csv`; `artifacts/rq1/bootstrap_ci.csv`; `reports/figures/rq1_accent_confusion_matrix.png`. | A representation retains measurable accent information when its probe result is consistently above the baselines. Compare representations by Macro-F1 and confidence intervals. Do not interpret the probe as the main application model. | Planned |
| RQ2 | Inputs required by each of the five emotion models. Target: Angry, Happy, Neutral, and Sad. Accent is used to group results. | Train SVM, XGBoost, CNN 2D, Wav2Vec2, and Pitch-Fusion on the same finalized split. Evaluate every model overall and separately for each accent on unseen test speakers. | Overall Macro-F1, per-accent Macro-F1, worst-accent Macro-F1, maximum-minus-minimum accent gap, per-class F1, confusion matrix, bootstrap 95% CI. | `artifacts/rq2/test_predictions.csv`; `artifacts/rq2/overall_metrics.csv`; `artifacts/rq2/per_accent_metrics.csv`; `artifacts/rq2/bootstrap_ci.csv`; `reports/figures/rq2_emotion_confusion_matrix.png`. | A model is more consistent across accents when it has a higher worst-accent Macro-F1 and a smaller accent gap without a material loss in overall Macro-F1. Conclusions must consider confidence intervals. | Planned |
| RQ3 | Results from the same five emotion models plus measured computational cost. Target remains the four emotion classes. | Combine the finalized RQ2 test metrics with training-time, inference-latency, model-size, and parameter-count measurements collected under a documented environment. Perform Pareto analysis. | Overall Macro-F1, worst-accent Macro-F1, training time, inference latency, model size, parameter count, Pareto dominance. | `artifacts/rq3/model_comparison.csv`; `artifacts/rq3/latency_runs.csv`; `artifacts/rq3/environment.json`; `reports/figures/rq3_pareto_plot.png`. | Prefer Pareto-efficient models. Do not create an arbitrary weighted score unless its weights are fixed before viewing final test results. The recommended model must state the performance and cost trade-off explicitly. | Planned |

## Evidence definitions

### Split manifest

`data/manifests/split_manifest.csv` will be the shared source of truth for sample assignment. At minimum, it must contain:

- `sample_id`
- `audio_path` or a repository-safe relative identifier
- `speaker_id`
- `emotion`
- `accent`
- `split`

Before modeling, automated checks must confirm that no speaker occurs in more than one split.

### Prediction records

The RQ2 prediction file must contain one row per evaluated sample and enough information to reproduce all reported metrics:

- `sample_id`
- `speaker_id`
- `accent`
- `y_true`
- `y_pred`
- one probability column for each emotion class
- `model_name`
- `run_id`

### Computational measurements

- Training time must specify whether feature extraction is included.
- Inference latency must record warm-up runs, measured runs, batch size, device, and software environment.
- Model size must use the saved deployable artifact or clearly state what file is measured.
- Parameter count applies to trainable neural models. For SVM and XGBoost, use model size and latency instead of forcing an incomparable parameter-count interpretation.

## Relationship to the project extension

RQ1-RQ3 and the five-model comparison form the core project evidence. Unconditional GRL, emotion-conditioned GRL, FedAvg, FedProx, and FedECAI are later extension experiments. Their results must be stored separately and must not replace the evidence required for the five core models.

## B1.3 acceptance criteria

- Each research question maps to a concrete experiment.
- Each research question has defined metrics and future evidence files.
- RQ1 clearly uses accent prediction only as a diagnostic probe.
- RQ2 evaluates emotion performance overall and by accent.
- RQ3 compares performance and computational cost without an arbitrary post-hoc score.
- All evidence paths are marked as future outputs, not existing results.
- The common protocol prevents speaker leakage and test-set tuning.
