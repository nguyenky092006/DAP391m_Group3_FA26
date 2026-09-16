# B7 — Evaluation and Pareto Analysis

## Objective

Evaluate prediction-derived evidence consistently for the locked centralized
benchmark and the official ResNet18 E1-E7 ladder without reopening the completed
backbone decision.

## Technical Details

- During development, recompute metrics from saved validation predictions. Use
  the identical fixed test only once after configurations are frozen and an
  explicit later execution instruction allows it.
- Use sample-level bootstrap, stratified by emotion where practical, and record
  the resampling unit accurately.
- Keep the four RQ3 Pareto dimensions unchanged.
- Preserve the archived ResNet-18 versus MobileNetV3 candidate evidence and the
  2026-09-15 human selection; do not reopen or automatically rescore it.
- For E1-E7 require overall Macro-F1, UAR, weighted-F1 and accuracy;
  North/Central/South and worst/best accent Macro-F1, accent gap/std;
  per-emotion F1/recall and per-emotion-within-accent where supported;
  representation accent-probe Macro-F1/UAR; total/trainable parameters, model
  size, training time, inference latency and communication cost where applicable.
- Do not judge FedECAI by Overall Macro-F1 alone.

## Metrics

Overall/per-class emotion metrics, confusion matrices, all accent-robustness
metrics, cost metrics, confidence intervals, effect sizes where appropriate,
federated/client metrics when later authorized, and Pareto-front membership.

## Products & Deliverables

Prediction-derived result tables/figures, error analysis, uncertainty evidence,
four-axis Pareto outputs and E1-E7 comparisons with traceable source data.

## Definition of Done

All reported models use the correct common split membership, metrics are
traceable, uncertainty units are correct, and the completed human selection is
reported truthfully. Final-test completion remains pending until separately
authorized.

## Current validation-only evidence (2026-09-15)

The canonical active centralized evidence is
`reports/tables/b6_final/centralized_b6_final_locked.csv`. The selected E1
reference is `b6_resnet18_mel3_finetune_seed391`, based on 646 saved validation
predictions. The fixed 665-sample test remains untouched.

E2-E7 validation comparison is now complete and traceable in
`reports/tables/fedecai/`. All experiments use the same 646 validation IDs;
checkpoint hashes and replay checks pass. Fixed-test evaluation, approved
uncertainty analysis, and final-test conclusions remain pending. The human
backbone decision is complete. ViT remains part of locked RQ3 but is not an
active FedECAI ladder backbone.
