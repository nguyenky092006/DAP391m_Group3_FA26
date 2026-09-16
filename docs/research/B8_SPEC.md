# B8 — Tuning and Cross-Validation

## Objective

Tune fairly on development evidence, analyze `lambda_adv`, enforce the utility
budget and freeze configurations before the one-time final test.

## Technical Details

- SVM/XGBoost use deterministic emotion-stratified CV with fold-local preprocessing.
- Deep models use active train/validation membership.
- Federated tuning uses development clients only.
- Initial lambda grid: 0, 0.05, 0.10, 0.25, 0.50, 1.00.
- Planning utility budget: no more than 0.01–0.02 absolute Overall Macro-F1
  degradation against the matched baseline.
- ResNet18 is selected, but this specification update does not authorize tuning.
- Prefer a non-zero setting only when it respects the utility budget while
  preferably increasing Worst-accent Macro-F1, reducing Accent Gap and reducing
  accent-probe performance.
- Tune and select only from development evidence; never use the fixed test.
- The E2/E3/E6/E7 controlled runs used the preregistered single value
  `lambda_adv=0.10`; they are not an RQ6 sweep and do not select a final lambda.
- Current RQ6 lambda-sweep status is `NOT_STARTED`.

## Metrics

Stratified-CV mean/std, validation Macro-F1, lambda trade-off, utility-budget
violations, runtime and failed trials.

## Products & Deliverables

Search-space configs, trial registry, lambda ablation and locked final configs.

## Definition of Done

Searches and failures are logged, E2/E3 and E6/E7 use matched non-adversarial
comparators, the utility budget is enforced, and final configs are frozen before
test. Backbone selection is already complete and must not be reopened.
