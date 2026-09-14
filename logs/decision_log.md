# FedECAI Decision Log

## 2026-09-13 — B4 shared three-channel Mel contract

- Decision: set `mel_3ch_mode` to `repeat_normalized_log_mel` for ResNet-18, MobileNetV3, and ViT.
- Transformation: clip the existing 80-band log-Mel matrix to its fixed `[-80, 0] dB` extractor range, scale to `[0, 1]`, and repeat it into channels-first `(3, 80, T)`.
- Storage: construct the tensor on load from the existing indexed log-Mel artifact; do not create a second 3× artifact collection.
- Duration handling: select `target_frames` from training speakers only, then use the shared center-crop/right-pad adapter for all three consumers.
- Reason: this supplies one deterministic, model-independent three-channel contract without re-extracting audio, fitting full-dataset statistics, or changing the fixed speaker-disjoint split.
- Evidence: `reports/tables/b4/b4_mel_3ch_contract_report.json` validates 4,992/4,992 existing log-Mel artifacts.
- Boundary: this decision defines B4 inputs only. It does not implement or select a B6 model/backbone and does not initiate B5 work.

## 2026-09-13 — B6 current-backbone implementation boundary

- Decision: implement ResNet-18 as the current primary E1-E7 path, as required
  by `PROJECT_RULES.md`, while retaining MobileNetV3-Small as the efficiency
  candidate in the centralized benchmark.
- This is an implementation default, not the final backbone selection. Final
  selection still requires measured Macro-F1, worst-accent Macro-F1, latency,
  model size and federated practicality evidence.
- The compact one-channel CNN E1-E7 reports and original configs remain
  unchanged as legacy preliminary evidence.
- New ResNet-18 experiments use a separate config and distinct output IDs, so
  running them cannot silently resume from or overwrite the legacy CNN ladder.
- The final test remains reserved; no B7 evaluation was started.

## 2026-09-13 — Centralized-only backbone selection gate supersedes prior B6 implementation default

- The attached human workflow establishes a hard stop after the eight-model
  centralized benchmark and candidate analysis.
- No ResNet-18 or MobileNetV3 backbone is selected. The earlier unexecuted
  ResNet-18 implementation-default change from this worktree was withdrawn;
  legacy compact-CNN FedECAI artifacts remain historical only.
- Unconditional GRL, emotion-conditioned GRL, FedAvg, FedProx, federated GRL,
  FedECAI and lambda tuning are prohibited until the human provides an explicit
  `BACKBONE_DECISION`.
- Current evidence status: benchmark preflight BLOCKED; candidate trade-offs
  cannot yet be measured.
- `BACKBONE_SELECTION_STATUS = WAITING_FOR_HUMAN_DECISION`.

## 2026-09-13 — centralized infrastructure ready, decision gate unchanged

- The DAP301m runtime and all eight centralized smoke paths are verified.
- This changes only infrastructure readiness; it does not provide comparative
  evidence for ResNet-18 or MobileNetV3.
- No backbone configuration was selected or changed.
- `BACKBONE_SELECTION_STATUS = WAITING_FOR_HUMAN_DECISION`.
