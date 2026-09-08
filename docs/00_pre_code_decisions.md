# Pre-code protocol decisions

This document is the implementation contract derived from the final project plan,
the DAP391m guide, the supplied VESC resources, and the raw-audio audit. Earlier
planning drafts are historical context; the `plan gom rq` tab is authoritative.

## Non-negotiable scientific rules

1. An underlying utterance is the atomic unit. Its raw and `_vocals` files are two
   views and must always stay in the same fold, partition, and federated client.
2. Speakers must be disjoint between development and the locked final test set.
3. The final test set is selected before hyperparameter tuning and is never used
   for model selection, normalization fitting, threshold selection, or early stopping.
4. Normalization statistics and augmentation choices are fit/configured using
   training data only.
5. `_vocals` means a processed/denoised view. It is not described as clean ground
   truth or source-separated audio without additional provenance evidence.
6. Results from the supplied VESC paper are contextual references, not directly
   comparable unseen-speaker baselines because the supplied original split has
   speaker and paired-view leakage.
7. Neutral has no paired raw/processed utterances in the supplied data. Paired-view
   metrics and Label-Aware PVC coverage must therefore report class support explicitly.
8. OrionNet-LM uses a log-Mel input and is an adaptation of the published OrionNet,
   whose original input is a multi-feature acoustic representation.
9. Federated learning is not called formally privacy-preserving unless a formal
   privacy mechanism such as differential privacy or secure aggregation is implemented.
10. The cloud demo processes uploaded audio on a server. Training-data locality must
    not be conflated with on-device inference.

## Decisions that will be encoded next

- Build an immutable raw manifest and record every manual metadata decision.
- Create a fixed, speaker-disjoint test set and three grouped development folds.
- Use three virtual clients for the main experiment unless a constraint audit proves
  that five clients have adequate label, speaker, and paired-view support.
- Use mono 16 kHz audio, deterministic duration handling, log-Mel features, and
  train-only normalization.
- Keep FedAvg as the canonical federated baseline. FedProx and FedBN are comparison
  baselines; FedBN requires an explicit client-specific evaluation protocol.
- Treat Label-Aware PVC as the proposed method and compare it with no PVC and fixed
  PVC under the same optimizer, round, local-epoch, and aggregation budget.

## Dataset corrections policy

The code never silently changes folder labels based on filenames. Each questionable
sample must receive a human-reviewed entry in a future corrections file containing:

- relative path and checksum;
- observed folder label and filename label;
- decision (`keep`, `relabel`, or `exclude`);
- evidence and reviewer;
- review date.

Until that file exists, raw labels are preserved and anomalies are surfaced by the
manifest audit.

