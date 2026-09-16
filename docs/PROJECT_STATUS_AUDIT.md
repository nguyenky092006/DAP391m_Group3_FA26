# FedECAI Project Status — Protocol Reset Audit

Audit date: 2026-09-14; active-status addendum: 2026-09-15 after the explicit
final B6 lock and backbone decision.

## Effective protocol

- Dataset: `visec_clean_eligible_v1`, 4,992 eligible unique-waveform samples.
- Split: `emotion_stratified_v1`, seed 391, sample-level emotion stratification.
- Counts: train 3,681, validation 646, test 665.
- Primary emotion stratification plus secondary accent balancing retains all 36
  split–accent–emotion cells.
- Speaker overlap: allowed and diagnostic; 95/102/78 for train–validation,
  train–test and validation–test.
- Exact waveform/file identity duplication across splits: zero, hard-gate PASS.
- Deterministic rerun manifest SHA-256:
  `d1ff69387dedfee1a07ec46c7e157b04eb322b32adddf4bef6248b56faa9025f`.
- Classical CV: sample-level emotion-stratified folds with fold-local preprocessing.
- Fixed test: one membership for all eight models; no test tuning.

## Current step state

| Step | Status | Evidence boundary |
| --- | --- | --- |
| B1 | DONE | Rules, RQs and backbone policy migrated. |
| B2 | DONE | Raw/clean manifests and quality audit remain valid. |
| B3 | DONE | SQL database, query tables and figures regenerated from `emotion_stratified_v1`. |
| B4 | DONE | Mel contract revalidated; accent probe and active split-specific caches regenerated. |
| B5 | DONE | New split/report and regenerated main/stress client assignments pass. |
| B6 | LOCKED | Eight FINAL_ACTIVE validation runs; superseded evidence archived; centralized retraining disabled. |
| B7 | PARTIAL | Canonical final table exists and ResNet-18 is human-selected; no federated evaluation has started. |
| B8 | NOT STARTED | Backbone gate resolved; future tuning requires a separate explicit execution request. |
| B9 | NOT STARTED | Deployment remains downstream. |
| B10 | NOT STARTED | Final report requires future evidence. |

## Removed from active state

- Old split manifests/reports and `federated_speaker_partitions_v1.csv`.
- `artifacts/checkpoints`, `artifacts/predictions`, `artifacts/metrics`,
  `artifacts/b6`, `artifacts/model_selection` and old audit/archive contents.
- Split-specific `data/interim/features/b6_frozen_embeddings`.
- Old B3 generated tables, B4 accent-probe results and B6 tables/figures.
- Old `logs/runs`, infrastructure history and terminal-history files.
- Experiment registry and run index were reset to header-only files.

The execution safety layer rejected permanent bulk deletion of large binary
directories. Those directories were moved out of every active runner path to
`protocol_reset_quarantine/20260914_old_split/`, which is ignored by Git and is
recoverable. They are not current evidence and must not be reused or relabelled.

## Preserved because split-independent

- Raw ViSEC audio/source metadata and B2 raw/clean manifests.
- B2 quality/duplicate/invalid-audio evidence.
- Deterministic per-sample B4 raw feature artifacts and extraction reports.
- Shared Mel adapter and reusable extraction/model/evaluator source.
- Local pretrained Hugging Face/Torch encoder files required for future inference.
- DAP301m environment, dependency files and project configs migrated to the new split.
- AI audit: no prior dedicated file existed, so no history was reconstructed;
  the reset boundary is now recorded in `logs/ai_audit_log.md`.

## Backbone gate

ResNet-18 and MobileNetV3 remain candidates. No active rule names either as
primary/default/fallback. The comparable development table now exists and status
is `AWAITING_HUMAN_DECISION`; only the human may select. Federated/adversarial
training remains prohibited.

## Next files permitted to change

1. New B4 accent-probe outputs under `reports/tables/b4/`.
2. New split-specific ViT/Wav2Vec2 caches under
   `data/interim/features/b6_frozen_embeddings/`.
3. New B6 checkpoints, predictions, metrics, reports and per-run logs.
4. B7 candidate/Pareto outputs after all comparable evidence exists.

No old metric may be copied into these outputs.

Lightweight migration verification passed 93/93 repository unit tests and
validated 4,992/4,992 reusable log-Mel artifacts through the shared 3-channel adapter.

## Stale-reference classification

The required post-migration search was run over rules, plan, docs, configs,
source, tests, SQL, notebook, scripts and active logs.

| Occurrence group | Classification | Reason |
| --- | --- | --- |
| Rules/spec/probe wording that rejects a purely speaker-independent interpretation | ACTIVE_VALID | Required limitation under allowed speaker overlap. |
| B3 SQL comment about non-zero speaker overlap | ACTIVE_VALID | Required diagnostic interpretation. |
| Removed old filenames named in this audit | HISTORICAL | Cleanup evidence only, not an active path. |
| `create_speaker_split.py`, `create_speaker_split_v2.py`, `build_b5_partitions.py` | CODE_REFERENCE | Explicitly historical/inactive and not imported by the active pipeline. |
| Compact-CNN functions retained inside the guarded FedECAI runner | CODE_REFERENCE | Runner hard-stops before these functions; they cannot produce active evidence. |

```text
STALE_SPEAKER_DISJOINT_REFERENCES = 0
STALE_PRESELECTED_BACKBONE_REFERENCES = 0
```
