# FedECAI Project Log

## 2026-09-13 — B1 protocol reconciliation after takeover audit

- Scope: B1 documentation and planning completeness only.
- Reused without rework: existing problem statement, included/out-of-scope boundaries, ViSEC provenance, data dictionary, and local dataset integrity evidence.
- Added `docs/research/B1_SPEC.md`.
- Expanded B1 from RQ1–RQ3/H0 to RQ1–RQ6 and H1–H6.
- Reconciled RQ3 with the required eight centralized models and four Pareto dimensions.
- Added the RQ5 ResNet-18/MobileNetV3 evidence-based backbone policy.
- Added an evidence-grounded research gap, intended contributions, foundational literature, role-based ownership, and complete RQ evidence mapping.
- Updated dataset-provenance status from pending to locally verified where existing audit evidence supports it.
- Updated `doc.md` to report B1 as DONE while keeping downstream implementation gaps explicit.
- No source code, config, split, feature, model, prediction, metric, or result was created or changed.
- No experiment or training was run; no metric was invented.
- `docs/research/B3_SPEC.md`, requested as context, was not present. No B3 file was created or modified.

## 2026-09-13 — B3 completion after takeover audit

- Scope: B3 EDA/SQL completion only; no B4 work was started.
- Reused without regeneration: the existing univariate, bivariate, and multivariate EDA figures and their verified descriptive findings.
- Added `docs/research/B3_SPEC.md` and reconciled B3 with canonical `speaker_disjoint_v2`.
- Changed the B3 pipeline/notebook default from `split_manifest.csv` v1 to `split_manifest_v2_enriched.csv` and added the existing versioned federated speaker assignments as SQL input.
- Retained the four valid SQL queries and added four missing groups: samples/speakers per accent, pairwise split speaker overlap, main/stress federated-client distributions, and explicit manifest-quality JOIN.
- SQL evidence now contains 8/8 named query groups. Computed checks report zero speaker overlap for all three split pairs and no empty cell in the 36-cell split-v2 accent/emotion grid.
- Updated the B1 evidence matrix only to link B3 protocol diagnostics to RQ1/RQ2/RQ5 without treating them as hypothesis evidence.
- Targeted non-training B3 tests passed 2/2 with Python 3.12.9. SQL tables and summary were regenerated with `--skip-figures`; no model training or figure regeneration occurred.
- Added a narrow `.gitignore` exception for `reports/tables/b3/*.csv` and `b3_summary.json` so required B3 evidence remains versionable; report outputs for other steps remain ignored.
- No model metric, experimental result, split assignment, raw/cleaned data, or B4 artifact was created or changed.

## 2026-09-13 — B4 completion after takeover audit

- Scope: only the B4 gaps identified by the takeover audit; no B5 work was started.
- Reused without re-extraction or regeneration: 4,992-row MFCC/eGeMAPS/log-Mel/pitch/Wav2Vec2 evidence, feature table, three advanced figures, and Plotly dashboard.
- Added `mel_3ch_mode=repeat_normalized_log_mel`: fixed `[-80,0] dB` to `[0,1]` normalization, channels-first repeat, and a shared center-crop/right-pad adapter whose target size must be selected from training speakers only.
- The Mel adapter validated 4,992/4,992 indexed log-Mel artifacts on load; no duplicate three-channel artifact collection was created.
- Added a speaker-disjoint development accent-probe pipeline over handcrafted-174 and frozen Wav2Vec2-768 representations with a train-only scaler, majority baseline, 50 training-label permutations, and 2,000 validation-speaker bootstrap resamples.
- Accepted probe lineage: seed 391, split `speaker_disjoint_v2`, feature-table SHA-256 `89ed3bd079786b40a9f0a6e84f5952aae9f71beb8c5ec64048cb7ad4b8de994f`, git commit `675568845895309c157442b85fb2c88af3f9b8a3`, dirty worktree recorded, 0 train-validation speaker overlap, and 0 test samples loaded.
- One initial launch was terminated by a one-second tool timeout. A subsequent draft completed with convergence warnings and was rejected; only the rerun protected by a convergence-error quality gate is accepted. All attempts are recorded in `logs/experiment_registry.csv`.
- Accepted validation diagnostics: handcrafted-174 Macro-F1/UAR `0.4257/0.4937`; Wav2Vec2-768 `0.4905/0.5081`; majority baseline `0.2615/0.3333`. These are not final-test results and do not confirm H1.
- Corrected B4 documentation drift: full Wav2Vec2 extraction is complete, and the current feature-table SHA-256 is `89ed3bd079786b40a9f0a6e84f5952aae9f71beb8c5ec64048cb7ad4b8de994f` rather than the stale `22c98...` value.
- Added narrow `.gitignore` exceptions only for the new B4 contract/probe evidence files.
- Final targeted B4 suite passed 14/14. The refreshed Mel-contract report records config SHA-256 `0d3709e3594e1b0d92c14b51a09e73afc7ec8c42823d503963c0b827cabb2475`.

## 2026-09-13 — B5 Definition-of-Done re-verification

- Scope: read-only verification and missing B5 step documentation; no B6 work was started.
- The takeover audit already marked every B5 implementation requirement DONE, so no split/client code was changed and no generator was rerun.
- Added `docs/research/B5_SPEC.md` as the missing step-level context.
- Direct checks confirm 4,992 eligible and 288 excluded rows; train/validation/test remain 3,681/646/665 samples and 103/22/22 speakers.
- Train-validation, train-test, and validation-test speaker overlaps are all zero; duplicate eligible waveform hashes remain zero.
- Grouped CV has five folds over training only. All 103 training speakers have exactly one CV assignment; no validation/test speaker has a fold.
- `main_mixed_accent_v1` and `stress_accent_domain_v1` each contain three clients and assign the same 103 training speakers with no non-training speaker included.
- Targeted B5 tests passed 4/4. Existing split manifests, client assignments, reports, metrics, and results were not modified.
- B4 documentation was not changed because its canonical split-v2 dependency remains valid and no B5 state changed.

## 2026-09-13 — B6 implementation completion pass after takeover audit

- Scope: only B6 gaps from the takeover audit; no B7 evaluation or model
  training was started, and no metric/result was created.
- `docs/research/B6_SPEC.md` was absent at task start and was added from the
  authoritative project rules plus the audited B6 evidence boundary.
- Preserved the five-model validation reports/checkpoints and compact-CNN E1-E7
  outputs as legacy preliminary evidence. Original `b6_deep_v1.json` and
  `b6_fedecai_v1.json` remain byte-equivalent to HEAD.
- Added ResNet-18, MobileNetV3-Small and ViT-B/16 implementations using the
  shared B4 `repeat_normalized_log_mel` adapter. Their separate config selects
  target frames from the current training partition only.
- The consolidated development runner now requires exactly eight report and
  prediction pairs before publishing an eight-model summary.
- Added a separate ResNet-18 E1-E7 config/path with shared encoder, emotion
  classifier, GRL, accent discriminator and emotion conditioning. Output IDs
  differ from the legacy CNN ladder.
- New B6 run paths record config/seed/split/commit/output lineage, checkpoint
  hash and registry entries. Historical missing fields were not guessed.
- Non-training checks: 6/6 common/eight-model contract tests passed; 9/9 deep
  model shape/GRL tests passed in the available Torch runtime; image-model and
  ResNet-FedECAI preflights passed with 3,681 train and 646 validation rows,
  `test_loaded=false`.
- Independent data check retained 0 speaker overlap and derived 397 target
  frames from train only; a sampled adapted tensor had shape `(3, 80, 397)`.
- B6 remains PARTIAL because no checkpoints/results exist yet for the three new
  image models, the evidence-based backbone decision is pending, and the new
  ResNet-18 E1-E7 ladder has not run. Wav2Vec2/Pitch-Fusion variant boundaries
  remain as documented. The final test is still unopened.
- Runtime verification found the callable Torch environment is 2.5.1 CPU-only
  (`cuda=false`), while the project-pinned CUDA environment is absent. No
  resource-heavy run was launched in this mixed unsupported environment.

## 2026-09-13 11:31–11:41 +07:00 — B1–B6 centralized benchmark gate

- Applied the human-specified centralized-only workflow: no adversarial,
  federated, lambda-tuning or automatic-backbone-selection work was run.
- Audited B1–B6 specs, the relevant master-plan B6/completion gates, canonical
  manifest, configs, model registry, metric/prediction code, artifacts and
  runtime. `B2_SPEC.md` remains a documentation gap.
- Persisted the audit at `artifacts/audit/B1_B6_benchmark_audit.md` and automated
  preflight at `artifacts/audit/benchmark_preflight.json`.
- Data gates passed: 4,992 eligible rows; 147 released speaker IDs; expected
  four emotions/three accents; 0 speaker overlap; 0 cross-split duplicate
  hashes; fixed test has 665 samples/22 speakers.
- Added an isolated 168-feature MFCC/eGeMAPS-only classical config while keeping
  the legacy 174-feature config intact.
- Extended the shared metric and prediction schemas required for future
  sample-level comparison. No metric value or prediction was generated.
- Final preflight remains BLOCKED by six items: Wav2Vec2 registry mismatch,
  missing raw-waveform Wav2Vec2 runner, missing common fixed-test evaluator,
  incomplete computational-cost protocol, and absent Torch/Transformers in the
  callable unified runtime.
- Three preflight attempts have unique run IDs and retained stdout/stderr,
  summaries, environment records, null metrics and failure histories under
  `logs/runs/`; `logs/run_index.csv` indexes all attempts.
- Created blocked-state centralized and backbone-selection reports. No Pareto
  frontier or candidate metric is claimed; human backbone decision remains pending.
- Corrected the prior unexecuted ResNet-FedECAI implementation direction to the
  current hard gate. Partner legacy CNN/FedECAI artifacts were not deleted or run.
- Next allowed action: resolve centralized preflight blockers and rerun under a
  new run ID. Do not begin FedECAI.

## 2026-09-13 13:51–13:56 +07:00 — centralized infrastructure resolution

- Located and used the existing uv-managed `DAP301m/Scripts/python.exe`; no new
  environment was created and Torch/Transformers were not reinstalled.
- Verified Python 3.12.13, Torch 2.11.0+cu126, Transformers 4.57.6, CUDA 12.6
  and NVIDIA GeForce RTX 4060 Laptop GPU. Added the genuinely missing
  torchvision 0.26.0+cu126 to DAP301m. Torchaudio remains absent and optional.
- Replaced the official registry entry with `wav2vec2_raw` and added a mono-16-kHz
  raw waveform runner whose Wav2Vec2 encoder is fully frozen.
- Added one shared prediction validator/evaluator and shared parameter count,
  deployable-size, training-time and deployment-like latency utilities. Primary
  RQ3 latency is median batch-1 model-forward latency after warm-up; equivalent
  preprocessing is excluded, and Wav2Vec2 includes its encoder.
- Added a canonical artifact/run-output contract and a report generator that
  refuses to run until exactly eight complete real metrics files are provided.
- First validation run `20260913_135126_centralized_infrastructure_seed391`
  retained a harness failure caused by missing `PYTHONPATH=src`; all eight model
  smoke paths still passed. The harness was corrected without hiding the failure.
- Retry `20260913_135229_centralized_infrastructure_seed391` passed; final rerun
  `20260913_135517_centralized_infrastructure_seed391` also passed. After wiring
  the canonical output contract into every trainer, the definitive run
  `20260913_140149_centralized_infrastructure_seed391` passed with 11/11 unit
  tests and 8/8 model smoke paths.
- Final normalization/latency-path verification run
  `20260913_140332_centralized_infrastructure_seed391` was an intermediate READY
  verification; it also passed 11/11 unit tests and 8/8 model smoke paths.
- Compatibility publication for the raw Wav2Vec2 report was then wired into the
  existing eight-model summary. The superseding final run
  `20260913_140535_centralized_infrastructure_seed391` remained READY with 11/11
  unit tests and 8/8 smoke paths.
- The benchmark orchestrator was finally locked to the absolute DAP301m
  interpreter. Superseding run `20260913_140723_centralized_infrastructure_seed391`
  remained READY with 11/11 unit tests and 8/8 smoke paths. It was later
  superseded by the retained READY run
  `20260913_155544_centralized_infrastructure_seed391`.
- Updated preflight status is READY with no blocking failures. No full training,
  accepted metric, test evaluation, Pareto calculation, backbone selection,
  GRL, federated or FedECAI operation occurred.

## 2026-09-14 — repository documentation and log consolidation

- Kept the fixed eight-model protocol unchanged, including ViT. No model was
  trained, selected, renamed or removed; no metric/result was created.
- Consolidated 30 B1–B6 research documents into six authoritative files:
  `docs/research/B1_SPEC.md` through `docs/research/B6_SPEC.md`. Updated active
  script/report destinations and references so removed documents are not
  regenerated or linked as current evidence.
- Replaced the stale five-model/`.venv` README instructions with the verified
  eight-model/DAP301m centralized workflow. Updated `doc.md` to the current
  reader-facing status.
- Merged four root terminal logs into `logs/legacy_terminal_runs.log`.
- Preserved all 11 prior infrastructure-run summaries in
  `logs/infrastructure_run_history.csv`; archived the ten superseded complete
  run directories in `artifacts/archive/infrastructure_runs_pre_155544.zip`,
  then retained only the latest full READY run
  `20260913_155544_centralized_infrastructure_seed391` under `logs/runs/`.
- Archived the legacy preliminary FedECAI report directory in
  `artifacts/archive/legacy_fedecai_reports.zip`. These files remain out of the
  current centralized scope and are recoverable from the archive.
- Kept `artifacts/b6/deep`, `artifacts/.uv-cache`, all current checkpoints,
  predictions, metrics, B4 lineage reports and `data/interim`; checkpoint and
  resume evidence was not deleted.
- The exact cleanup/recovery record is
  `artifacts/audit/repository_cleanup_manifest.csv`.
- Post-cleanup verification used the exact DAP301m interpreter: affected files
  compiled, 11/11 centralized-infrastructure tests passed, and the lightweight
  hard-gate preflight remained READY with 0 speaker overlap, 0 cross-split
  duplicate waveform hashes and all eight smoke paths still marked PASS.

## 2026-09-14 09:15–09:33 +07:00 — time-efficient B6 development validation

- Kept the required eight-model centralized protocol unchanged. No final-test,
  GRL, FedAvg, FedProx, federated, FedECAI or backbone-selection action ran.
- Revalidated the DAP301m environment and 8/8 smoke paths in run
  `20260914_091536_centralized_infrastructure_seed391`; status was READY.
- Persisted the user-reported uncached Wav2Vec2 interruption after epochs 1–6
  as `INTERRUPTED`; no metric was reconstructed from the terminal excerpt.
- Reused complete CNN2D, ResNet-18 and MobileNetV3 artifacts. Trained missing
  SVM/XGBoost artifacts, a frozen-pretrained ViT head over a validated 4,327-row
  cache, and an official raw-waveform frozen-Wav2Vec2 head over the validated B4
  train+validation cache. Pitch-Fusion reused its completed epoch-30 state.
- The first orchestration stopped after SVM because a prior CSV rewrite had
  corrupted the registry header; the second reached Pitch-Fusion but its FP16
  softmax output failed the common probability-sum tolerance. Both failures are
  retained in `logs/runs/20260914_091713_b6_time_efficient_seed391/` and
  `logs/runs/20260914_091828_b6_time_efficient_seed391/`.
- Corrected only probability export by applying softmax to FP32 logits. The
  reused Pitch-Fusion checkpoint then passed the shared evaluator with 646/646
  validation predictions and Macro-F1 0.6022958153; no retraining occurred.
- Canonical development artifacts now pass for all 8 models. Consolidated real
  metrics are at `artifacts/metrics/b6_centralized_validation_summary.csv` and
  `reports/tables/b6/b6_all_models_summary.json`; test remains untouched.
- Tensor equality checks proved the 2,236,763,253-byte Pitch-Fusion resume file
  and 567,581,198-byte legacy artifact duplicated the best state retained in
  `artifacts/checkpoints/b6_pitch_fusion_paper_aligned_seed391.pt`. The two
  redundant copies were deleted; hashes/recovery evidence are in
  `artifacts/audit/repository_cleanup_manifest.csv`.
- B6 remains PARTIAL until one fixed-test evaluation per frozen model, report
  figures/Pareto outputs and an explicit human backbone decision are completed.
- Post-change verification passed 20/20 targeted unit tests. A preliminary
  manual invocation without `PYTHONPATH=src` failed during test import only;
  rerunning with the repository import path passed. The definitive persisted
  infrastructure run `20260914_093958_centralized_infrastructure_seed391`
  passed all harness checks, 8/8 smoke paths and the hard gate with status READY.
- Exported two report-ready development tables directly from
  `artifacts/metrics/b6_centralized_validation_summary.csv`: one performance/
  accent-robustness table and one computational-cost table. After correcting
  output placement, exactly two PNG files are retained under
  `reports/figures/b6/`; the temporary `artifacts/report/` outputs were removed.
  Titles and footnotes identify the fixed 646-sample validation partition and
  prohibit interpretation as final test.

## 2026-09-14 15:31–15:56 +07:00 — recovery after incorrect old-code clone

- Detected that branch `PhiHIen` at `ad6c3c7` retained ten newly generated B6
  reports/figures but had reverted the rest of the repository to the older
  `6755688` structure: consolidated specs/logs and centralized infrastructure
  modules were absent, while legacy duplicated docs and preliminary FedECAI
  reports had reappeared.
- Recovered the latest consolidated code/config/docs/log structure from reflog
  commit `3b2376f` without resetting HEAD or changing branches, then retained
  the ten current B6 reports/validation predictions and two B6 table figures
  from `ad6c3c7`. Ignored datasets, caches, metrics and checkpoints were not
  overwritten.
- The restored tree again contains exactly six research specs, the frozen-cache
  ViT/raw-Wav2Vec2 runners, shared evaluator/cost utilities, consolidated logs
  and exactly two report figures under `reports/figures/b6/`. Legacy preliminary
  FedECAI report files are absent from the working tree and remain recoverable
  from commit `ad6c3c7` or the existing archive.
- Post-recovery verification passed 20/20 targeted unit tests, validated all
  eight canonical development artifact sets with 646 identical validation IDs,
  and completed infrastructure run
  `20260914_155444_centralized_infrastructure_seed391` with status READY and 8/8
  smoke paths. No model was retrained and the fixed test remained untouched.
