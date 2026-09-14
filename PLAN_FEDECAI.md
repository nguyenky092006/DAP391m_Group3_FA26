# PLAN_FEDECAI.md

> **Master Project Plan — FedECAI**
>
> This file is the **master specification** for the project.  
> It is **NOT intended to be read in full for every Codex task**.
>
> Use the context-loading policy below to reduce token usage and avoid irrelevant context.

---

# 0. Context Loading Policy for Codex

## 0.1 Default rule

For normal implementation tasks, Codex must **NOT read this entire file**.

Codex should load context in this order:

1. `PROJECT_RULES.md` — always read.
2. The **single relevant step specification**:
   - B1 → `docs/research/B1_SPEC.md`
   - B2 → `docs/research/B2_SPEC.md`
   - B3 → `docs/research/B3_SPEC.md`
   - B4 → `docs/research/B4_SPEC.md`
   - B5 → `docs/research/B5_SPEC.md`
   - B6 → `docs/research/B6_SPEC.md`
   - B7 → `docs/research/B7_SPEC.md`
   - B8 → `docs/research/B8_SPEC.md`
   - B9 → `docs/research/B9_SPEC.md`
   - B10 → `docs/research/B10_SPEC.md`
3. Read adjacent step specs only when the task crosses boundaries.
4. Read this master plan only when the task changes a **project-wide rule**.

## 0.2 Read this master plan only when

Codex may read `PLAN_FEDECAI.md` when:

- starting a new major project phase;
- changing the dataset or label definitions;
- changing the speaker-disjoint policy;
- changing the list of benchmark models;
- changing RQs or hypotheses;
- changing the federated backbone decision policy;
- changing Pareto dimensions;
- changing the cloud/service architecture;
- changing the final report structure;
- resolving conflicting requirements across B-steps.

## 0.3 Never read all specs by default

Codex must not load all `B1_SPEC.md` ... `B10_SPEC.md` for a single normal task.

Examples:

- Implement ResNet-18 benchmark → read `PROJECT_RULES.md` + `B6_SPEC.md`
- Add Pareto plotting → read `PROJECT_RULES.md` + `B7_SPEC.md`
- Implement lambda ablation → read `PROJECT_RULES.md` + `B8_SPEC.md`
- Add Cloud Run deployment → read `PROJECT_RULES.md` + `B9_SPEC.md`

## 0.4 Documentation rule

`PLAN_FEDECAI.md` = master specification.  
`PROJECT_RULES.md` = compact always-on rules.  
`docs/research/B*_SPEC.md` = task-level implementation context.  
`doc.md` = human-readable project documentation, **not AI context**.  
`logs/*` = historical evidence and audit trail.

---

# 1. Project Definition

## English title

**Emotion-Conditioned Accent-Invariant Learning for Federated Vietnamese Speech Emotion Recognition**

## Method

**FedECAI — Federated Emotion-Conditioned Accent-Invariant Learning**

## Dataset

**ViSEC — Vietnamese Speech Emotion Corpus**

Project assumptions to verify by code:

- 5,280 speech samples.
- 4 emotion classes:
  - Angry
  - Happy
  - Neutral
  - Sad
- 3 accent groups:
  - North
  - Central
  - South
- Speaker-disjoint evaluation.
- Exact speaker count must be computed from dataset metadata.

---

# 2. Research Questions

## RQ1 — Accent information in representations

How much accent information remains in speech representations when evaluated on unseen speakers?

Main evidence:

- Accent-probe Macro-F1
- Accent-probe UAR
- Majority baseline
- Permutation baseline
- Speaker-bootstrap 95% CI

Primary steps: B4, B6, B7.

---

## RQ2 — Performance disparity across accents

How different is emotion-recognition performance across North, Central, and South accents?

Main evidence:

- Overall Macro-F1
- Per-accent Macro-F1
- Worst-accent Macro-F1
- Accent gap
- Per-class F1 within accent

Primary steps: B3, B5, B7.

---

## RQ3 — Centralized benchmark and Pareto trade-off

Compare exactly 8 models:

1. SVM
2. XGBoost
3. CNN 2D
4. ResNet-18 — 3-channel Mel-Spectrogram
5. MobileNetV3 — 3-channel Mel-Spectrogram
6. ViT — 3-channel Mel-Spectrogram
7. Wav2Vec2
8. Pitch-Fusion

Required Pareto dimensions:

- Overall Macro-F1 ↑
- Worst-accent Macro-F1 ↑
- Inference Latency (ms) ↓
- Model Size (MB) ↓

Primary steps: B6, B7, B8, B9.

---

## RQ4 — Conditional vs unconditional adversarial learning

Compare:

- no GRL;
- unconditional GRL;
- emotion-conditioned GRL.

Main evidence:

- Emotion Macro-F1
- Worst-accent Macro-F1
- Accent gap
- Accent-probe Macro-F1

Primary steps: B6, B7, B8.

---

## RQ5 — FedECAI in Federated Learning

Compare:

- FedAvg
- FedProx
- FedAvg + unconditional GRL
- FedECAI

Federated backbone policy:

- Primary: **ResNet-18**
- Efficiency alternative: **MobileNetV3**

Selection must be evidence-based using performance + latency + size + FL practicality.

Primary steps: B5, B6, B7, B8.

---

## RQ6 — Utility vs accent invariance

Study the trade-off controlled by adversarial weight `lambda_adv`.

Main evidence:

- Emotion Macro-F1
- Worst-accent Macro-F1
- Accent-probe Macro-F1
- Mean/Worst-client Macro-F1
- Stability across seeds
- Pareto frontier

Primary steps: B7, B8.

---

# 3. Project-Wide Non-Negotiable Rules

1. Speaker-disjoint split is mandatory.
2. No speaker leakage between train/validation/test.
3. Same fixed test set for all centralized models.
4. No test-set tuning.
5. Training-only fitting for scalers/preprocessing.
6. Training-only augmentation.
7. Macro-F1 is the main emotion metric.
8. Accuracy must never be the only metric.
9. Per-accent evaluation is mandatory.
10. Worst-accent Macro-F1 is mandatory.
11. Accent invariance must be measured with a probe.
12. UMAP/t-SNE are illustrative only.
13. All experiments need config + seed + git commit + split version.
14. No fabricated metrics, citations, dataset counts, or conclusions.
15. Federated Learning is simulated unless explicitly changed.
16. Do not claim FL alone guarantees privacy.
17. Do not use the system for psychological diagnosis.
18. Major protocol changes must be logged.

---

# 4. B1 — Problem Definition and Research Scope

## Objective

Define the problem, research gap, six RQs, six hypotheses, baseline literature, scope, intended contribution, and evidence requirements.

## Technical Details

Required content:

- Problem Statement
- Research Gap
- Expected Contributions
- RQ1–RQ6
- H1–H6
- Baseline papers
- Scope / Out-of-scope
- Owners
- Evidence mapping

Minimum foundational references:

- ViSEC / Pitch-Fusion
- DANN / GRL
- FedAvg
- FedProx

## Metrics

Planning completeness:

- 6/6 RQs
- 6/6 hypotheses
- 6/6 RQ-to-evidence mappings
- 0 unsupported novelty claims

## Products & Deliverables

- `docs/research/B1_SPEC.md`
- `doc.md` update
- project/decision log update

## Definition of Done

- Problem statement is explicit.
- All six RQs and hypotheses exist.
- RQ3 lists all 8 models.
- RQ5 contains ResNet-18/MobileNetV3 backbone policy.
- Research gap is evidence-based.
- Scope is explicit.
- Human-facing docs updated.

---

# 5. B2 — Data Understanding and Quality Audit

## Objective

Verify dataset provenance, schema, label integrity, audio quality, speaker statistics, duplicates, missing data, and preprocessing risks.

## Technical Details

Required checks:

- metadata row count;
- audio file count;
- valid paths;
- unique speakers;
- emotion distribution;
- accent distribution;
- gender distribution if available;
- duration stats;
- sample rate;
- channels;
- corrupted audio;
- silence/clipping;
- missing metadata;
- duplicate files/waveforms;
- inconsistent emotion mappings;
- speakers mapped to multiple accents;
- rows without audio.

Preprocessing rules:

- target 16 kHz where required;
- no manual audio editing;
- no default peak normalization;
- VAD/denoising/RMS normalization require evidence;
- all preprocessing scripted/config-driven.

## Metrics

- invalid-file rate
- missing-data rate
- duplicate rate
- class/accent distributions
- speaker distribution
- duration quantiles

## Products & Deliverables

- `docs/research/B2_SPEC.md`
- `data/processed/visec_manifest.csv`
- `artifacts/data_audit.json`
- `artifacts/invalid_audio.csv`
- `artifacts/duplicate_report.csv`
- `reports/tables/b2/b2_quality_report.md`
- `doc.md` update
- logs update

## Definition of Done

- Exact speaker count is computed.
- Every exclusion has a reason.
- Label mappings are verified.
- Dataset quality issues are documented.
- Raw manifest is versioned.

---

# 6. B3 — EDA and Advanced SQL

## Objective

Analyze data composition, imbalance, potential confounders, speaker patterns, and RQ-supporting SQL evidence.

## Technical Details

Required EDA:

- univariate;
- bivariate;
- multivariate.

Required SQL examples:

1. samples/speakers per accent;
2. emotion distribution within accent;
3. speaker ranking with `RANK()`;
4. least-represented emotion per accent;
5. speaker overlap across splits;
6. federated-client distributions;
7. manifest + quality JOIN;
8. CTE/window-function emotion proportions.

Every query must map to an RQ.

## Metrics

- class imbalance
- accent imbalance
- speaker concentration
- subgroup counts
- duration distribution
- potential confounders

## Products & Deliverables

- `docs/research/B3_SPEC.md`
- `notebooks/B3_eda_sql.ipynb`
- `sql/B3_queries.sql`
- DuckDB/SQL database if used
- `reports/tables/b3/`
- `doc.md` update

## Definition of Done

- Univariate/bivariate/multivariate EDA exists.
- Advanced SQL exists.
- RQ mapping exists.
- Every major chart/table has a quantitative takeaway.

---

# 7. B4 — Feature Engineering and Representation Analysis

## Objective

Create reproducible features for all 8 models, define the 3-channel Mel pipeline, and run accent-probe analysis.

## Technical Details

### Classical features

MFCC/eGeMAPS for:

- SVM
- XGBoost

### Mel pipeline

Used by:

- CNN 2D
- ResNet-18
- MobileNetV3
- ViT

Configurable:

- sample rate
- n_fft
- hop length
- n_mels
- frequency range
- duration
- padding/truncation

### 3-channel Mel

For ResNet-18, MobileNetV3, ViT.

Use one consistent policy:

- preferred simple baseline: repeat normalized log-Mel across 3 channels;
- optional alternative: log-Mel + delta + delta-delta if explicitly validated.

Record `mel_3ch_mode`.

### Pitch

For Pitch-Fusion and RQ1:

- F0 contour
- mean/std/range
- voiced fraction

### Wav2Vec2

- raw 16 kHz waveform
- frozen baseline first
- partial fine-tuning only after baseline is stable

### Accent probe

- frozen embedding
- simple accent classifier
- speaker-disjoint
- Macro-F1/UAR
- majority baseline
- permutation baseline
- speaker-bootstrap CI

## Metrics

- feature extraction success
- feature dimensions
- probe Macro-F1/UAR
- runtime/storage where useful

## Products & Deliverables

- `docs/research/B4_SPEC.md`
- `src/features/`
- `configs/features/`
- accent-probe outputs
- advanced visualizations
- `doc.md` update

## Definition of Done

- Every feature family is reproducible.
- One documented 3-channel Mel policy is used.
- Accent probe is speaker-disjoint.
- At least three advanced visualizations exist.

---

# 8. B5 — Dataset Partition and Federated Clients

## Objective

Create versioned speaker-disjoint global splits and federated client partitions.

## Technical Details

### Global split

Rules:

- same speaker belongs to one split only;
- fixed test set;
- no audio-level random leakage;
- augmentation only in train;
- reasonable emotion/accent balance where possible.

### Cross-validation

- GroupKFold / StratifiedGroupKFold where appropriate;
- grouping by `speaker_id`.

### Federated partitions

At least:

- Main partition: moderate accent skew.
- Stress partition: strong accent/emotion skew.

Each client records:

- sample count
- speaker count
- accent distribution
- emotion distribution

## Metrics

- speakers per split
- samples per split
- per-client support
- accent skew
- emotion skew
- speaker overlap = 0

## Products & Deliverables

- `docs/research/B5_SPEC.md`
- `splits/global_split_v*.csv`
- `splits/federated_main_v*.csv`
- `splits/federated_stress_v*.csv`
- leakage tests
- `doc.md` update

## Definition of Done

- Zero speaker leakage.
- Fixed test split.
- Main + stress federated partitions exist.
- Split versions are persisted.

---

# 9. B6 — Modelling

## Objective

Train the 8 centralized models, compare ResNet-18 vs MobileNetV3 for FL suitability, and implement FedECAI.

## Technical Details

### Centralized models

1. SVM — MFCC/eGeMAPS
2. XGBoost — MFCC/eGeMAPS
3. CNN 2D — log-Mel
4. ResNet-18 — 3-channel Mel
5. MobileNetV3 — 3-channel Mel
6. ViT — 3-channel Mel
7. Wav2Vec2 — waveform
8. Pitch-Fusion — waveform + pitch

### Federated backbone selection

Primary default:

- ResNet-18

Efficiency fallback:

- MobileNetV3

Selection criteria:

- Overall Macro-F1
- Worst-accent Macro-F1
- latency
- model size
- training stability
- FL communication burden

### FedECAI architecture

```text
3-channel Mel
    |
Encoder: ResNet-18 or MobileNetV3
    |
Embedding
   / \
  /   \
Emotion Head   GRL
   |            |
4 classes   Accent Discriminator
                 ^
                 |
          Emotion Condition
```

### Required experiment ladder

- E1: backbone baseline
- E2: backbone + unconditional GRL
- E3: backbone + emotion-conditioned GRL
- E4: FedAvg
- E5: FedProx
- E6: FedAvg + unconditional GRL
- E7: FedECAI

## Metrics

For 8-model benchmark:

- Overall Macro-F1
- Worst-accent Macro-F1
- latency
- model size
- parameter count
- training time

For adversarial/federated:

- Emotion Macro-F1
- accent-probe Macro-F1
- accent gap
- client metrics
- convergence metrics

## Products & Deliverables

- `docs/research/B6_SPEC.md`
- `src/models/`
- centralized training scripts
- federated scripts
- configs for all models
- predictions/checkpoints
- model cards
- `doc.md` update
- experiment registry update

## Definition of Done

- All 8 models run.
- Default results exist before tuning.
- ResNet/MobileNet 3-channel Mel inputs are consistent.
- E1–E7 run end-to-end.
- Backbone decision is evidence-based.
- Experiments are reproducible.

---

# 10. B7 — Evaluation and Pareto Analysis

## Objective

Evaluate fairly, answer RQs with evidence, perform 8-model Pareto analysis, and complete error/statistical analysis.

## Technical Details

### Emotion metrics

- Macro-F1
- UAR
- Weighted-F1
- Accuracy
- per-class metrics
- confusion matrix

### Accent metrics

- North Macro-F1
- Central Macro-F1
- South Macro-F1
- Worst-accent Macro-F1
- Best-accent Macro-F1
- accent gap
- accent std

### RQ3 mandatory Pareto

Four primary dimensions:

1. Overall Macro-F1 ↑
2. Worst-accent Macro-F1 ↑
3. Inference Latency (ms) ↓
4. Model Size (MB) ↓

All 8 models must be included.

Latency protocol must state:

- hardware
- batch size
- warm-up
- timed runs
- precision
- preprocessing inclusion/exclusion

Model-size definition must be consistent.

### Accent invariance

- accent-probe Macro-F1
- accent-probe UAR
- majority baseline
- permutation baseline

### Federated metrics

- Global Macro-F1
- Mean-client Macro-F1
- Median-client Macro-F1
- Worst-client Macro-F1
- per-accent metrics
- rounds
- rounds-to-target
- estimated communication cost
- convergence stability

### Statistics

For key experiments:

- at least 3 seeds where feasible;
- mean ± std;
- speaker-bootstrap 95% CI;
- effect sizes where appropriate.

## Metrics

All evaluation metrics above plus Pareto-front membership.

## Products & Deliverables

- `docs/research/B7_SPEC.md`
- result CSVs
- Pareto table/plots
- confusion matrices
- error analysis
- `doc.md` update

## Definition of Done

- Same test set across all 8 models.
- Predictions are saved.
- Per-accent metrics exist.
- Four-axis Pareto analysis exists.
- Latency/size protocol is documented.
- Error analysis exists.

---

# 11. B8 — Tuning and Cross-Validation

## Objective

Tune fairly, analyze `lambda_adv`, apply utility budget, and freeze configs before final test.

## Technical Details

### Tuning

Classical models:

- grouped CV

Deep models:

- development split/grouped protocol

Federated models:

- development client partitions

### Lambda grid

Initial:

- 0.00
- 0.05
- 0.10
- 0.25
- 0.50
- 1.00

### Utility budget

Initial planning threshold:

- Overall Macro-F1 degradation no greater than 0.01–0.02 absolute versus matched baseline.

Prefer configurations where:

- probe score decreases;
- worst-accent improves and/or gap shrinks;
- utility budget is respected.

## Metrics

- grouped-CV mean/std
- validation Macro-F1
- lambda trade-off
- utility-budget violations
- tuning runtime
- failed trials

## Products & Deliverables

- `docs/research/B8_SPEC.md`
- tuning configs
- trial registry
- lambda ablation
- locked configs
- `doc.md` update

## Definition of Done

- Search spaces documented.
- Failed trials logged.
- Lambda ablation complete.
- Utility budget enforced.
- Final configs frozen before final test.

---

# 12. B9 — Pipeline, API, Cloud Services, Application

## Objective

Deploy a reproducible model-serving pipeline with FastAPI + Docker + Cloud Run, Pub/Sub alerts, Dialogflow Q&A, and demo UI.

## Technical Details

### Required stack

- FastAPI
- Docker
- GCP Cloud Run
- Pub/Sub
- Dialogflow
- Secret Manager/environment variables
- Streamlit or Dash

### API

Required endpoints:

- `/health`
- `/predict`

Optional:

- `/model-info`

Prediction response includes:

- predicted emotion
- 4 probabilities
- confidence
- abstain flag
- model version
- processing time
- error field

### Pub/Sub

Used for technical alerts/events:

- endpoint failure
- repeated invalid input
- low-confidence event if desired

Alert target:

- webhook and/or email

### Dialogflow

Minimum intents:

1. supported emotions;
2. how to upload WAV and interpret result.

No medical/psychological advice.

### Security

- no hard-coded secrets
- input validation
- CORS
- rate limiting where practical
- do not log raw audio
- do not retain audio unnecessarily

## Metrics

- API latency
- success/error rate
- alert delivery success
- Dialogflow demo success
- cost estimate

## Products & Deliverables

- `docs/research/B9_SPEC.md`
- FastAPI service
- Dockerfile
- deployment config
- Pub/Sub integration
- webhook/email demo
- Dialogflow config/docs
- dashboard/app
- architecture diagram
- deployment README
- `doc.md` update

## Definition of Done

- `/predict` works on deployed service.
- App calls real endpoint.
- Pub/Sub alert path works.
- Dialogflow has 2+ working intents.
- Architecture diagram exists.
- Security controls documented.

---

# 13. B10 — Final Report, Conclusions, Reproducibility, AI Reflection

## Objective

Answer RQ1–RQ6 with real evidence, produce the final report, complete reproducibility docs, and preserve the audit trail.

## Technical Details

### RQ traceability

Each RQ must include:

- question
- baseline
- hypothesis
- method
- experiment IDs
- evidence
- result
- conclusion
- limitation
- owner

### Final Report — exactly 7 top-level sections

1. Abstract
2. Introduction
3. Methodology
4. Results
5. Discussion
6. Conclusion and Future Work
7. References

No extra top-level sections.

Dataset, Experimental Setup, Service Architecture, and Limitations must be subsections under the seven main sections.

### Required Results coverage

- 8-model benchmark
- per-accent results
- worst-accent results
- latency
- model size
- RQ3 Pareto frontier
- accent-probe results
- conditional vs unconditional GRL
- FedAvg/FedProx/FedECAI comparison
- lambda ablation
- convergence results

### `doc.md`

Human-readable only.

Must include:

1. Project overview
2. Dataset
3. Research questions
4. Methodology
5. Eight centralized models
6. FedECAI architecture
7. Split/client design
8. Metrics
9. Main results
10. Pareto trade-off
11. Cloud architecture
12. Reproduction guide
13. Limitations
14. Current status
15. Links to detailed docs/logs

### AI Audit Log

Record:

- prompt
- output summary
- use/modify/reject decision
- human reasoning
- verification evidence
- resulting artifact
- hallucination/error if any
- final owner

## Metrics

- 6/6 RQs answered or marked inconclusive
- 7/7 report top-level sections
- 0 fabricated results
- 0 unresolved placeholders
- 0 leaked credentials

## Products & Deliverables

- `docs/research/B10_SPEC.md`
- final paper source
- final PDF
- final figures/tables
- `doc.md`
- project log
- decision log
- experiment registry
- AI audit log
- reproducibility README

## Definition of Done

- All RQs are evidence-backed.
- Final report has exactly 7 top-level sections.
- No fake values remain.
- Demo/services work.
- `doc.md` accurately reflects implementation.
- Audit trail is complete.

---

# 14. Required Compact Context File

The repository should contain a compact file:

`PROJECT_RULES.md`

This file is the only project-wide context Codex should read for every task.

It should contain:

- project identity;
- speaker-disjoint rule;
- 8 centralized models;
- 4 Pareto dimensions;
- federated backbone policy;
- no-fabrication rule;
- documentation/log-update rule;
- map from B-step to spec file.

Keep it approximately 100–200 lines where possible.

---

# 15. B-Step Spec Files

Each detailed step specification should live in a separate file:

```text
docs/research/
├── B1_SPEC.md
├── B2_SPEC.md
├── B3_SPEC.md
├── B4_SPEC.md
├── B5_SPEC.md
├── B6_SPEC.md
├── B7_SPEC.md
├── B8_SPEC.md
├── B9_SPEC.md
└── B10_SPEC.md
```

Each file should contain only:

- Objective
- Technical Details
- Metrics
- Products & Deliverables
- Definition of Done

Avoid repeating full project-wide rules inside every spec.

---

# 16. Human Documentation vs AI Context

## `doc.md`

For human readers.

Do not optimize for AI context.

Should explain:

- what the project is;
- what has been implemented;
- current results;
- architecture;
- limitations;
- how to reproduce.

## `PLAN_FEDECAI.md`

Master source of truth.

Read only when needed.

## `PROJECT_RULES.md`

Compact always-on context.

## `B*_SPEC.md`

Task-level context.

## `logs/*`

Historical evidence.

---

# 17. Logging Policy

Recommended files:

```text
logs/
├── project_log.md
├── decision_log.md
├── experiment_registry.csv
└── ai_audit_log.md
```

Do not require Codex to read the full history every time.

Recommended policy:

- read only recent relevant project-log entries;
- search decision log for the specific topic;
- filter experiment registry by model/RQ/step;
- do not load entire AI audit log unless specifically needed.

---

# 18. Experiment Registry Minimum Schema

```text
experiment_id
timestamp
step
rq
model
variant
dataset_version
split_version
client_partition_version
feature_config
model_config
seed
git_commit
hardware
train_time_sec
inference_latency_ms
model_size_mb
overall_macro_f1
worst_accent_macro_f1
accent_gap
accent_probe_macro_f1
mean_client_macro_f1
worst_client_macro_f1
status
checkpoint_path
prediction_path
notes
```

Use blank/null where not applicable.

Never invent values.

---

# 19. Codex Task Workflow

## Before task

Codex should:

1. read `PROJECT_RULES.md`;
2. identify active B-step;
3. read only the relevant `B*_SPEC.md`;
4. inspect existing code/config;
5. read only relevant log entries if needed.

## During task

Codex should:

- follow config-driven design;
- avoid protocol drift;
- preserve split semantics;
- avoid secrets/hard-coded paths;
- save predictions/metrics needed downstream.

## After task

Update only relevant artifacts:

- code/config;
- relevant `docs/research/B*.md`;
- `doc.md` if human-visible state changed;
- project log;
- decision log if architecture/protocol changed;
- experiment registry if an experiment ran;
- AI audit log if required.

Do not rewrite unrelated docs.

---

# 20. Change-Control Rule

Any change to these requires updating this master plan:

- dataset version;
- emotion labels;
- accent grouping;
- speaker-disjoint policy;
- 8-model benchmark list;
- 3-channel Mel strategy;
- federated backbone policy;
- federated client partition policy;
- primary metrics;
- utility budget;
- Pareto dimensions;
- deployment stack;
- seven-section report structure.

No silent protocol drift.

---

# 21. Project Completion Gates

## Data gate

- exact dataset stats computed;
- speaker-disjoint split verified;
- no duplicate leakage.

## Benchmark gate

- all 8 models evaluated;
- same test set;
- predictions saved;
- latency measured;
- model size measured.

## Pareto gate

All 4 dimensions available:

- Overall Macro-F1
- Worst-accent Macro-F1
- latency
- model size

## FedECAI gate

- ResNet-18 or MobileNetV3 selected with evidence;
- baseline;
- unconditional GRL;
- conditional GRL;
- FedAvg;
- FedProx;
- FedECAI.

## Deployment gate

- FastAPI works;
- Docker builds;
- Cloud Run deploys;
- Pub/Sub alerts work;
- Dialogflow Q&A works.

## Documentation gate

- `doc.md` complete;
- B1–B10 docs complete;
- logs complete;
- report complete.

---

# 22. Definition of Project Completion

The project is complete only when:

1. ViSEC pipeline is reproducible;
2. speaker-disjoint evaluation is verified;
3. all 8 centralized models are benchmarked;
4. four-axis Pareto analysis is complete;
5. federated backbone is evidence-selected;
6. FedECAI experiments are complete;
7. RQ1–RQ6 are answered or explicitly inconclusive;
8. deployment architecture works;
9. final report has exactly 7 main sections;
10. `doc.md` accurately reflects implemented reality;
11. logs preserve the evidence trail;
12. every reported number is traceable to experiment/config/split/seed/predictions.
