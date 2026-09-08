# B1 — Problem Understanding and Research Questions

Status: **FROZEN — team-approved on 2026-09-08**  
Authoritative planning source: Google Docs tab `plan gom rq`  
Project: **PVC-FedOrion**

## 1. Working title

**PVC-FedOrion: Label-Aware Paired-View Federated Learning for View-Robust
Vietnamese Speech Emotion Recognition**

Vietnamese working title:

**PVC-FedOrion: Học liên kết kết hợp tính nhất quán có trọng số giữa các phiên
bản âm thanh cho nhận diện cảm xúc tiếng nói tiếng Việt**

The title intentionally excludes the terms `privacy-preserving`, `regional
Vietnamese SER`, `medical diagnosis`, and `state of the art`. The current study
does not implement formal privacy protection, does not have regional metadata,
does not predict clinical states, and has not established a state-of-the-art
comparison under a common protocol.

## 2. Problem statement

### 2.1 Report-ready English draft

Vietnamese Speech Emotion Recognition (SER) remains difficult under limited
data, speaker variability, background interference, and distribution shift. The
VESC corpus provides 904 audio files across five emotion labels—Angry, Anxiety,
Happy, Neutral, and Sad—and includes a subset of utterances represented by both
an original recording and a processed `_vocals` version. These two files are not
independent observations: they share the same linguistic content, speaker, and
emotion label. Consequently, a file-level split can place different views of the
same utterance in training and testing, while an unconstrained split can also
place the same speaker in both partitions. Both mechanisms can produce an
optimistic estimate of unseen-speaker performance.

The paired views also create an opportunity. A model can be trained to produce
stable emotion predictions across the original and processed recordings.
However, the processed view must not be treated as clean ground truth because
its generation pipeline is not fully documented and processing may remove
emotion cues or introduce artifacts. Moreover, only part of VESC is paired, and
the availability of pairs differs across emotions, sources, speakers, and future
federated clients.

This project therefore develops PVC-FedOrion, a leakage-aware federated SER
framework built around an OrionNet-derived log-Mel backbone. The study first
constructs a speaker-disjoint and pair-safe evaluation protocol. It then
compares centralized deep-learning architectures under a common acoustic
front-end, establishes federated baselines under speaker-level non-IID data,
and evaluates a Label-Aware Paired-View Consistency objective that weights each
pair according to how strongly both views support the observed emotion label.
The method is evaluated using global recognition performance, unseen-speaker
generalization, paired-view stability, client disparity, convergence, and
computation/communication cost.

### 2.2 Giải thích ngắn bằng tiếng Việt

Đề tài không đơn thuần hỏi “model nào có Accuracy cao nhất”. Trọng tâm là ba
vấn đề có quan hệ với nhau:

1. Kết quả trên VESC có thể bị thổi phồng nếu cùng speaker hoặc hai view của
   cùng một phát ngôn xuất hiện ở train và test.
2. Raw và `_vocals` có cùng nội dung nhưng model có thể dự đoán không nhất quán.
3. Khi phân phối speaker vào các virtual clients, emotion và paired-data ratio
   trở nên non-IID, làm chênh lệch hiệu năng giữa các client.

## 3. Scope

### 3.1 In scope

- VESC with five labels: Angry, Anxiety, Happy, Neutral, and Sad.
- Original/raw and processed/denoised `_vocals` views.
- Speaker-disjoint and pair-safe evaluation.
- A common mono 16 kHz log-Mel front-end.
- Five centralized architectures: OrionNet-LM, Transformer Encoder,
  CNN–BiLSTM–Attention, STACN, and ESERNet.
- OrionNet-LM as the common federated backbone.
- Centralized, Local-only, FedAvg, FedProx, and FedBN baselines.
- Raw-only CE, utterance-balanced raw+processed CE, fixed PVC, and Label-Aware
  PVC ablations.
- Main and heterogeneity-stress virtual-client settings.
- Three seeds, grouped development cross-validation, and a locked final test.
- AWS-based model serving, alert, Q&A, database, and decision-support demo.

### 3.2 Out of scope / future work

- Regional North/Central/South claims without verified region metadata.
- Formal privacy guarantees, differential privacy, or secure aggregation.
- Medical or mental-health diagnosis.
- Cross-corpus generalization and real teleconsultation deployment.
- Claims that virtual clients represent real institutions or geographic regions.
- Claims that `_vocals` is artifact-free clean ground truth.

## 4. Research gap

Prior work covers federated learning, federated SER, non-IID optimization,
multiview learning, clean/noisy consistency, and advanced SER architectures.
Therefore, the project does not claim to be the first federated SER study.

The narrower research gap is the joint treatment of:

- naturally paired original and processed views rather than two artificial
  augmentations;
- incomplete and emotion-dependent pair availability;
- possible artifacts in the processed view;
- speaker-disjoint and pair-safe evaluation;
- speaker-level non-IID clients with different label and pair distributions;
- label-aware consistency weighting in this low-resource Vietnamese SER setting.

This gap remains a **working gap**, not a proven novelty claim, until the
systematic related-work search is completed and documented.

## 5. Scientific contributions

### C1 — Leakage-aware and utterance-aware VESC protocol

The project will create and release a reproducible evaluation protocol with:

- speaker-disjoint partitions;
- pair-safe grouping at the underlying-utterance level;
- augmentation after splitting only;
- normalization fitted on training data only;
- a newly initialized model and optimizer per fold;
- a final test locked before tuning;
- utterance-balanced loss and aggregation weights;
- explicit comparison with an original-like file-level protocol.

C1 is supported only if the study quantifies leakage and measures protocol
sensitivity. Merely implementing a grouped split is an engineering step, not a
complete scientific contribution.

### C2 — Label-Aware Paired-View Consistency

The proposed objective assigns a larger consistency weight when both views
support the observed label and a smaller weight when either view is unreliable.
It is compared with raw-only training, adding processed data without
consistency, and fixed-weight consistency. This separation is necessary to show
whether an improvement comes from more supervised samples or from the proposed
mechanism.

### C3 — Federated non-IID evaluation

The proposed method will be evaluated across virtual clients that contain
disjoint speakers and differ in emotion distribution, sample count, source
distribution, and paired-data availability. Evidence includes global metrics,
client disparity, convergence, and computation/communication cost.

Five-model comparison, use of standard FL baselines, SQL, cloud services, and
the application are important course deliverables but are not presented as the
three main scientific contributions.

## 6. Research questions and hypotheses

### Basic RQ1 — Dataset structure and leakage

**Question.** What forms of imbalance, confounding, and data leakage are induced
by the speaker, emotion, source, and raw–processed pair structure of VESC?

**B-H1.** File-level evaluation produces optimistic estimates because speakers
or paired views overlap across partitions.

**Required evidence.** Versioned manifest, pair manifest, speaker overlap,
cross-partition pair count, source–emotion and speaker–emotion tables, pair
quality statistics, Sankey diagram, and heatmaps.

### Basic RQ2 — Deep-learning architecture comparison

**Question.** Under the same pair-safe, speaker-disjoint protocol and common
acoustic front-end, how do five deep-learning architectures differ in Macro-F1,
UAR, and computational cost?

**B-H2.** More complex architectures may improve recognition performance, while
OrionNet-LM may offer a better parameter/latency trade-off for repeated local
training.

**Required evidence.** Grouped cross-validation, Macro-F1, UAR, Accuracy,
parameter count, training time, inference latency, and pre/post-tuning results.

### Basic RQ3 — Evaluation-protocol sensitivity

**Question.** How does OrionNet-LM performance change when moving from an
original-like file-level protocol to a speaker-disjoint and pair-safe protocol?

**B-H3.** The clean protocol yields lower scores but a more credible estimate of
unseen-speaker generalization.

**Required evidence.** Original-like and clean results, leakage counts,
Macro-F1/UAR differences, and a confidence interval for the difference. The
leakage-prone result is never reported as the main model performance.

### Extension RQ1 — Federated baselines

**Question.** Under pair-safe, speaker-disjoint, speaker-non-IID VESC data, how
do Centralized, Local-only, FedAvg, FedProx, and FedBN differ in global and
client-level performance?

**E-H1.** At least one of FedProx or FedBN improves client-level stability over
FedAvg, but no method is assumed to win before measurement.

**Required evidence.** Raw Macro-F1/UAR, Accuracy, per-class recall, mean-client
Macro-F1, client standard deviation, confusion matrices, and convergence curves.

### Extension RQ2 — Proposed Label-Aware PVC

**Question.** Does Label-Aware Paired-View Consistency improve original-audio
emotion recognition and original–processed prediction stability compared with
supervised FL, utterance-balanced raw+processed training, and fixed PVC?

**E-H2.** Label-Aware PVC improves raw Macro-F1/UAR, reduces paired-view
divergence, outperforms fixed PVC when processed views contain artifacts, and
does not materially degrade singleton performance.

**Required evidence.** Four mandatory ablations, raw and processed metrics,
paired/singleton subset performance, agreement, Jensen–Shannon divergence, and
both-correct/one-correct/both-wrong rates.

### Extension RQ3 — Heterogeneity and cost

**Question.** How does the benefit of PVC-FedOrion change with client non-IID
level and paired-data availability, and what local-computation or communication
cost does it introduce?

**E-H3.** The benefit is larger when clients have adequate paired data. PVC does
not increase the transmitted model-update size but can increase local
computation because paired examples require two forward passes.

**Required evidence.** Global/mean-client Macro-F1, client standard deviation,
performance versus pair ratio and heterogeneity, local time per round, rounds to
a predeclared target, and cumulative transmitted bytes.

## 7. Decision rules

The team must define thresholds before final experiments. Initial rules are:

1. A contribution is not supported by Accuracy alone; Macro-F1 and UAR are
   primary recognition metrics.
2. C1 requires non-zero measured leakage and a quantified protocol effect.
3. C2 requires Label-Aware PVC to be compared with all four mandatory ablations.
4. A PVC gain is meaningful only when raw performance and pair stability are
   considered together and singleton performance does not collapse.
5. C3 requires both a main split and a stronger heterogeneity split with measured
   client-distribution statistics.
6. FedBN is not assigned a single global-BN evaluation implicitly; its test-time
   BN policy must be specified before the baseline is run.
7. Neutral is excluded from class-specific pair claims because it has no paired
   families in the supplied corpus; overall pair results must disclose support.

Numerical minimum-effect thresholds will be selected during B5/B7 using the
development protocol, not by looking at the locked final test.

## 8. Claim-safety rules

Use:

- `federated learning simulation`;
- `training without exchanging raw audio in the simulated protocol`;
- `processed/denoised view`;
- `emotion classification`;
- `working research gap` or `we propose and evaluate`.

Do not use without additional evidence:

- `privacy-preserving` or `guarantees privacy`;
- `clean ground truth` or verified `source-separated vocals`;
- `medical diagnosis`, `depression detection`, or `anxiety disorder detection`;
- `regional Vietnamese SER`;
- `state of the art`, `first`, or `novel` as an absolute claim.

## 9. B1 completion state

B1 was frozen by the project owner on 2026-09-08 after approval of the six RQs,
three hypotheses, three contributions, and the in-scope/out-of-scope boundary.
This version is the research contract for B2–B10.

Reopening B1 requires a dated change record that states the new evidence or
instructor request, the exact wording changed, and the impact on downstream
experiments. Model results alone are not a valid reason to rewrite an RQ or
hypothesis retrospectively.
