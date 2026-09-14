# B1 — Problem Definition and Research Scope

## Objective

Define the FedECAI problem, evidence-based research gap, expected contributions, six research questions, six testable hypotheses, foundational literature, scope, ownership, and evidence requirements without claiming unrun results.

## Technical Details

- Dataset: ViSEC; four emotion classes and three accent groups.
- Evaluation unit: speaker-disjoint train/validation/test; the same fixed test set is required for every centralized model.
- RQ1–RQ6 and H1–H6 are defined below in this consolidated specification.
- RQ3 compares exactly eight centralized models: SVM, XGBoost, CNN 2D, ResNet-18 with 3-channel Mel-Spectrogram, MobileNetV3 with 3-channel Mel-Spectrogram, ViT with 3-channel Mel-Spectrogram, Wav2Vec2, and Pitch-Fusion.
- RQ3 uses four primary Pareto dimensions: Overall Macro-F1, Worst-accent Macro-F1, inference latency, and model size.
- RQ5 treats ResNet-18 as the primary federated-backbone candidate and MobileNetV3 as the efficiency alternative. Selection requires performance, latency, size, stability, communication burden, and FL practicality evidence.
- Required method ladder: baseline, unconditional GRL, emotion-conditioned GRL, FedAvg, FedProx, FedAvg with unconditional GRL, and FedECAI.
- Foundational literature covers ViSEC/Pitch-Fusion, DANN/GRL, FedAvg, and FedProx.
- Existing five-model and compact-CNN E1–E7 outputs are preliminary legacy evidence. They do not satisfy the eight-model benchmark or final FedECAI-backbone requirements.
- Ownership is assigned by workstream role because no verified individual assignment is present in the repository. Named-person allocation remains a project-management action, not a scientific requirement or result.

## Metrics

Planning-completeness checks:

- 6/6 research questions defined.
- 6/6 hypotheses defined.
- 6/6 RQ-to-method, metric, evidence, decision-rule, and owner mappings defined.
- Exactly 8 centralized models in RQ3.
- Four mandatory Pareto dimensions in RQ3.
- ResNet-18/MobileNetV3 federated-backbone policy in RQ5.
- Zero claimed experimental values not supported by existing artifacts.
- Zero unsupported “first”, “novel”, privacy-guarantee, or diagnostic claims.

## Consolidated research questions and hypotheses

- **RQ1:** How much accent information remains in MFCC/eGeMAPS, log-Mel,
  pitch-related and Wav2Vec2 representations on unseen speakers?
- **RQ2:** How much does emotion recognition differ among North, Central and
  South accents on unseen speakers?
- **RQ3:** Which of the eight centralized models is Pareto-efficient across
  overall Macro-F1, worst-accent Macro-F1, latency and model size?
- **RQ4:** Does emotion-conditioned adversarial learning improve the
  utility–accent-invariance trade-off relative to no GRL and unconditional GRL?
- **RQ5:** How does FedECAI compare with FedAvg, FedProx and FedAvg plus
  unconditional GRL under the versioned client scenarios?
- **RQ6:** How does `lambda_adv` control utility versus accent invariance?

H1–H6 are preregistered expectations, not findings: measurable residual accent
information; non-zero emotion performance gaps by accent; a non-trivial
four-axis centralized Pareto frontier; conditional GRL competitive with or
better than unconditional GRL within a utility budget; FedECAI improvement in
at least one federated scenario; and at least one non-zero Pareto-efficient
`lambda_adv` setting. None may be claimed without its required experiment.

## Consolidated dataset provenance and dictionary

- Source: ViSEC release used by the ICASSP 2024 Pitch-Fusion work; the pinned
  local Parquet and SHA-256 in `data/manifests/raw_audit.json` are authoritative.
- License recorded by the acquisition source: CC BY 4.0. Attribution remains
  mandatory; raw audio, personal data and large artifacts are not committed.
- Verified raw population: 5,280 rows, 147 released speaker IDs, four emotions
  (`angry`, `happy`, `neutral`, `sad`) and three canonical accents
  (`central`, `north`, `south`).
- Core identity fields are `sample_id`, `source_row_index`, `speaker_id`,
  `audio_sha256`, emotion and accent. Audit fields preserve WAV readability,
  sample rate, channel count, duration, clipping/silence flags and duplicate
  membership. Split/client fields are downstream assignments, never raw facts.
- Missing/unreadable/invalid-label rows cannot enter modelling. Exact duplicate
  waveforms cannot cross splits. Conflicting duplicates are excluded rather
  than relabelled. Speaker ID `0` may be a catch-all; claims are limited to the
  released identifier.

## Products & Deliverables

- This consolidated `docs/research/B1_SPEC.md`.
- `data/manifests/raw_manifest.csv`, `raw_audit.json`, `invalid_audio.csv` and
  `duplicate_report.csv`.
- `doc.md`, `docs/PROJECT_STATUS_AUDIT.md` and `logs/project_log.md`.

## Definition of Done

- Problem statement is explicit.
- Research gap is tied to the foundational literature without an unsupported novelty claim.
- Expected contributions are stated as intended, evidence-dependent outputs.
- RQ1–RQ6 and H1–H6 exist and map one-to-one.
- RQ3 lists exactly the required eight models and four Pareto dimensions.
- RQ5 contains the ResNet-18/MobileNetV3 backbone policy.
- Included scope, out-of-scope boundaries, non-negotiable evaluation rules, and workstream owners are explicit.
- Every RQ maps to a method, metrics, future evidence, a decision rule, and an owner.
- Human-facing documentation reflects that B1 is complete while downstream implementation remains incomplete.
