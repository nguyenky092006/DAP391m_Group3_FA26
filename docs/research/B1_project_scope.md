# B1 Project Scope and Problem Statement

## Document status

- Project: Emotion-Conditioned Accent-Invariant Learning for Federated Vietnamese Speech Emotion Recognition
- Dataset: ViSEC (Vietnamese Speech Emotion Corpus)
- Planning baseline: the submitted Project Planning document
- Current phase: B1 - Business Understanding and Data Collection
- Important: this document defines the planned scope. Dataset statistics and model performance are not treated as verified results until they are produced by code and recorded in later B1 artifacts.

## Problem statement

Vietnamese speech emotion recognition models may perform differently across Northern, Central, and Southern accents. A model can also appear to perform well if recordings from the same speaker occur in both training and evaluation data. The project therefore needs a speaker-disjoint evaluation protocol and accent-level analysis to determine whether a model works consistently for unseen speakers rather than only achieving a strong overall score.

The project will use ViSEC to recognize four emotions: Angry, Happy, Neutral, and Sad. Five models will be compared under the same data split: SVM, XGBoost, CNN 2D, Wav2Vec2, and Pitch-Fusion. The comparison will consider overall emotion-recognition performance, performance for the weakest accent group, and computational cost.

## Project objective

Build and evaluate a reproducible Vietnamese speech emotion recognition pipeline that:

1. prevents speaker leakage between training, validation, and test data;
2. compares five models using the same finalized split;
3. measures performance separately for each accent group;
4. examines how much accent information remains in different audio representations; and
5. identifies a model that provides a practical balance between recognition performance, accent robustness, and computational cost.

## Intended users and outputs

- Researchers and project reviewers can use the experiment results to compare model behavior across accent groups.
- Application users can upload a WAV file and receive a predicted emotion, class probabilities, and confidence information.
- Accent is used for evaluation and domain-invariance experiments. Accent prediction is not the main function of the application.

## Core research questions

### RQ1

How much accent information remains in each audio representation when testing on unseen speakers?

- Inputs: MFCC/eGeMAPS, log-Mel spectrogram, pitch-related features, and Wav2Vec2 embeddings.
- Secondary target: Northern, Central, and Southern accent labels.
- Main measures: accent-probe Macro-F1, UAR, confusion matrix, majority baseline, permutation-label baseline, and bootstrap 95% confidence intervals.

### RQ2

How much does emotion recognition performance differ among Northern, Central, and Southern accents when testing on unseen speakers?

- Inputs: the audio representation required by each model.
- Target: Angry, Happy, Neutral, and Sad.
- Main measures: overall Macro-F1, per-accent Macro-F1, worst-accent Macro-F1, maximum-minus-minimum accent gap, per-class F1, and bootstrap 95% confidence intervals.

### RQ3

Which of the five models provides the best balance between overall performance, performance for the weakest accent group, and computational cost?

- Compared models: SVM, XGBoost, CNN 2D, Wav2Vec2, and Pitch-Fusion.
- Main measures: overall Macro-F1, worst-accent Macro-F1, training time, inference latency, model size, parameter count, and Pareto analysis.

## Included scope

- Use the official ViSEC dataset as the only dataset in the main experiments.
- Inspect the raw audio and metadata before cleaning or feature extraction.
- Preserve an immutable raw-data layer and store generated metadata in manifests.
- Use speaker-disjoint train, validation, and test partitions.
- Apply data augmentation only to the training partition and record it in configuration files.
- Extract MFCC/eGeMAPS, log-Mel, pitch-related features, and Wav2Vec2 representations as required by the selected models.
- Train and compare the five planned models on the same finalized evaluation protocol.
- Report both overall results and accent-level results.
- Build a Streamlit application and a FastAPI prediction endpoint after the core model pipeline is stable.

## Planned extension after the core baseline

After the five-model baseline is working, the project will investigate unconditional GRL, emotion-conditioned GRL, FedAvg, FedProx, and the proposed FedECAI approach. These experiments must reuse documented partitions and must not replace the core RQ1-RQ3 evaluation.

## Out of scope

- Collecting a new speech corpus.
- Adding another dataset to the main comparison.
- Using accent prediction as the main application output.
- Speaker identification or storing a user's identity.
- Claiming clinical, psychological, or safety-critical emotion assessment.
- Multilingual speech emotion recognition.
- Production-scale deployment or real-time service guarantees.
- Applying augmentation to validation or test data.
- Selecting the final model using test-set results.

## Constraints

- The team has two members and a ten-week implementation schedule.
- Wav2Vec2, Pitch-Fusion, and federated experiments may require substantially more compute than SVM, XGBoost, and CNN 2D.
- The implementation must retain enough time for the application, cloud integration, report, presentation, and individual AI Audit Logs.
- Raw audio, credentials, checkpoints, and large generated artifacts must not be committed to GitHub.

## Planning assumptions that require verification

The Project Planning document reports 5,280 utterances from 147 speakers, four emotion labels, three accent groups, and a CC BY 4.0 license. B1 data collection and audit must verify these statements against the downloaded dataset and its official documentation before they are used as experimental facts.

## Non-negotiable evaluation rules

1. No speaker may appear in more than one of train, validation, and test.
2. The finalized test set must remain unchanged and must not be used for tuning.
3. All five models must use the same finalized split.
4. Preprocessing fitted from data must be fitted on the training partition only.
5. Data augmentation must be applied only to training samples.
6. Accent labels may be used for analysis, probing, GRL, and federated partition design, but they are not the primary application target.
7. Any correction to the submitted assumptions must be supported by an official source or a reproducible data-audit result and recorded in the Audit Log.

## B1.2 acceptance criteria

- The problem, objective, intended users, and application output are stated clearly.
- RQ1-RQ3 match the submitted Project Planning document.
- The core five-model comparison is separated from the later GRL and federated extension.
- Included and excluded scope are explicit.
- Speaker-disjoint evaluation and train-only augmentation are fixed project rules.
- Unverified dataset counts are identified as assumptions rather than reported as measured results.

## Primary references

- ViSEC paper: https://doi.org/10.1109/ICASSP48485.2024.10448373
- Official ViSEC repository: https://github.com/thanhpv2102/ViSEC
- ViSEC dataset card: https://huggingface.co/datasets/hustep-lab/ViSEC

