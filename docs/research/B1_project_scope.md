# B1 Pitch Fusion Multi Prosody Project Scope

## Problem

The project reproduces the robust Pitch-Fusion baseline on ViSEC and tests
whether a richer temporal prosody representation improves Vietnamese speech
emotion recognition. It then stops after M4 until a human reviews the evidence
and approves an M5 architecture.

## Research questions

- RQ1: What performance does the reproduced Pitch-Fusion baseline achieve on
  ViSEC under the fixed project protocol?
- RQ2: Does replacing pitch-only input with five-channel multi-prosody improve
  speech emotion recognition?
- RQ3: After M1-M4 evidence is available, does an approved improved CNN1D offer
  a useful performance and complexity trade-off over the simple CNN1D?

## Approved model sequence

- M1: fixed SVM-RBF classical baseline.
- M2: Wav2Vec2-only.
- M3: original pitch-only Pitch-Fusion baseline.
- M4: M3 with log-F0, voicing confidence, log-energy, delta log-F0, and delta
  log-energy.
- M5: not implemented until a separate human decision.

M3 to M4 measures feature gain. No spectrogram/CNN2D, XGBoost, federated
learning, GRL, or unapproved extra model is part of the main comparison.

## Baseline and related work

- Pham, Ngo, Pham, and Nguyen, *A Robust Pitch-Fusion Model for Speech
  Emotion Recognition in Tonal Languages*, ICASSP 2024, DOI
  `10.1109/ICASSP48485.2024.10448373`. This is the main reproduction baseline.
- Ha Nguyen Manh et al., *Advancing Vietnamese Speech Emotion Recognition:
  The OrionNet Architecture and VESC Corpus*, Proceedings of CSA 2025,
  DOI `10.1007/978-981-95-7896-2_21`. This supplies Vietnamese SER context;
  it does not replace ViSEC or the main Pitch-Fusion baseline.
- Aftab et al., *Light-SERNet: A Lightweight Fully Convolutional Neural
  Network for Speech Emotion Recognition*, ICASSP 2022. This is related work
  for the later human decision about a lightweight improved CNN1D; its
  architecture is not copied into M5 without approval.
