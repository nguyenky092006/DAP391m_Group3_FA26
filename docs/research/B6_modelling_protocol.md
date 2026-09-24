# B6 M1 to M5 Modelling Protocol

## Approved scope

B6 runs five models in evidence order. M1 is a fixed SVM-RBF classical
baseline. M2 is a raw-waveform Wav2Vec2-only classifier. M3 is the pitch-only
Pitch-Fusion baseline. M4 changes only the prosody input from log-F0 to the
approved five-channel representation. After the M1-M4 review, M5 was approved
as the same M4 pipeline plus lightweight residual dilated CNN1D blocks.

## Fair-comparison contract

- Every model uses `speaker_disjoint_v2` and seed 391.
- M3 and M4 share waveform preprocessing, Wav2Vec2, temporal Conv1D encoder,
  bidirectional cross-attention, self-attention, classifier, loss, optimizer,
  batch size, and epoch policy.
- M3 receives one normalized log-F0 channel.
- M4 receives log-F0, voicing confidence, log-energy, delta log-F0, and delta
  log-energy.
- Prosody normalization statistics are fitted on training frames only.
- M4 and M5 reuse the same cached five-channel prosody input. M5 changes only
  the prosody CNN architecture by adding residual bottleneck blocks with
  dilations 1, 2, and 4; it does not initialize from an M4 checkpoint.
- Test samples are reserved for the later final evaluation.
- Spectrogram/CNN2D, XGBoost, federated learning, and GRL are outside this
  experiment.

## M1 policy

M1 uses SVM-RBF with `C=1`, `gamma=scale`, balanced class weights, and a
training-only `StandardScaler`. The configuration is fixed in B6. Existing
speaker-grouped folds provide diagnostic Macro-F1 and UAR; hyperparameter
tuning is deferred to B8.

## Metrics and artifacts

Macro-F1 and UAR are the main metrics. Reports also store accuracy, per-class
F1, accent breakdowns, predictions, runtime, model hashes, config hashes, and
input lineage. Neural checkpoints are written after every epoch for safe
resume. The validation split is used only for B6 reporting, not for fitting
normalizers or selecting SVM hyperparameters.
