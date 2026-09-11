# B6 Modelling Protocol

## Scope

B6 trains the five planned emotion classifiers under the frozen
`speaker_disjoint_v2` protocol. The first consolidated runner covers the SVM
and XGBoost baselines on the 174 fixed handcrafted features. CNN 2D,
Wav2Vec2 and pitch-fusion adaptations use the same data and metric contracts
in the GPU B6 runner. The current B4 artifacts contain pooled Wav2Vec2
embeddings rather than waveform hidden-state sequences, so these are honestly
reported as a frozen-embedding MLP and gated pitch-summary fusion, not as
end-to-end fine-tuning or an exact paper reproduction. Their report display
names are therefore **Wav2Vec2 Frozen Embedding MLP** and **Pitch Summary Gated
Fusion Ablation**. The stable experiment IDs are retained only for artifact
compatibility.

The separate `run_b6_pitch_fusion_paper.py` path restores the paper's
raw-waveform Wav2Vec2 sequence, Conv1D pitch-sequence encoder, bidirectional
cross-attention, self-attention, projection, and classification design. It is
labelled paper-aligned rather than exact: it uses the already validated B4 pYIN
contour instead of the official script's Kaldi-derived sequence, applies
padding-aware pooling, and preserves the fixed B5 speaker-disjoint protocol.

## Leakage boundary

- Model and preprocessing selection use only the 3,681 training samples.
- The supplied `cv_fold` labels define speaker-grouped cross-validation.
- Standardization is fitted inside each training fold through a scikit-learn
  pipeline.
- SVM probabilities are calibrated with the same speaker-grouped folds after
  hyperparameter selection; row-level calibration folds are not used. The
  classifier's maximum-margin decision supplies the predicted label, while the
  calibrator supplies confidence values only.
- B6 reports one validation result after selection.
- The 665 test samples remain reserved for final B7 evaluation.
- No B4 exploratory PCA or full-dataset standardization is reused.

## Inputs and outputs

The classical runner joins the 4,992-row B4 feature table to the enriched B5
manifest by unique `sample_id` and requires exactly 174 finite handcrafted
features. It stores deployable model files under ignored `artifacts/b6/` and
publishes small CV, validation-prediction, metric, timing, and lineage reports
under `reports/tables/b6/`.

## GPU runtime

The workstation has an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM, but
the original `.venv` intentionally contains a CPU-only PyTorch build used for
B4 reproduction. Deep B6 models use a separate `.venv-gpu` created by
`scripts/setup_b6_gpu.ps1`. The isolated environment uses the official PyTorch
CUDA 12.6 wheel and leaves the verified CPU environment unchanged.

## Metrics

The primary selection metric is Macro-F1. Reports also include accuracy, UAR,
per-class F1, per-accent Macro-F1/UAR, worst-accent Macro-F1, and the maximum
minus minimum accent gap. Each parameter candidate retains Macro-F1 and UAR for
every individual fold, not only mean and standard deviation. The dominant
speaker-0 fold is retained as defined in B5, so its per-fold result must be
interpreted explicitly.

The neural runner evaluates all 30 epochs in each grouped-CV fold because the
small speaker groups produce noisy validation curves. It retains the
highest-Macro-F1 checkpoint from each fold and chooses the final training
length as the median best epoch across the five folds. The final model is not
selected on the held-out validation partition. The runner checkpoints after
every fold so an interrupted run can resume without repeating completed folds.

## E1-E7 extension ladder

The extension runner uses the log-Mel CNN as a shared encoder. E2 adds an
unconditional Gradient Reversal accent discriminator; E3 conditions that
discriminator on ground-truth emotion during training. E4 and E5 implement
sample-weighted FedAvg and FedProx. E6 federates unconditional GRL, while E7 is
the emotion-conditioned federated FedECAI experiment. E4-E7 run separately on
the versioned B5 main mixed-accent and stress accent-domain scenarios. The
external test partition remains unloaded throughout B6.
