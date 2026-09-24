# DAP391m Pitch Fusion Multi Prosody

This repository implements the approved cleaned-data ViSEC comparison through
M5. M5 uses the human-approved lightweight residual dilated CNN1D extension.

## Fixed research comparison

- M1: SVM-RBF over the existing 174 handcrafted summary features.
- M2: raw waveform to Vietnamese Wav2Vec2 to the shared emotion classifier.
- M3: pitch-only Pitch-Fusion baseline using log-F0.
- M4: the same M3 architecture with five prosody channels: log-F0, voicing
  confidence, log-energy, delta log-F0, and delta log-energy.
- M5: the same five-channel input, Wav2Vec2 branch, fusion, classifier, and
  training policy as M4, with three lightweight residual CNN1D blocks using
  dilations 1, 2, and 4 after the M4 pitch encoder.
- M3 to M4 changes only the prosody input channel count. The temporal Conv1D
  encoder, Wav2Vec2 branch, bidirectional cross-attention fusion, classifier,
  split, seed, loss, optimizer, batch size, and epoch policy stay fixed.
- Spectrogram and CNN2D are excluded from the main Pitch-Fusion comparison.

## Data and leakage contract

- Use `data/manifests/split_manifest_v2_enriched.csv` for every model.
- Train and validation speakers must be disjoint.
- Test is reserved and is never used for M1-M4 selection or reporting.
- Multi-prosody features are extracted deterministically for every eligible
  sample, but channel normalization is fitted from training frames only.
- M1 uses `StandardScaler` inside each grouped training fold and fits the final
  scaler only on the full training partition.

## Environment

Create the environment yourself; no setup command is executed automatically:

```powershell
py -3.14 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
```

The default Wav2Vec2 configuration first looks for the existing pinned local
snapshot. Add `--allow-download` only when you intentionally allow Hugging Face
to download the pinned revision.

## Run in evidence order

Generate and validate the B1-B5 foundation package first:

```powershell
.\scripts\run_b1_b5.ps1
```

This writes the data dictionary, quality summary, five SQL result tables, EDA
notebook, six figures, t-SNE projection, interactive dashboard, split/CV
validation, completion report, and a timestamped log. It uses existing frozen
features and does not train a model.

Run M1 first:

```powershell
.\scripts\run_m1_m4.ps1 -Model m1
```

The wrapper first creates or validates the 174-feature M1 table using the new
`pitchfusion` extractor, then runs the fixed SVM-RBF baseline.

Run M2:

```powershell
.\scripts\run_m1_m4.ps1 -Model m2
```

M3 and M4 require the multi-prosody preparation step. The wrapper runs it
automatically and safely resumes existing artifacts:

```powershell
.\scripts\run_paper_reproduction.ps1
.\scripts\run_m4_clean.ps1 -PreflightOnly
.\scripts\run_m4_clean.ps1
.\scripts\run_m5_clean.ps1 -PreflightOnly
.\scripts\run_m5_clean.ps1
```

To execute the complete approved sequence:

```powershell
.\scripts\run_m1_m4.ps1 -Model all
```

For an online first run, append `-AllowDownload`. Use `-Force` only when you
intend to replace completed model outputs.

## Direct commands

```powershell
.\.venv-gpu\Scripts\python.exe src\pitchfusion\features\prepare_m1_features.py
.\.venv-gpu\Scripts\python.exe src\pitchfusion\models\run_m1_svm.py --preflight-only

.\.venv-gpu\Scripts\python.exe src\pitchfusion\features\prepare_multi_prosody.py

.\.venv-gpu\Scripts\python.exe src\pitchfusion\models\run_m2_m4.py `
  --models m2_wav2vec2 --device cuda --preflight-only
```

Remove `--preflight-only` when you want to train. M2-M4 checkpoints are saved
atomically after every epoch and a repeated command resumes model, optimizer,
mixed-precision scaler, best weights, history, runtime, and random states. M1
saves a resumable checkpoint after every grouped-CV fold, final fit, and
probability-calibration stage.

Every wrapper run writes a complete PowerShell/Python transcript to
`logs/m1_m4/run_<model>_<timestamp>.log`, including errors and checkpoint paths.
After every M2-M4 epoch, the console and log show the model name, epoch progress,
train/validation loss, Macro-F1, UAR, best epoch, learning rate, elapsed time,
estimated remaining time, and the checkpoint path.

The runner stores Numba/Librosa compilation cache in
`data/interim/numba_cache/`. The first pitch extraction may spend several
seconds compiling; later samples reuse that cache.

Static and unit checks can be run separately without starting training:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest
```

## Outputs

- Model artifacts: `artifacts/m1_m4/m1/` and `artifacts/m1_m4/pitch_fusion/`
- Metrics and predictions: `reports/m1_m4/`
- Raw multi-prosody artifacts and train-only normalization:
  `data/interim/features/multi_prosody_v1/`

The code reports Macro-F1 and UAR as the main metrics and also records accuracy,
per-class F1, configuration hashes, input hashes, runtime, and test predictions.
M5 writes to separate artifact, report, prediction, and log paths and safely
resumes from its own epoch checkpoint.

## Paper reproduction before the M3-M5 comparison

The published ViSEC result (UA 72.72%, WA 71.90%) used an utterance-level
stratified split rather than the project's speaker-disjoint split. A separate
reproduction runner is provided so this check cannot overwrite M1-M4 evidence.
Per the project decision, it uses the 4,992 utterances accepted after cleaning,
then creates a deterministic 80/10/10 emotion-stratified split with speaker
overlap allowed. It therefore measures a cleaned-data reproduction and is not
claimed as an exact historical reproduction of the paper's 5,280-row split.

The pitch feature is extracted by Kaldi's official
`compute-kaldi-pitch-feats` executable. On Windows, install Miniforge and create
the repository-local Kaldi environment with:

```powershell
.\scripts\setup_paper_reproduction.ps1 -InstallMiniforge
```

If Conda/Miniforge is already installed, omit `-InstallMiniforge`. The Kaldi
binary is isolated under `.kaldi-win`; model training continues to use the
existing `.venv-gpu` environment.

Run a one-sample model and Kaldi-pitch check first, then start the full run:

```powershell
.\scripts\run_paper_reproduction.ps1 -PreflightOnly
.\scripts\run_paper_reproduction.ps1
```

Pitch extraction is cached and resumable. Training saves a last checkpoint
after every epoch, keeps the best validation-UA checkpoint, prints train loss,
validation loss, UA, WA, Macro-F1, learning rate, elapsed time and ETA after
every epoch, and evaluates the test split once after model selection. Complete
console output is duplicated to `logs/paper_reproduction/`. Final results are
written to `reports/paper_reproduction/m3_paper_reproduction_report.json` and
the independent model artifact is stored under
`artifacts/paper_reproduction/m3_pitch_fusion/`.

M4 uses the same cleaned split and training contract as this M3. Its five
channels are Kaldi log-F0, Kaldi NCCF, frame log-energy, delta log-F0, and
delta log-energy. Train-only channel normalization, feature cache, checkpoints,
logs, and final test reporting are handled by `run_m4_clean.ps1`. The older
speaker-disjoint M3/M4 path is retained only as legacy evidence and its wrapper
now refuses to launch it accidentally.
