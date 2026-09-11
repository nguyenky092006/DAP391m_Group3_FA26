# B4 ViSEC Feature Pilot Diagnostics

## Result

B4.3 passes for the balanced 12-sample pilot. Three reproducible figures were
created and visually reviewed. All 12 accent-emotion cells are present, feature
arrays load without pickle, and the quantitative duration-to-frame check has
zero error.

## Reproduction

Run from the repository root after the B4.2 `balanced_12` artifacts exist:

```powershell
.\.venv\Scripts\python.exe src\fedecai\features\render_feature_pilot_diagnostics.py --run-name balanced_12
```

The script reads only the existing pilot manifest, feature index, log-Mel
arrays, and pitch arrays. It does not reread raw audio, regenerate features,
normalize values, pad or crop sequences, or fit a transformation.

## Outputs

| Artifact | Purpose |
| --- | --- |
| `reports/figures/b4/b4_pilot_log_mel_grid.png` | Compare the 80-band log-Mel representation across all 12 pilot cells |
| `reports/figures/b4/b4_pilot_pitch_grid.png` | Check plausible F0 ranges and preservation of unvoiced gaps |
| `reports/figures/b4/b4_pilot_frame_duration_check.png` | Verify variable frame counts follow full utterance duration |
| `reports/tables/b4/b4_pilot_diagnostic_report.json` | Store machine-readable coverage and diagnostic statistics |

Each grid uses rows for Central, North, and South and columns for Angry, Happy,
Neutral, and Sad. Each cell is one selected utterance from a distinct speaker.

## Quantitative checks

| Measure | Result |
| --- | ---: |
| Pilot utterances | 12 |
| Accent-emotion cells | 12 of 12 |
| Log-Mel frame range | 161 to 801 |
| Configured hop | 160 samples / 10 ms |
| Maximum absolute frame-count error | 0 |
| Minimum voiced-frame fraction | 0.121 |
| Mean voiced-frame fraction | 0.557 |
| Maximum voiced-frame fraction | 0.930 |

For an utterance with `N` samples, centered librosa analysis with the B4 hop
uses `floor(N / 160) + 1` frames. All 12 observed log-Mel frame counts match
this calculation. This confirms that sequence length is currently driven by
audio duration and that no hidden fixed-length padding or cropping occurred.

## Visual review

- Log-Mel panels contain non-empty time-frequency structure in every cell and
  consistently use an 80-band vertical axis and a -80 to 0 dB color scale.
- Pitch values stay within the configured 50 to 500 Hz search range. Unvoiced
  frames remain gaps rather than being imputed with zero or interpolated.
- The duration-to-frame points lie on the expected 10 ms-hop line. Accent is
  encoded by color and emotion by marker; labels are limited to the shortest
  and longest utterances to prevent overlap.
- Titles, axes, row labels, colorbar, and legend were manually inspected after
  rendering. No clipping or overlapping panel labels remain.

## Interpretation boundary

These figures are extraction diagnostics, not evidence that an accent or an
emotion has a particular spectral or pitch pattern. There is only one selected
utterance per accent-emotion cell, so speaker, sentence, duration, and recording
effects are confounded. Population comparisons must use the full eligible data
with speaker-aware inference and the finalized B5 split.

## Acceptance decision

B4.3 is accepted. The pilot provides no evidence of a shape, range, missingness,
or unintended padding/cropping defect in log-Mel and pitch extraction. This
acceptance does not authorize model training or train-fitted preprocessing and
does not mark B4 complete. The remaining B4 requirements are tracked in
`B4_completion_checklist.md`.
