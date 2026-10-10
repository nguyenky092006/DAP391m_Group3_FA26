from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from cf_ser.acoustics import acoustic_features
from cf_ser.config import ARTIFACTS, AUDIO_EXAMPLES, SAMPLE_RATE, TRANSFORMS

pred = pd.read_csv(ARTIFACTS / "counterfactual_predictions.csv")
assert len(pred) == pred.sample_id.nunique() * pred.model.nunique() * len(TRANSFORMS)
assert not pred.isna().any().any()
prob_cols = ["p_happy_transformed", "p_neutral_transformed", "p_sad_transformed", "p_angry_transformed"]
assert np.allclose(pred[prob_cols].sum(axis=1), 1.0, atol=1e-5)
assert (pred.groupby(["sample_id", "model"]).size() == len(TRANSFORMS)).all()

rows = []
for path in sorted(AUDIO_EXAMPLES.glob("*.wav")):
    y, sr = sf.read(path, dtype="float32")
    feat = acoustic_features(y, sr)
    transformation = next(t for t in TRANSFORMS if path.stem.endswith(t))
    rows.append({"file": path.name, "transformation": transformation, "sample_rate": sr,
                 "samples": len(y), "finite": bool(np.isfinite(y).all()),
                 "peak": float(np.max(np.abs(y))), "clipped_fraction": float(np.mean(np.abs(y) >= .999)), **feat})
qc = pd.DataFrame(rows)
assert len(qc) == 5 * len(TRANSFORMS)
assert (qc.sample_rate == SAMPLE_RATE).all() and (qc.samples > 0).all() and qc.finite.all()
assert (qc.peak <= 1.0).all() and (qc.clipped_fraction < .01).all()
qc.to_csv(ARTIFACTS / "qc_summary.csv", index=False)

means = qc.groupby("transformation")[["logf0_mean", "log_rms_mean", "duration", "peak", "clipped_fraction"]].mean()
report = "# Automated quality-control report\n\n"
report += f"- Prediction rows: {len(pred):,}; complete design: yes.\n"
report += f"- Saved examples: {len(qc)} (5 for each T0-T7).\n"
report += "- All example waveforms load, remain 16 kHz, are non-empty and finite: yes.\n"
report += f"- Maximum peak: {qc.peak.max():.6f}; maximum sample-level clipping fraction: {qc.clipped_fraction.max():.4%}.\n"
report += "- Automated checks cannot replace human listening or confirm lexical-tone preservation.\n\n"
report += means.to_markdown() + "\n"
(ARTIFACTS / "QC_REPORT.md").write_text(report, encoding="utf-8")
print(report)
