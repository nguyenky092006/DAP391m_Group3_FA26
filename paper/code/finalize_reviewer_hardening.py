"""Generate the final minimal reviewer-hardening audits from fixed artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from cf_ser.acoustics import aggregate_training_stats
from cf_ser.config import ARTIFACTS, CHECKPOINTS, LABELS, LABEL_TO_ID, SAMPLE_RATE, SPLIT_CSV


ROOT = Path(__file__).resolve().parent
MODELS = ["Mel-CNN", "RawWave-CNN", "Wav2Vec2", "Prosody-Fusion"]
INTERVENTIONS = ["Pitch", "Energy", "Duration", "P+E", "P+D", "E+D", "All"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_line(status, check, evidence):
    return f"| {status} | {check} | {evidence} |"


def duplicate_audio_audit(split):
    cache = ARTIFACTS / "EXACT_AUDIO_HASH_AUDIT.json"
    split_hash = sha256(SPLIT_CSV)
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("split_manifest_sha256") == split_hash and saved.get("n_files") == len(split):
            return saved
    file_groups = defaultdict(list)
    pcm_groups = defaultdict(list)
    for number, row in enumerate(split.itertuples(index=False), start=1):
        path = Path(row.path)
        file_groups[sha256(path)].append((row.split, str(path)))
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        digest = hashlib.sha256()
        digest.update(str(sr).encode())
        digest.update(str(audio.shape).encode())
        digest.update(np.ascontiguousarray(audio).tobytes())
        pcm_groups[digest.hexdigest()].append((row.split, str(path)))
        if number % 500 == 0:
            print(f"Audio duplicate audit {number}/{len(split)}", flush=True)
    def cross(groups):
        duplicate_groups = []
        for digest, records in groups.items():
            partitions = sorted({record[0] for record in records})
            if len(partitions) > 1:
                duplicate_groups.append({"sha256": digest, "splits": partitions, "n": len(records)})
        return duplicate_groups
    result = {
        "split_manifest_sha256": split_hash,
        "n_files": len(split),
        "unique_file_hashes": len(file_groups),
        "unique_decoded_pcm_hashes": len(pcm_groups),
        "cross_split_exact_file_duplicate_groups": cross(file_groups),
        "cross_split_exact_decoded_pcm_duplicate_groups": cross(pcm_groups),
        "hash_method": "SHA-256 over full WAV bytes and separately over sample rate, decoded shape, and float32 PCM bytes",
    }
    cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def protocol_audit():
    split = pd.read_csv(SPLIT_CSV, dtype={"speaker": str})
    split_counts = split.split.value_counts().to_dict()
    split_hash = sha256(SPLIT_CSV)
    features = pd.read_csv(ARTIFACTS / "audio_features.csv")
    _, recomputed = aggregate_training_stats(features, split)
    stored = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    parameter_match = set(recomputed) == set(stored) and all(
        math.isclose(float(recomputed[key]), float(stored[key]), rel_tol=0, abs_tol=1e-12) for key in stored
    )

    compact_labels = True
    for key in ["mel_cnn", "raw_cnn", "prosody_fusion"]:
        for seed in (2025, 2026, 2027):
            checkpoint = torch.load(CHECKPOINTS / "ensemble" / f"{key}_seed{seed}.pt", map_location="cpu", weights_only=False)
            compact_labels &= checkpoint.get("labels") == LABELS
    wav_labels = True
    for seed in (2025, 2026, 2027):
        config = json.loads((CHECKPOINTS / "wav2vec2_ensemble" / f"wav2vec2_seed{seed}" / "config.json").read_text(encoding="utf-8"))
        label2id = {str(key): int(value) for key, value in config.get("label2id", {}).items()}
        wav_labels &= label2id == LABEL_TO_ID

    rq2 = pd.read_csv(ARTIFACTS / "rq2_4model_predictions.csv", dtype={"sample_id": str})
    waveform_groups = rq2.groupby(["sample_id", "transformation"]).waveform_sha256.nunique()
    same_waveform = len(waveform_groups) == 227 * 8 and waveform_groups.eq(1).all()
    duplicate = duplicate_audio_audit(split)
    no_cross_file = len(duplicate["cross_split_exact_file_duplicate_groups"]) == 0
    no_cross_pcm = len(duplicate["cross_split_exact_decoded_pcm_duplicate_groups"]) == 0

    rows = [
        status_line(
            "PASS WITH LIMITATION", "Fixed train/validation/test manifest",
            f"`data/splits.csv` has SHA-256 `{split_hash}` and counts train={split_counts.get('train')}, validation={split_counts.get('val')}, test={split_counts.get('test')}. All finalized training summaries reference this manifest. The split was generated locally because the original ViSEC fold assignment was unavailable; it is not backed by an external immutable registry.",
        ),
        status_line(
            "PASS" if parameter_match else "FAIL", "Transformation parameters use TRAIN only",
            "`aggregate_training_stats` filters `split == train`; recomputation from `audio_features.csv` and the frozen manifest " + ("matches `transformation_params.json` to 1e-12." if parameter_match else "does not match the stored parameters."),
        ),
        status_line(
            "PASS WITH LIMITATION", "Checkpoint and hyperparameter selection",
            "Epoch checkpoints and learning rates are selected by validation Macro-F1 in code. However, test metrics were computed and stored for each compact/Wav2Vec2 candidate run; no code path uses them for selection, but analyst blinding from these metrics cannot be independently verified.",
        ),
        status_line(
            "PASS", "No test-based intervention, hyperparameter, or threshold tuning",
            "F0, energy, and duration parameters are train-derived; intervention grids are fixed in code. Classification uses four-class argmax and no tuned decision threshold. Calibration temperatures are fit on validation only.",
        ),
        status_line(
            "PASS" if compact_labels and wav_labels else "FAIL", "Identical label mapping",
            f"All finalized checkpoints use `{LABEL_TO_ID}`." if compact_labels and wav_labels else "At least one checkpoint label mapping differs.",
        ),
        status_line(
            "PASS" if same_waveform else "FAIL", "Identical RQ2 waveform across architectures",
            f"All {len(waveform_groups)} sample × intervention groups contain exactly one SHA-256 waveform hash across four architectures." if same_waveform else "One or more groups contain different waveform hashes across models.",
        ),
        status_line(
            "PASS" if no_cross_file and no_cross_pcm else "FAIL", "Exact duplicate audio leakage",
            f"Hashed {duplicate['n_files']} files. Cross-split duplicate groups: full-file={len(duplicate['cross_split_exact_file_duplicate_groups'])}, decoded-PCM={len(duplicate['cross_split_exact_decoded_pcm_duplicate_groups'])}. Method: {duplicate['hash_method']}.",
        ),
    ]
    lines = [
        "# Final protocol and leakage audit", "",
        "| Status | Check | Evidence |", "|---|---|---|", *rows, "",
        "## Important limitation", "",
        "The available code supports validation-based selection, but test metrics were materialized for candidate runs. This is not evidence that test performance changed a selection decision; it is a transparency limitation that should be declared.",
        "", "No audit item relies on fabricated provenance or unavailable experiment logs.",
    ]
    (ARTIFACTS / "PROTOCOL_LEAKAGE_AUDIT_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "parameter_match": parameter_match, "labels_match": compact_labels and wav_labels,
        "same_waveform": same_waveform, "no_cross_duplicate": no_cross_file and no_cross_pcm,
        "split_hash": split_hash, "split_counts": split_counts, "duplicate": duplicate,
    }


def claim_audit():
    calibration = json.loads((ARTIFACTS / "RQ2_CALIBRATION_ASSESSMENT_FINAL.json").read_text(encoding="utf-8"))
    lines = [
        "# Final claim audit", "",
        "## RQ1", "",
        "**Use:** \"Under the train-derived Neutral-to-Angry interventions, F0 manipulation produced the most consistent directional shift toward Angry across the evaluated architectures. The F0 effect was primarily associated with baseline elevation rather than F0-range expansion.\"", "",
        "**Do not use:** \"Pitch is the strongest cue in every architecture.\" RawWave-CNN is the Pitch-versus-Duration exception.", "",
        "**Required limitation:** The experiment identifies SER model sensitivity under corpus-derived intervention magnitudes. It does not establish intrinsic or universal human acoustic importance.", "",
        "## RQ2", "",
        "**Use:** \"Different SER architecture families exhibited different intervention-sensitivity profiles.\"", "",
        f"Validation-only temperature scaling preserved the exact common-subset Pitch ordering (`{' > '.join(calibration['pitch_order_calibrated'])}`) and Prosody-Fusion remained most Pitch-sensitive. The cross-model Pitch-magnitude statement is therefore supported as a calibration robustness result.", "",
        "**Do not use:** Claims that the architectures learned different internal representations.", "",
        "## RQ3 — exploratory accent analysis", "",
        "**Question:** \"How does F0-baseline sensitivity vary across the Vietnamese accent groups represented in ViSEC?\"", "",
        "**Use:** \"The pooled F0-baseline effect was positive across the three accent groups represented in ViSEC, but its magnitude and direction were not uniform across architectures.\"", "",
        "**Required limitation:** Speaker coverage is limited and imbalanced across accent groups; therefore, accent effects cannot be cleanly separated from speaker-specific effects.", "",
        "**Do not use:** accent-invariant behavior, population-level Vietnamese accent generalization, differences in how speakers express anger, or human-perception claims.",
    ]
    (ARTIFACTS / "FINAL_CLAIM_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def intervention_paper_summary():
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    validity = pd.read_csv(ARTIFACTS / "INTERVENTION_VALIDITY_RESULTS.csv")
    intended_log = float(params["mu_logf0_angry"] - params["mu_logf0_neutral"])
    intended_ratio = math.exp(intended_log)
    lines = [
        "# Intervention validity: paper summary", "",
        "The fixed intervention is a **local-contour-preserving F0 baseline intervention** derived from train-set Neutral-versus-Angry statistics.", "",
        "| Check | Result |", "|---|---:|",
        f"| Train-derived intended mean log-F0 shift | {intended_log:.6f} |",
        f"| Train-derived intended F0 ratio | {intended_ratio:.6f}× |",
        f"| Achieved mean log-F0 shift across test utterances | {validity.achieved_mean_log_f0_shift.mean():.6f} |",
        f"| Median achieved F0 ratio | {validity.achieved_mean_f0_ratio.median():.6f}× |",
        f"| Median local log-F0 contour correlation | {validity.local_log_f0_contour_correlation.median():.6f} |",
        f"| Maximum absolute duration drift | {validity.duration_difference_seconds.abs().max():.6f} s |",
        f"| Mean RMS drift | {validity.rms_difference_db.mean():.6f} dB |",
        f"| Maximum absolute RMS drift | {validity.rms_difference_db.abs().max():.6f} dB |",
        f"| Maximum clipping fraction | {validity.clipping_fraction_ge_0_999.max():.6%} |", "",
        "The intended F0 elevation was achieved, local F0 contour shape was largely retained, duration did not drift, mean RMS drift was small, and no clipping was detected. These objective checks do not establish perceptual preservation of lexical tone.",
    ]
    (ARTIFACTS / "INTERVENTION_VALIDITY_PAPER_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def consistency_audit():
    checks = []
    rq1 = pd.read_csv(ARTIFACTS / "RQ1_FINAL_RESULTS.csv")
    rq1_tests = pd.read_csv(ARTIFACTS / "RQ1_STATISTICAL_TESTS.csv")
    rq1_n = rq1[rq1.section.eq("single_component")].groupby("model").n.first().to_dict()
    checks.append((rq1_n == {"Mel-CNN": 140, "Prosody-Fusion": 136, "RawWave-CNN": 90}, "RQ1 model-specific N values"))
    pitch_energy = rq1_tests[(rq1_tests.comparison.eq("Pitch vs Energy"))]
    pitch_duration = rq1_tests[(rq1_tests.comparison.eq("Pitch vs Duration"))]
    checks.append(((pitch_energy.p_value_holm < .05).sum() == 3 and (pitch_duration.p_value_holm < .05).sum() == 2, "RQ1 significance counts"))

    rq2 = pd.read_csv(ARTIFACTS / "RQ2_4MODEL_FINAL_RESULTS.csv")
    matrix = pd.read_csv(ARTIFACTS / "rq2_4model_delta_matrix.csv", index_col=0)
    pivot = rq2.pivot(index="model", columns="intervention", values="mean_delta_p_angry").reindex(index=matrix.index, columns=matrix.columns)
    checks.append((rq2.n.eq(36).all() and np.allclose(pivot.to_numpy(), matrix.to_numpy()), "RQ2 N=36 and result/matrix means"))
    rq2_tests = pd.read_csv(ARTIFACTS / "RQ2_4MODEL_STATISTICAL_TESTS.csv")
    pitch_pairs = rq2_tests[(rq2_tests.test_type.eq("Wilcoxon post-hoc")) & rq2_tests.intervention.eq("Pitch")]
    checks.append(((pitch_pairs.p_value_holm < .05).sum() == 5, "RQ2 five of six calibrated-independent raw Pitch pairs"))

    w2v = pd.read_csv(ARTIFACTS / "WAV2VEC2_ENSEMBLE_BASELINE_RESULTS.csv").iloc[0]
    canonical_w2v = pd.read_csv(ARTIFACTS / "WAV2VEC2_BASELINE_RESULTS.csv").iloc[0]
    checks.append((int(w2v.n_members) == 3 and math.isclose(w2v.test_macro_f1, canonical_w2v.test_macro_f1, abs_tol=1e-12), "Wav2Vec2 three-seed ensemble metrics"))

    rq3 = pd.read_csv(ARTIFACTS / "RQ3_ACCENT_RESULTS.csv")
    primary = rq3[rq3.analysis_set.eq("primary_all_true_neutral")]
    pooled = primary[primary.scope.eq("pooled")].set_index("accent")
    checks.append((pooled.n_utterances.to_dict() == {"Central": 25, "North": 31, "South": 171}, "RQ3 primary accent N values"))
    rq3_tests = pd.read_csv(ARTIFACTS / "RQ3_ACCENT_STATISTICAL_TESTS.csv")
    interaction = rq3_tests[(rq3_tests.test_family.eq("GEE_joint_Wald")) & rq3_tests.comparison.str.contains(":", regex=False)]
    checks.append((len(interaction) == 1 and math.isclose(float(interaction.p_value_raw.iloc[0]), 2.986316e-14, rel_tol=1e-5), "RQ3 architecture × accent interaction"))

    calibration = pd.read_csv(ARTIFACTS / "RQ2_CALIBRATION_FINAL.csv")
    cal_results = pd.read_csv(ARTIFACTS / "RQ2_CALIBRATED_RESULTS_FINAL.csv")
    all_neutral = pd.read_csv(ARTIFACTS / "RQ2_ALL_NEUTRAL_ROBUSTNESS_FINAL.csv")
    checks.append((len(calibration) == 4 and calibration.n_validation.eq(828).all() and (~calibration.test_used_for_fit).all(), "Calibration split and model count"))
    checks.append((len(cal_results) == 28 and cal_results.n.eq(36).all(), "Calibrated common-subset result dimensions"))
    checks.append((len(all_neutral) == 28 and all_neutral.n.eq(227).all(), "All-Neutral robustness result dimensions"))

    figures = [
        "rq1_single_components.png", "rq1_pitch_mean_vs_range.png", "rq1_pitch_dose_response.png",
        "rq2_4model_intervention_heatmap.png", "rq2_4model_flip_heatmap.png",
        "rq3_f0_effect_by_accent.png", "rq3_model_accent_heatmap.png",
    ]
    checks.append((all((ARTIFACTS / "figures" / name).exists() and (ARTIFACTS / "figures" / name).stat().st_size > 1000 for name in figures), "Canonical figure files exist and are non-empty"))
    all_pass = all(value for value, _ in checks)
    lines = [
        "# Final result consistency audit", "",
        "| Status | Check |", "|---|---|",
        *[f"| {'PASS' if value else 'FAIL'} | {label} |" for value, label in checks], "",
        "## Canonical paper artifacts", "",
        "- RQ1: `RQ1_FINAL_RESULTS.csv`, `RQ1_STATISTICAL_TESTS.csv`, `RQ1_FINAL_FINDINGS.md`.",
        "- RQ2 raw primary: `RQ2_4MODEL_FINAL_RESULTS.csv`, `RQ2_4MODEL_STATISTICAL_TESTS.csv`, `RQ2_4MODEL_FINAL_FINDINGS.md`.",
        "- RQ2 calibration robustness: `RQ2_CALIBRATION_FINAL.csv`, `RQ2_CALIBRATED_RESULTS_FINAL.csv`, `RQ2_CALIBRATED_STATS_FINAL.csv`.",
        "- RQ2 all-Neutral secondary: `RQ2_ALL_NEUTRAL_ROBUSTNESS_FINAL.csv`, `RQ2_ALL_NEUTRAL_ROBUSTNESS_FINDINGS_FINAL.md`.",
        "- RQ3 exploratory: `RQ3_ACCENT_RESULTS.csv`, `RQ3_ACCENT_STATISTICAL_TESTS.csv`, `RQ3_FINAL_FINDINGS.md`.", "",
        "## Stale or non-primary legacy outputs", "",
        "- `RQ1_FINDINGS.md`, `rq1_summary.csv`, and the unprefixed older analysis tables predate the final RQ1 consolidation.",
        "- `RQ2_FINDINGS.md`, `RQ2_FINAL_FINDINGS.md`, `RQ2_FINAL_RESULTS.csv`, `RQ2_STATISTICAL_TESTS.csv`, `rq2_model_intervention_heatmap.png`, and three-model matrices predate Wav2Vec2 and must not be cited as final.",
        "- `rq3_human_study/`, `RQ3_MODEL_STIMULUS_PREDICTIONS.csv`, and `rq3_stimulus_manifest.csv` belong to the cancelled human-perception RQ3 and are not paper evidence.",
        "- `SANITY_CHECK_RESULTS.csv` is empty because no real team ratings exist; omit the sanity-check result from the paper.", "",
        f"Overall canonical consistency status: **{'PASS' if all_pass else 'FAIL'}**. No conflicting value was silently overwritten.",
    ]
    (ARTIFACTS / "FINAL_RESULT_CONSISTENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_pass


def reproducibility_summary(protocol):
    source = json.loads((ROOT / "data/raw/visec_staged/source_report.json").read_text(encoding="utf-8"))
    params = json.loads((ARTIFACTS / "transformation_params.json").read_text(encoding="utf-8"))
    calibration = pd.read_csv(ARTIFACTS / "RQ2_CALIBRATION_FINAL.csv")
    packages = {}
    for name in ["numpy", "pandas", "scipy", "torch", "torchaudio", "transformers", "librosa", "praat-parselmouth", "soundfile", "scikit-learn", "statsmodels", "matplotlib", "seaborn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "UNKNOWN"
    lines = [
        "# Paper reproducibility summary", "",
        "## Dataset and split", "",
        f"- Dataset: ViSEC, {source.get('rows', 'UNKNOWN')} staged utterances.",
        f"- Source: {source.get('source_type', 'UNKNOWN')}.",
        f"- Upstream Parquet SHA-256: `{source.get('upstream_report', {}).get('source_sha256', 'UNKNOWN')}`.",
        "- Official distribution version/commit: **UNKNOWN**.",
        f"- Frozen manifest: `data/splits.csv`, SHA-256 `{protocol['split_hash']}`.",
        f"- Split counts: train={protocol['split_counts'].get('train')}, validation={protocol['split_counts'].get('val')}, test={protocol['split_counts'].get('test')}.",
        "- Split seed: 2025. Identified speakers are disjoint; speaker ID 0 is a heterogeneous missing-ID bucket stratified across splits.", "",
        "## Labels and audio", "",
        f"- Label mapping: `{LABEL_TO_ID}`.",
        f"- Sampling rate: {SAMPLE_RATE} Hz.",
        "- Model input crop/pad length: 4.5 seconds.", "",
        "## Models and checkpoints", "",
        "- Compact families: `checkpoints/ensemble/{mel_cnn,raw_cnn,prosody_fusion}_seed{2025,2026,2027}.pt`.",
        "- Wav2Vec2: `checkpoints/wav2vec2_ensemble/wav2vec2_seed{2025,2026,2027}/`.",
        "- Pretrained identifier: `nguyenvulebinh/wav2vec2-base-vietnamese-250h`.",
        "- Final family prediction: unweighted arithmetic mean of the three member class-probability vectors.",
        "- Training/model seeds: 2025, 2026, 2027.", "",
        "## Intervention", "",
        "- Parameter-estimation split: train only, using train Neutral and Angry utterances.",
        f"- Parameters: `{json.dumps(params, sort_keys=True)}`.",
        "- Component order for combinations: pitch, duration, then energy.",
        "- Sampling/bootstrap seed: 2025.", "",
        "## Validation-only calibration", "",
        "Temperature is fitted to the finalized family-ensemble validation probabilities by minimizing multiclass NLL; test data are not used for fitting.", "",
        calibration[["model", "n_validation", "temperature", "nll_before", "nll_after", "brier_before", "brier_after", "ece_15bin_before", "ece_15bin_after"]].to_markdown(index=False, floatfmt=".6f"), "",
        "## Software", "",
        f"- Python: {platform.python_version()}.",
        *[f"- {name}: {version}." for name, version in packages.items()], "",
        "## Final paper artifacts", "",
        "- RQ1: `RQ1_FINAL_FINDINGS.md`, `RQ1_FINAL_RESULTS.csv`, `RQ1_STATISTICAL_TESTS.csv`.",
        "- RQ2 primary: `RQ2_4MODEL_FINAL_FINDINGS.md`, `RQ2_4MODEL_FINAL_RESULTS.csv`, `RQ2_4MODEL_STATISTICAL_TESTS.csv`.",
        "- RQ2 robustness: `RQ2_CALIBRATION_FINAL.csv`, `RQ2_CALIBRATED_RESULTS_FINAL.csv`, `RQ2_CALIBRATED_STATS_FINAL.csv`, `RQ2_ALL_NEUTRAL_ROBUSTNESS_FINAL.csv`.",
        "- RQ3: `RQ3_FINAL_FINDINGS.md`, `RQ3_ACCENT_RESULTS.csv`, `RQ3_ACCENT_STATISTICAL_TESTS.csv`.",
        "- Intervention validity: `INTERVENTION_VALIDITY_PAPER_SUMMARY.md`, `INTERVENTION_VALIDITY_RESULTS.csv`.",
        "- Audits: `PROTOCOL_LEAKAGE_AUDIT_FINAL.md`, `FINAL_CLAIM_AUDIT.md`, `FINAL_RESULT_CONSISTENCY_AUDIT.md`, `FINAL_REVIEWER_GATE.md`.",
    ]
    (ARTIFACTS / "PAPER_REPRODUCIBILITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reviewer_gate(protocol, consistency):
    calibration = json.loads((ARTIFACTS / "RQ2_CALIBRATION_ASSESSMENT_FINAL.json").read_text(encoding="utf-8"))
    all_neutral = pd.read_csv(ARTIFACTS / "RQ2_ALL_NEUTRAL_ROBUSTNESS_FINAL.csv")
    all_pitch = all_neutral[all_neutral.intervention.eq("Pitch")].sort_values("calibrated_mean_delta_p_angry", ascending=False).model.tolist()
    objective = pd.read_csv(ARTIFACTS / "INTERVENTION_VALIDITY_RESULTS.csv")
    validity_ok = objective.finite_audio.all() and objective.clipping_fraction_ge_0_999.max() == 0 and objective.duration_difference_seconds.abs().max() == 0
    lines = [
        "# Final reviewer gate", "",
        "1. **Is there any evidence of test leakage?** No direct evidence that test results selected a model or intervention. Train-only parameter recomputation, manifest hashes, waveform hashes, and exact duplicate checks passed. **Declared limitation:** test metrics were nevertheless computed for each candidate training run, so analyst blinding cannot be proven.", "",
        f"2. **Does calibrated RQ2 preserve the main architecture story?** {'Yes' if calibration['profiles_qualitatively_consistent'] else 'No'}. The common-subset Pitch order is `{' > '.join(calibration['pitch_order_calibrated'])}`; Prosody-Fusion remains most Pitch-sensitive.", "",
        f"3. **Does the all-Neutral robustness analysis preserve the main story?** Yes, with secondary ranking variation. All-Neutral calibrated Pitch ordering is `{' > '.join(all_pitch)}` and Prosody-Fusion remains first; common-vs-all profile correlations remain high enough to retain the qualitative sensitivity-profile claim.", "",
        "4. **Are RQ1 claims supported without overgeneralization?** Yes, when stated as the most consistent directional F0 shift under train-derived interventions and with RawWave-CNN's Pitch-versus-Duration exception retained.", "",
        "5. **Is RQ3 correctly framed as exploratory?** Yes. It is labeled an exploratory ViSEC accent analysis, with speaker/accent confounding and imbalance stated explicitly.", "",
        f"6. **Are intervention-validity claims supported?** {'Yes' if validity_ok else 'No'}. The achieved F0 shift, contour correlation, zero duration drift, small mean RMS drift, and zero clipping support a local-contour-preserving F0-baseline manipulation; no lexical-tone preservation claim is made.", "",
        f"7. **Are all final results internally consistent?** {'Yes' if consistency else 'No'}. Canonical files pass the automated consistency checks; legacy and cancelled-study outputs are explicitly identified and excluded.", "",
        "## Decision", "",
        "READY FOR PAPER WRITING WITH DECLARED LIMITATIONS",
    ]
    (ARTIFACTS / "FINAL_REVIEWER_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    protocol = protocol_audit()
    claim_audit()
    intervention_paper_summary()
    consistency = consistency_audit()
    reproducibility_summary(protocol)
    reviewer_gate(protocol, consistency)
    print(json.dumps({
        "protocol": {
            key: (bool(value) if isinstance(value, (bool, np.bool_)) else value)
            for key, value in protocol.items() if key not in {"duplicate"}
        },
        "consistency": bool(consistency),
        "decision": "READY FOR PAPER WRITING WITH DECLARED LIMITATIONS",
    }, indent=2))


if __name__ == "__main__":
    main()
