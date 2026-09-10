"""Create the frozen ViSEC speaker-disjoint train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
TARGET_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPEAKER_COUNTS = {"train": 103, "validation": 22, "test": 22}
DEFAULT_SEED = 391
DEFAULT_RESTARTS = 32
DEFAULT_STEPS = 30_000


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_speaker_profiles(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _is_true(row["is_eligible_for_split"]):
            continue
        speaker_id = row["speaker_id"]
        profile = profiles.setdefault(
            speaker_id,
            {
                "samples": 0,
                "accent": row["accent_clean"],
                "gender": row["gender_clean"],
                "emotions": Counter(),
            },
        )
        if profile["accent"] != row["accent_clean"]:
            raise ValueError(f"Speaker {speaker_id} has multiple clean accent labels")
        if profile["gender"] != row["gender_clean"]:
            raise ValueError(f"Speaker {speaker_id} has multiple clean gender labels")
        profile["samples"] += 1
        profile["emotions"][row["emotion"]] += 1
    return profiles


def _feature_vector(profile: dict[str, Any]) -> dict[str, int]:
    vector = {"samples": profile["samples"]}
    for emotion in ("angry", "happy", "neutral", "sad"):
        vector[f"emotion:{emotion}"] = profile["emotions"][emotion]
    for accent in ("central", "north", "south"):
        vector[f"accent_samples:{accent}"] = profile["samples"] if profile["accent"] == accent else 0
        vector[f"accent_speakers:{accent}"] = 1 if profile["accent"] == accent else 0
    for gender in ("female", "male"):
        vector[f"gender_samples:{gender}"] = profile["samples"] if profile["gender"] == gender else 0
        vector[f"gender_speakers:{gender}"] = 1 if profile["gender"] == gender else 0
    return vector


def _feature_weights(feature_names: list[str]) -> dict[str, float]:
    weights = {}
    for name in feature_names:
        if name == "samples":
            weights[name] = 60.0
        elif name.startswith("emotion:"):
            weights[name] = 5.0
        elif name.startswith("accent_samples:"):
            weights[name] = 4.0
        elif name.startswith("accent_speakers:"):
            weights[name] = 2.0
        elif name.startswith("gender_samples:"):
            weights[name] = 1.0
        else:
            weights[name] = 0.5
    return weights


def optimize_assignment(
    profiles: dict[str, dict[str, Any]],
    split_counts: dict[str, int],
    seed: int,
    restarts: int,
    steps: int,
    forced_train_speaker: str,
) -> tuple[dict[str, str], float]:
    speaker_ids = sorted(profiles, key=int)
    if sum(split_counts.values()) != len(speaker_ids):
        raise ValueError("Speaker counts do not add up to the number of speakers")
    if forced_train_speaker not in profiles:
        raise ValueError("Forced train speaker is missing")

    vectors = {speaker_id: _feature_vector(profile) for speaker_id, profile in profiles.items()}
    feature_names = list(next(iter(vectors.values())))
    global_totals = {
        feature: sum(vector[feature] for vector in vectors.values()) for feature in feature_names
    }
    weights = _feature_weights(feature_names)

    def totals_for(assignment: dict[str, str]) -> dict[str, dict[str, int]]:
        totals = {split: {feature: 0 for feature in feature_names} for split in SPLITS}
        for speaker_id, split in assignment.items():
            for feature, value in vectors[speaker_id].items():
                totals[split][feature] += value
        return totals

    def score(totals: dict[str, dict[str, int]]) -> float:
        value = 0.0
        for split in SPLITS:
            target = TARGET_FRACTIONS[split]
            for feature in feature_names:
                total = global_totals[feature]
                if total:
                    difference = totals[split][feature] / total - target
                    value += weights[feature] * difference * difference
            for emotion in ("angry", "happy", "neutral", "sad"):
                if totals[split][f"emotion:{emotion}"] == 0:
                    value += 100.0
            for accent in ("central", "north", "south"):
                if totals[split][f"accent_speakers:{accent}"] == 0:
                    value += 100.0
        return value

    rng = random.Random(seed)
    movable = [speaker_id for speaker_id in speaker_ids if speaker_id != forced_train_speaker]
    best_assignment: dict[str, str] | None = None
    best_score = math.inf

    for _ in range(restarts):
        shuffled = movable[:]
        rng.shuffle(shuffled)
        assignment = {forced_train_speaker: "train"}
        train_needed = split_counts["train"] - 1
        for speaker_id in shuffled[:train_needed]:
            assignment[speaker_id] = "train"
        offset = train_needed
        for speaker_id in shuffled[offset : offset + split_counts["validation"]]:
            assignment[speaker_id] = "validation"
        offset += split_counts["validation"]
        for speaker_id in shuffled[offset:]:
            assignment[speaker_id] = "test"

        totals = totals_for(assignment)
        current_score = score(totals)
        by_split = {
            split: [speaker_id for speaker_id, assigned in assignment.items() if assigned == split]
            for split in SPLITS
        }

        for step in range(steps):
            split_a, split_b = rng.sample(SPLITS, 2)
            candidates_a = [speaker_id for speaker_id in by_split[split_a] if speaker_id != forced_train_speaker]
            if not candidates_a:
                continue
            speaker_a = rng.choice(candidates_a)
            speaker_b = rng.choice(by_split[split_b])
            for feature in feature_names:
                totals[split_a][feature] += vectors[speaker_b][feature] - vectors[speaker_a][feature]
                totals[split_b][feature] += vectors[speaker_a][feature] - vectors[speaker_b][feature]
            new_score = score(totals)
            temperature = 0.01 * (1.0 - step / steps) + 1e-6
            accept = new_score <= current_score or rng.random() < math.exp((current_score - new_score) / temperature)
            if accept:
                assignment[speaker_a], assignment[speaker_b] = split_b, split_a
                by_split[split_a].remove(speaker_a)
                by_split[split_a].append(speaker_b)
                by_split[split_b].remove(speaker_b)
                by_split[split_b].append(speaker_a)
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_assignment = dict(assignment)
            else:
                for feature in feature_names:
                    totals[split_a][feature] += vectors[speaker_a][feature] - vectors[speaker_b][feature]
                    totals[split_b][feature] += vectors[speaker_b][feature] - vectors[speaker_a][feature]

    if best_assignment is None:
        raise RuntimeError("Split optimization did not produce an assignment")
    return best_assignment, best_score


def apply_assignment(
    rows: list[dict[str, str]], assignment: dict[str, str], seed: int
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row: dict[str, Any] = dict(source)
        eligible = _is_true(row["is_eligible_for_split"])
        row.update(
            {
                "split": assignment[row["speaker_id"]] if eligible else "excluded",
                "split_version": "speaker_disjoint_v1",
                "split_seed": seed,
                "test_membership_frozen": eligible,
            }
        )
        output.append(row)
    return output


def build_report(
    rows: list[dict[str, Any]],
    assignment: dict[str, str],
    objective_score: float,
    seed: int,
    forced_train_speaker: str,
) -> dict[str, Any]:
    eligible = [row for row in rows if row["split"] in SPLITS]
    total_samples = len(eligible)
    split_speakers = {
        split: sorted({row["speaker_id"] for row in eligible if row["split"] == split}, key=int)
        for split in SPLITS
    }
    distributions = {}
    for split in SPLITS:
        members = [row for row in eligible if row["split"] == split]
        distributions[split] = {
            "samples": len(members),
            "sample_fraction": len(members) / total_samples,
            "speakers": len(split_speakers[split]),
            "accent_samples": dict(sorted(Counter(row["accent_clean"] for row in members).items())),
            "accent_speakers": dict(
                sorted(Counter(
                    next(row["accent_clean"] for row in members if row["speaker_id"] == speaker_id)
                    for speaker_id in split_speakers[split]
                ).items())
            ),
            "emotion_samples": dict(sorted(Counter(row["emotion"] for row in members).items())),
            "gender_samples": dict(sorted(Counter(row["gender_clean"] for row in members).items())),
        }

    overlap = {}
    for index, split_a in enumerate(SPLITS):
        for split_b in SPLITS[index + 1 :]:
            overlap[f"{split_a}_{split_b}"] = sorted(
                set(split_speakers[split_a]) & set(split_speakers[split_b]), key=int
            )
    assignment_text = "\n".join(f"{speaker_id}={assignment[speaker_id]}" for speaker_id in sorted(assignment, key=int))
    eligible_hashes = [row["audio_sha256"] for row in eligible]
    largest_profile = [row for row in eligible if row["speaker_id"] == forced_train_speaker]
    validation = {
        "speaker_overlap_zero": all(not values for values in overlap.values()),
        "eligible_audio_hashes_unique": len(eligible_hashes) == len(set(eligible_hashes)),
        "all_splits_have_all_emotions": all(
            len(distributions[split]["emotion_samples"]) == 4 for split in SPLITS
        ),
        "all_splits_have_all_accents": all(
            len(distributions[split]["accent_samples"]) == 3 for split in SPLITS
        ),
        "speaker_counts_match_plan": all(
            distributions[split]["speakers"] == SPEAKER_COUNTS[split] for split in SPLITS
        ),
        "excluded_rows_not_assigned": all(
            row["split"] == "excluded" for row in rows if not _is_true(row["is_eligible_for_split"])
        ),
    }
    if not all(validation.values()):
        raise ValueError(f"Split validation failed: {validation}")

    return {
        "split_version": "speaker_disjoint_v1",
        "status": "FROZEN",
        "seed": seed,
        "target_sample_fractions": TARGET_FRACTIONS,
        "planned_speaker_counts": SPEAKER_COUNTS,
        "optimizer": {
            "restarts": DEFAULT_RESTARTS,
            "steps_per_restart": DEFAULT_STEPS,
            "objective_score": objective_score,
        },
        "assignment_sha256": hashlib.sha256(assignment_text.encode("utf-8")).hexdigest(),
        "distributions": distributions,
        "speaker_overlap": overlap,
        "validation": validation,
        "dominant_speaker": {
            "speaker_id": forced_train_speaker,
            "samples": len(largest_profile),
            "eligible_dataset_fraction": len(largest_profile) / total_samples,
            "assigned_split": assignment[forced_train_speaker],
            "decision": "Forced into train because placing this speaker in validation or test would dominate evaluation.",
            "source_limitation": "The released metadata does not document whether speaker_id 0 is a single verified identity or a catch-all identifier.",
        },
        "freeze_rule": "Do not change speaker-to-split membership for model tuning or model comparison. A new split requires a new version and written justification.",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "data/manifests/clean_manifest.csv")
    parser.add_argument("--output", type=Path, default=repo_root / "data/manifests/split_manifest.csv")
    parser.add_argument("--report", type=Path, default=repo_root / "data/manifests/split_report.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input.resolve())
    profiles = build_speaker_profiles(rows)
    forced_train_speaker = max(profiles, key=lambda speaker_id: profiles[speaker_id]["samples"])
    assignment, objective_score = optimize_assignment(
        profiles,
        SPEAKER_COUNTS,
        args.seed,
        DEFAULT_RESTARTS,
        DEFAULT_STEPS,
        forced_train_speaker,
    )
    output = apply_assignment(rows, assignment, args.seed)
    report = build_report(output, assignment, objective_score, args.seed, forced_train_speaker)
    write_csv(output, args.output.resolve())
    write_json(report, args.report.resolve())
    print(json.dumps({
        "status": report["status"],
        "assignment_sha256": report["assignment_sha256"],
        "output": str(args.output.resolve()),
        "report": str(args.report.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
