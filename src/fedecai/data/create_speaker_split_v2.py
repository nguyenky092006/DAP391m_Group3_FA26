"""Create ViSEC speaker-disjoint split v2 with full accent-emotion coverage.

Version 1 is never read as an output target and is never overwritten. Version 2
starts from the B2 clean manifest, assigns whole speakers, and requires every
train/validation/test x accent x emotion cell to contain at least one speaker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .create_speaker_split import (
        DEFAULT_SEED,
        SPEAKER_COUNTS,
        SPLITS,
        TARGET_FRACTIONS,
        build_speaker_profiles,
        read_manifest,
    )
except ImportError:  # Allows direct execution from the repository root.
    from create_speaker_split import (  # type: ignore
        DEFAULT_SEED,
        SPEAKER_COUNTS,
        SPLITS,
        TARGET_FRACTIONS,
        build_speaker_profiles,
        read_manifest,
    )


ACCENTS = ("central", "north", "south")
EMOTIONS = ("angry", "happy", "neutral", "sad")
SPLIT_VERSION = "speaker_disjoint_v2"
DEFAULT_RESTARTS = 64
DEFAULT_STEPS = 40_000
CONSTRUCTION_ATTEMPTS = 2_000
COVERAGE_PENALTY = 1_000_000.0


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def speaker_cells(profile: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (profile["accent"], emotion)
        for emotion in EMOTIONS
        if profile["emotions"][emotion] > 0
    }


def coverage_gaps(
    profiles: dict[str, dict[str, Any]], assignment: dict[str, str]
) -> list[dict[str, str]]:
    observed = {
        (split, *cell)
        for speaker_id, split in assignment.items()
        for cell in speaker_cells(profiles[speaker_id])
    }
    return [
        {"split": split, "accent": accent, "emotion": emotion}
        for split in SPLITS
        for accent in ACCENTS
        for emotion in EMOTIONS
        if (split, accent, emotion) not in observed
    ]


def _choose_covering_speakers(
    profiles: dict[str, dict[str, Any]],
    candidates: set[str],
    target_count: int,
    rng: random.Random,
) -> set[str] | None:
    uncovered = {(accent, emotion) for accent in ACCENTS for emotion in EMOTIONS}
    selected: set[str] = set()
    while uncovered:
        scored = []
        for speaker_id in sorted(candidates - selected, key=int):
            gain = len(speaker_cells(profiles[speaker_id]) & uncovered)
            if gain:
                scored.append((gain, speaker_id))
        if not scored:
            return None
        best_gain = max(gain for gain, _ in scored)
        tied = sorted(
            (speaker_id for gain, speaker_id in scored if gain == best_gain), key=int
        )
        rng.shuffle(tied)
        # Prefer smaller speakers for evaluation when coverage gain is equal.
        tied.sort(key=lambda speaker_id: profiles[speaker_id]["samples"])
        speaker_id = tied[0]
        selected.add(speaker_id)
        uncovered -= speaker_cells(profiles[speaker_id])
        if len(selected) > target_count:
            return None
    return selected


def construct_feasible_assignment(
    profiles: dict[str, dict[str, Any]],
    split_counts: dict[str, int],
    seed: int,
    forced_train_speaker: str,
    attempts: int = CONSTRUCTION_ATTEMPTS,
) -> dict[str, str]:
    """Construct exact speaker counts with complete cross-stratum coverage."""
    if sum(split_counts.values()) != len(profiles):
        raise ValueError("Speaker counts do not add up to the number of speakers")
    if forced_train_speaker not in profiles:
        raise ValueError("Forced train speaker is missing")

    all_speakers = set(profiles)
    for attempt in range(attempts):
        rng = random.Random(seed + attempt * 10_007)
        remaining = all_speakers - {forced_train_speaker}
        chosen: dict[str, set[str]] = {}
        evaluation_order = ["validation", "test"]
        rng.shuffle(evaluation_order)

        failed = False
        for split in evaluation_order:
            selected = _choose_covering_speakers(
                profiles, remaining, split_counts[split], rng
            )
            if selected is None:
                failed = True
                break
            remaining -= selected
            chosen[split] = selected

        if failed:
            continue

        # Fill remaining evaluation capacity only after both evaluation splits
        # have secured their coverage speakers.
        for split in evaluation_order:
            extra_needed = split_counts[split] - len(chosen[split])
            extras = sorted(remaining, key=int)
            rng.shuffle(extras)
            extras.sort(key=lambda speaker_id: profiles[speaker_id]["samples"])
            chosen[split].update(extras[:extra_needed])
            remaining -= set(extras[:extra_needed])
        chosen["train"] = remaining | {forced_train_speaker}
        if any(len(chosen[split]) != split_counts[split] for split in SPLITS):
            continue

        assignment = {
            speaker_id: split
            for split in SPLITS
            for speaker_id in sorted(chosen[split], key=int)
        }
        if not coverage_gaps(profiles, assignment):
            return assignment
    raise RuntimeError(
        "Could not construct a coverage-complete assignment. Increase attempts "
        "or inspect which accent-emotion cells have too few speakers."
    )


def _feature_vector(profile: dict[str, Any]) -> dict[str, int]:
    vector = {"samples": profile["samples"]}
    for emotion in EMOTIONS:
        vector[f"emotion:{emotion}"] = profile["emotions"][emotion]
    for accent in ACCENTS:
        vector[f"accent_samples:{accent}"] = (
            profile["samples"] if profile["accent"] == accent else 0
        )
        vector[f"accent_speakers:{accent}"] = int(profile["accent"] == accent)
        for emotion in EMOTIONS:
            count = profile["emotions"][emotion] if profile["accent"] == accent else 0
            vector[f"cell_samples:{accent}:{emotion}"] = count
            vector[f"cell_speakers:{accent}:{emotion}"] = int(count > 0)
    for gender in ("female", "male"):
        vector[f"gender_samples:{gender}"] = (
            profile["samples"] if profile["gender"] == gender else 0
        )
        vector[f"gender_speakers:{gender}"] = int(profile["gender"] == gender)
    return vector


def _feature_weight(name: str) -> float:
    if name == "samples":
        return 60.0
    if name.startswith("emotion:"):
        return 5.0
    if name.startswith("accent_samples:"):
        return 4.0
    if name.startswith("accent_speakers:"):
        return 2.0
    if name.startswith("cell_samples:"):
        return 8.0
    if name.startswith("cell_speakers:"):
        return 4.0
    if name.startswith("gender_samples:"):
        return 1.0
    return 0.5


def optimize_assignment_v2(
    profiles: dict[str, dict[str, Any]],
    split_counts: dict[str, int],
    seed: int,
    restarts: int,
    steps: int,
    forced_train_speaker: str,
) -> tuple[dict[str, str], float]:
    """Balance marginals and cross-strata while preserving hard coverage."""
    vectors = {speaker_id: _feature_vector(profile) for speaker_id, profile in profiles.items()}
    feature_names = list(next(iter(vectors.values())))
    global_totals = {
        feature: sum(vector[feature] for vector in vectors.values())
        for feature in feature_names
    }

    def totals_for(assignment: dict[str, str]) -> dict[str, dict[str, int]]:
        totals = {split: {feature: 0 for feature in feature_names} for split in SPLITS}
        for speaker_id, split in assignment.items():
            for feature, value in vectors[speaker_id].items():
                totals[split][feature] += value
        return totals

    def score(totals: dict[str, dict[str, int]]) -> float:
        result = 0.0
        for split in SPLITS:
            for feature in feature_names:
                total = global_totals[feature]
                if total:
                    difference = totals[split][feature] / total - TARGET_FRACTIONS[split]
                    result += _feature_weight(feature) * difference * difference
            for accent in ACCENTS:
                for emotion in EMOTIONS:
                    if totals[split][f"cell_speakers:{accent}:{emotion}"] == 0:
                        result += COVERAGE_PENALTY
        return result

    best_assignment: dict[str, str] | None = None
    best_score = math.inf
    for restart in range(restarts):
        assignment = construct_feasible_assignment(
            profiles,
            split_counts,
            seed + restart * 1_000_003,
            forced_train_speaker,
        )
        totals = totals_for(assignment)
        current_score = score(totals)
        rng = random.Random(seed + restart * 97_409)
        by_split = {
            split: [speaker_id for speaker_id, value in assignment.items() if value == split]
            for split in SPLITS
        }

        if current_score < best_score:
            best_assignment, best_score = dict(assignment), current_score

        for step in range(steps):
            split_a, split_b = rng.sample(SPLITS, 2)
            candidates_a = [
                speaker_id
                for speaker_id in by_split[split_a]
                if speaker_id != forced_train_speaker
            ]
            if not candidates_a:
                continue
            speaker_a = rng.choice(candidates_a)
            speaker_b = rng.choice(by_split[split_b])
            for feature in feature_names:
                totals[split_a][feature] += vectors[speaker_b][feature] - vectors[speaker_a][feature]
                totals[split_b][feature] += vectors[speaker_a][feature] - vectors[speaker_b][feature]
            new_score = score(totals)
            temperature = 0.01 * (1.0 - step / max(steps, 1)) + 1e-8
            accept = new_score <= current_score or rng.random() < math.exp(
                (current_score - new_score) / temperature
            )
            if accept:
                assignment[speaker_a], assignment[speaker_b] = split_b, split_a
                by_split[split_a].remove(speaker_a)
                by_split[split_a].append(speaker_b)
                by_split[split_b].remove(speaker_b)
                by_split[split_b].append(speaker_a)
                current_score = new_score
                if current_score < best_score:
                    best_assignment, best_score = dict(assignment), current_score
            else:
                for feature in feature_names:
                    totals[split_a][feature] += vectors[speaker_a][feature] - vectors[speaker_b][feature]
                    totals[split_b][feature] += vectors[speaker_b][feature] - vectors[speaker_a][feature]

    if best_assignment is None or coverage_gaps(profiles, best_assignment):
        raise RuntimeError("The optimizer did not produce a coverage-complete assignment")
    return best_assignment, best_score


def apply_assignment_v2(
    rows: list[dict[str, str]], assignment: dict[str, str], seed: int
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row: dict[str, Any] = dict(source)
        eligible = _is_true(row["is_eligible_for_split"])
        row.update(
            {
                "split": assignment[row["speaker_id"]] if eligible else "excluded",
                "split_version": SPLIT_VERSION,
                "split_seed": seed,
                "test_membership_frozen": eligible,
            }
        )
        output.append(row)
    return output


def build_report_v2(
    rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    objective_score: float,
    seed: int,
    forced_train_speaker: str,
    restarts: int,
    steps: int,
) -> dict[str, Any]:
    eligible = [row for row in rows if row["split"] in SPLITS]
    split_speakers = {
        split: sorted(
            {row["speaker_id"] for row in eligible if row["split"] == split}, key=int
        )
        for split in SPLITS
    }
    cross_strata = []
    for split in SPLITS:
        for accent in ACCENTS:
            for emotion in EMOTIONS:
                members = [
                    row
                    for row in eligible
                    if row["split"] == split
                    and row["accent_clean"] == accent
                    and row["emotion"] == emotion
                ]
                cross_strata.append(
                    {
                        "split": split,
                        "accent": accent,
                        "emotion": emotion,
                        "samples": len(members),
                        "speakers": len({row["speaker_id"] for row in members}),
                    }
                )

    assignment_text = "\n".join(
        f"{speaker_id}={assignment[speaker_id]}" for speaker_id in sorted(assignment, key=int)
    )
    eligible_hashes = [row["audio_sha256"] for row in eligible]
    overlap = {
        f"{left}_{right}": sorted(set(split_speakers[left]) & set(split_speakers[right]), key=int)
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    validation = {
        "speaker_overlap_zero": all(not values for values in overlap.values()),
        "eligible_audio_hashes_unique": len(eligible_hashes) == len(set(eligible_hashes)),
        "speaker_counts_match_plan": all(
            len(split_speakers[split]) == SPEAKER_COUNTS[split] for split in SPLITS
        ),
        "all_cross_strata_have_samples": all(item["samples"] > 0 for item in cross_strata),
        "all_cross_strata_have_speakers": all(item["speakers"] > 0 for item in cross_strata),
        "excluded_rows_not_assigned": all(
            row["split"] == "excluded"
            for row in rows
            if not _is_true(row["is_eligible_for_split"])
        ),
        "dominant_speaker_forced_to_train": assignment[forced_train_speaker] == "train",
    }
    if not all(validation.values()):
        raise ValueError(f"Split v2 validation failed: {validation}")

    distributions = {}
    for split in SPLITS:
        members = [row for row in eligible if row["split"] == split]
        distributions[split] = {
            "samples": len(members),
            "sample_fraction": len(members) / len(eligible),
            "speakers": len(split_speakers[split]),
            "accent_samples": dict(sorted(Counter(row["accent_clean"] for row in members).items())),
            "emotion_samples": dict(sorted(Counter(row["emotion"] for row in members).items())),
            "gender_samples": dict(sorted(Counter(row["gender_clean"] for row in members).items())),
        }

    return {
        "split_version": SPLIT_VERSION,
        "status": "FROZEN",
        "supersedes": "speaker_disjoint_v1",
        "supersession_reason": (
            "Version 1 had no Central-Sad test sample. Version 2 requires complete "
            "split x accent x emotion coverage."
        ),
        "seed": seed,
        "target_sample_fractions": TARGET_FRACTIONS,
        "planned_speaker_counts": SPEAKER_COUNTS,
        "optimizer": {
            "restarts": restarts,
            "steps_per_restart": steps,
            "objective_score": objective_score,
        },
        "assignment_sha256": hashlib.sha256(assignment_text.encode("utf-8")).hexdigest(),
        "distributions": distributions,
        "cross_strata": cross_strata,
        "speaker_overlap": overlap,
        "validation": validation,
        "dominant_speaker": {
            "speaker_id": forced_train_speaker,
            "samples": profiles[forced_train_speaker]["samples"],
            "assigned_split": assignment[forced_train_speaker],
        },
        "freeze_rule": (
            "Do not overwrite split v1. Use the versioned v2 files for all models only "
            "after reviewing this report; any later change requires another version."
        ),
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
    parser.add_argument(
        "--output", type=Path, default=repo_root / "data/manifests/split_manifest_v2.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=repo_root / "data/manifests/split_report_v2.json"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.input.resolve())
    profiles = build_speaker_profiles(rows)
    forced_train_speaker = max(profiles, key=lambda speaker_id: profiles[speaker_id]["samples"])
    assignment, objective_score = optimize_assignment_v2(
        profiles,
        SPEAKER_COUNTS,
        args.seed,
        args.restarts,
        args.steps,
        forced_train_speaker,
    )
    output = apply_assignment_v2(rows, assignment, args.seed)
    report = build_report_v2(
        output,
        profiles,
        assignment,
        objective_score,
        args.seed,
        forced_train_speaker,
        args.restarts,
        args.steps,
    )
    write_csv(output, args.output.resolve())
    write_json(report, args.report.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "split_version": report["split_version"],
                "assignment_sha256": report["assignment_sha256"],
                "cross_strata_complete": report["validation"]["all_cross_strata_have_samples"],
                "output": str(args.output.resolve()),
                "report": str(args.report.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
