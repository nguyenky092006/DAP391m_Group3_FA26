"""Build deterministic grouped-CV and federated partitions for B5 split v2."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any


ACCENTS = ("central", "north", "south")
EMOTIONS = ("angry", "happy", "neutral", "sad")
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_ATTEMPTS = 512


def _speaker_cells(profile: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (profile["accent"], emotion)
        for emotion in EMOTIONS
        if profile["emotions"][emotion] > 0
    }


def _speaker_marginals(profile: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        ("accent", profile["accent"]),
        *{
            ("emotion", emotion)
            for emotion in EMOTIONS
            if profile["emotions"][emotion] > 0
        },
    }


def _fold_counts(number_of_speakers: int, folds: int) -> dict[str, int]:
    base, remainder = divmod(number_of_speakers, folds)
    return {
        f"fold_{index}": base + (1 if index < remainder else 0)
        for index in range(folds)
    }


def _coverage_complete(
    profiles: dict[str, dict[str, Any]], assignment: dict[str, str], fold: str
) -> bool:
    observed = {
        label
        for speaker_id, assigned_fold in assignment.items()
        if assigned_fold == fold
        for label in _speaker_marginals(profiles[speaker_id])
    }
    required = {
        *(('accent', accent) for accent in ACCENTS),
        *(('emotion', emotion) for emotion in EMOTIONS),
    }
    return required <= observed


def _choose_cover(
    profiles: dict[str, dict[str, Any]],
    remaining: set[str],
    capacity: int,
    rng: random.Random,
) -> set[str] | None:
    uncovered = {
        *(('accent', accent) for accent in ACCENTS),
        *(('emotion', emotion) for emotion in EMOTIONS),
    }
    selected: set[str] = set()
    while uncovered:
        candidates = []
        for speaker_id in sorted(remaining - selected, key=int):
            gain = len(_speaker_marginals(profiles[speaker_id]) & uncovered)
            if gain:
                candidates.append((gain, speaker_id))
        if not candidates:
            return None
        best_gain = max(gain for gain, _ in candidates)
        tied = sorted(
            (speaker_id for gain, speaker_id in candidates if gain == best_gain), key=int
        )
        rng.shuffle(tied)
        tied.sort(key=lambda speaker_id: profiles[speaker_id]["samples"])
        selected.add(tied[0])
        uncovered -= _speaker_marginals(profiles[tied[0]])
        if len(selected) > capacity:
            return None
    return selected


def _balance_score(
    profiles: dict[str, dict[str, Any]], assignment: dict[str, str], folds: tuple[str, ...]
) -> float:
    totals = {
        "samples": sum(profile["samples"] for profile in profiles.values()),
        **{
            f"emotion:{emotion}": sum(
                profile["emotions"][emotion] for profile in profiles.values()
            )
            for emotion in EMOTIONS
        },
        **{
            f"accent:{accent}": sum(
                profile["samples"]
                for profile in profiles.values()
                if profile["accent"] == accent
            )
            for accent in ACCENTS
        },
    }
    score = 0.0
    target = 1.0 / len(folds)
    for fold in folds:
        members = [profiles[speaker_id] for speaker_id, value in assignment.items() if value == fold]
        observed = {
            "samples": sum(profile["samples"] for profile in members),
            **{
                f"emotion:{emotion}": sum(profile["emotions"][emotion] for profile in members)
                for emotion in EMOTIONS
            },
            **{
                f"accent:{accent}": sum(
                    profile["samples"] for profile in members if profile["accent"] == accent
                )
                for accent in ACCENTS
            },
        }
        for feature, total in totals.items():
            if total:
                weight = 40.0 if feature == "samples" else 4.0
                score += weight * (observed[feature] / total - target) ** 2
    return score


def assign_grouped_cv(
    profiles: dict[str, dict[str, Any]],
    seed: int,
    folds: int = DEFAULT_CV_FOLDS,
    attempts: int = DEFAULT_CV_ATTEMPTS,
) -> tuple[dict[str, str], float]:
    """Assign whole training speakers to balanced CV folds with marginal coverage."""
    if folds < 2:
        raise ValueError("Grouped CV requires at least two folds")
    required_labels = {
        *(('accent', accent) for accent in ACCENTS),
        *(('emotion', emotion) for emotion in EMOTIONS),
    }
    for label in sorted(required_labels):
        available = sum(label in _speaker_marginals(profile) for profile in profiles.values())
        if available < folds:
            raise ValueError(
                f"Cannot give every CV fold {label[0]}={label[1]} coverage: "
                f"only {available} speakers are available"
            )

    capacities = _fold_counts(len(profiles), folds)
    fold_names = tuple(capacities)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    all_speakers = set(profiles)

    for attempt in range(attempts):
        rng = random.Random(seed + attempt * 104_729)
        remaining = set(all_speakers)
        selected: dict[str, set[str]] = {}
        order = list(fold_names)
        rng.shuffle(order)
        failed = False
        for fold in order:
            covered = _choose_cover(profiles, remaining, capacities[fold], rng)
            if covered is None:
                failed = True
                break
            selected[fold] = covered
            remaining -= covered
        if failed:
            continue

        assignment = {
            speaker_id: fold
            for fold in fold_names
            for speaker_id in sorted(selected[fold], key=int)
        }
        leftovers = sorted(remaining, key=int)
        rng.shuffle(leftovers)
        leftovers.sort(key=lambda speaker_id: profiles[speaker_id]["samples"], reverse=True)
        sample_totals = {
            fold: sum(profiles[speaker_id]["samples"] for speaker_id in selected[fold])
            for fold in fold_names
        }
        for speaker_id in leftovers:
            candidates = [
                fold
                for fold in fold_names
                if sum(value == fold for value in assignment.values()) < capacities[fold]
            ]
            rng.shuffle(candidates)
            candidates.sort(
                key=lambda fold: sample_totals[fold] / max(capacities[fold], 1)
            )
            fold = candidates[0]
            assignment[speaker_id] = fold
            sample_totals[fold] += profiles[speaker_id]["samples"]

        if not all(_coverage_complete(profiles, assignment, fold) for fold in fold_names):
            continue
        score = _balance_score(profiles, assignment, fold_names)
        if score < best_score:
            best_assignment, best_score = dict(assignment), score

    if best_assignment is None:
        raise RuntimeError("Could not construct marginal-coverage-complete grouped CV folds")
    return best_assignment, best_score


def assign_federated_clients(
    profiles: dict[str, dict[str, Any]], assignment: dict[str, str]
) -> dict[str, str]:
    """Create accent-domain clients from training speakers only."""
    return {
        speaker_id: f"client_{profiles[speaker_id]['accent']}"
        for speaker_id, split in assignment.items()
        if split == "train"
    }


def build_partition_report(
    rows: list[dict[str, Any]],
    cv_assignment: dict[str, str],
    client_assignment: dict[str, str],
    cv_score: float,
    seed: int,
    expected_cv_folds: int = DEFAULT_CV_FOLDS,
) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    train_speakers = {row["speaker_id"] for row in train_rows}
    fold_names = tuple(sorted(set(cv_assignment.values())))
    client_names = tuple(sorted(set(client_assignment.values())))
    cv_missing_cross_strata = [
        {"fold": fold, "accent": accent, "emotion": emotion}
        for fold in fold_names
        for accent in ACCENTS
        for emotion in EMOTIONS
        if not any(
            cv_assignment[row["speaker_id"]] == fold
            and row["accent_clean"] == accent
            and row["emotion"] == emotion
            for row in train_rows
        )
    ]

    validation = {
        "cv_train_speakers_covered_once": set(cv_assignment) == train_speakers,
        "cv_fold_count": len(fold_names) == expected_cv_folds,
        "cv_marginal_coverage_complete": all(
            all(any(row["accent_clean"] == accent for row in members) for accent in ACCENTS)
            and all(any(row["emotion"] == emotion for row in members) for emotion in EMOTIONS)
            for fold in fold_names
            for members in [[
                row
                for row in train_rows
                if cv_assignment[row["speaker_id"]] == fold
            ]]
        ),
        "clients_train_speakers_covered_once": set(client_assignment) == train_speakers,
        "clients_are_accent_aligned": all(
            client_assignment[row["speaker_id"]] == f"client_{row['accent_clean']}"
            for row in train_rows
        ),
        "each_client_has_all_emotions": all(
            all(
                any(
                    client_assignment[row["speaker_id"]] == client
                    and row["emotion"] == emotion
                    for row in train_rows
                )
                for emotion in EMOTIONS
            )
            for client in client_names
        ),
    }
    if not all(validation.values()):
        raise ValueError(f"B5 partition validation failed: {validation}")

    cv_folds = []
    for fold in fold_names:
        members = [row for row in train_rows if cv_assignment[row["speaker_id"]] == fold]
        cv_folds.append(
            {
                "fold": fold,
                "samples": len(members),
                "speakers": len({row["speaker_id"] for row in members}),
                "accent_samples": dict(sorted(Counter(row["accent_clean"] for row in members).items())),
                "emotion_samples": dict(sorted(Counter(row["emotion"] for row in members).items())),
            }
        )
    clients = []
    for client in client_names:
        members = [row for row in train_rows if client_assignment[row["speaker_id"]] == client]
        clients.append(
            {
                "client_id": client,
                "samples": len(members),
                "speakers": len({row["speaker_id"] for row in members}),
                "emotion_samples": dict(sorted(Counter(row["emotion"] for row in members).items())),
            }
        )
    assignment_text = "\n".join(
        f"{speaker_id}={cv_assignment[speaker_id]}"
        for speaker_id in sorted(cv_assignment, key=int)
    )
    return {
        "cv": {
            "method": (
                f"speaker-grouped {expected_cv_folds}-fold cross-validation on "
                "split-v2 training speakers only"
            ),
            "seed": seed,
            "balance_score": cv_score,
            "assignment_sha256": hashlib.sha256(assignment_text.encode("utf-8")).hexdigest(),
            "coverage_policy": (
                "Every fold must contain every accent and every emotion. Full "
                "accent-emotion cross coverage is reported but not required because "
                "some training cross-strata contain fewer speakers than CV folds."
            ),
            "cross_strata_complete": not cv_missing_cross_strata,
            "missing_cross_strata": cv_missing_cross_strata,
            "folds": cv_folds,
        },
        "federated_clients": {
            "method": "non-IID accent-domain clients on split-v2 training speakers only",
            "clients": clients,
        },
        "validation": validation,
    }
