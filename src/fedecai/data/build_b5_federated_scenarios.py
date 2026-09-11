"""Build versioned main and stress federated scenarios without changing B5 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ACCENTS = ("central", "north", "south")
EMOTIONS = ("angry", "happy", "neutral", "sad")
MAIN_VERSION = "main_mixed_accent_v1"
STRESS_VERSION = "stress_accent_domain_v1"


def speaker_profiles(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    train = frame[frame["split"] == "train"]
    profiles: dict[str, dict[str, Any]] = {}
    for speaker, rows in train.groupby("speaker_id", sort=False):
        accents = set(rows["accent_clean"])
        if len(accents) != 1:
            raise ValueError(f"Training speaker {speaker} has multiple clean accents")
        profiles[str(speaker)] = {
            "samples": int(len(rows)),
            "accent": str(next(iter(accents))),
            "emotions": Counter(rows["emotion"]),
        }
    return profiles


def assign_main_clients(profiles: dict[str, dict[str, Any]], clients: int = 3) -> dict[str, str]:
    """Create deterministic mixed-accent clients with moderated accent skew."""
    names = [f"main_client_{index:02d}" for index in range(clients)]
    assignment: dict[str, str] = {}
    totals = Counter({name: 0 for name in names})
    accent_totals = {
        name: Counter({accent: 0 for accent in ACCENTS}) for name in names
    }
    population = Counter()
    for profile in profiles.values():
        population[profile["accent"]] += profile["samples"]
    population_total = sum(population.values())
    target_share = {accent: population[accent] / population_total for accent in ACCENTS}

    # Seed every client with one speaker from every accent. Large speakers are
    # distributed first and the rotation prevents one client receiving all
    # first-ranked accent speakers.
    for accent_index, accent in enumerate(ACCENTS):
        speakers = sorted(
            (speaker for speaker, value in profiles.items() if value["accent"] == accent),
            key=lambda speaker: (-profiles[speaker]["samples"], int(speaker)),
        )
        if len(speakers) < clients:
            raise ValueError(f"Need {clients} {accent} speakers for main clients; found {len(speakers)}")
        for offset, speaker in enumerate(speakers[:clients]):
            client = names[(offset + accent_index * 2) % clients]
            assignment[speaker] = client
            totals[client] += profiles[speaker]["samples"]
            accent_totals[client][accent] += profiles[speaker]["samples"]

    remaining = sorted(
        (speaker for speaker in profiles if speaker not in assignment),
        key=lambda speaker: (-profiles[speaker]["samples"], int(speaker)),
    )
    for speaker in remaining:
        accent = profiles[speaker]["accent"]
        samples = profiles[speaker]["samples"]

        def candidate_score(candidate: str) -> tuple[float, float, str]:
            divergences = []
            projected_totals = dict(totals)
            projected_totals[candidate] += samples
            for name in names:
                total = projected_totals[name]
                if not total:
                    continue
                divergence = 0.0
                for value in ACCENTS:
                    count = accent_totals[name][value]
                    if name == candidate and value == accent:
                        count += samples
                    divergence += abs(count / total - target_share[value])
                divergences.append(divergence)
            accent_divergence = sum(divergences) / len(divergences)
            load_range = (max(projected_totals.values()) - min(projected_totals.values())) / population_total
            return accent_divergence + 0.05 * load_range, totals[candidate], candidate

        client = min(names, key=candidate_score)
        assignment[speaker] = client
        totals[client] += samples
        accent_totals[client][accent] += samples
    return assignment


def assign_stress_clients(profiles: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {speaker: f"stress_client_{value['accent']}" for speaker, value in profiles.items()}


def scenario_summary(
    profiles: dict[str, dict[str, Any]], assignment: dict[str, str], version: str
) -> dict[str, Any]:
    clients = []
    for client in sorted(set(assignment.values())):
        speakers = [speaker for speaker, target in assignment.items() if target == client]
        accent_samples = Counter()
        emotion_samples = Counter()
        for speaker in speakers:
            profile = profiles[speaker]
            accent_samples[profile["accent"]] += profile["samples"]
            emotion_samples.update(profile["emotions"])
        sample_total = int(sum(accent_samples.values()))
        clients.append({
            "client_id": client,
            "samples": sample_total,
            "sample_share": sample_total / sum(value["samples"] for value in profiles.values()),
            "speakers": len(speakers),
            "accent_samples": {key: int(accent_samples[key]) for key in ACCENTS},
            "max_accent_share": max(accent_samples.values()) / sample_total,
            "emotion_samples": {key: int(emotion_samples[key]) for key in EMOTIONS},
        })
    assignment_text = "\n".join(f"{speaker}={assignment[speaker]}" for speaker in sorted(assignment, key=int))
    validation = {
        "all_train_speakers_assigned_once": set(assignment) == set(profiles),
        "all_clients_have_multiple_speakers": all(row["speakers"] >= 2 for row in clients),
        "all_clients_have_all_emotions": all(all(row["emotion_samples"][emotion] > 0 for emotion in EMOTIONS) for row in clients),
        "main_clients_have_all_accents": version != MAIN_VERSION or all(all(row["accent_samples"][accent] > 0 for accent in ACCENTS) for row in clients),
        "main_max_single_accent_share_le_0_85": version != MAIN_VERSION or all(row["max_accent_share"] <= 0.85 for row in clients),
        "stress_clients_are_single_accent": version != STRESS_VERSION or all(sum(value > 0 for value in row["accent_samples"].values()) == 1 for row in clients),
    }
    if not all(validation.values()):
        raise ValueError(f"Invalid {version} federated scenario: {validation}")
    return {
        "version": version,
        "assignment_sha256": hashlib.sha256(assignment_text.encode("utf-8")).hexdigest(),
        "clients": clients,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/split_manifest_v2_enriched.csv")
    parser.add_argument("--main-clients", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = ROOT / args.manifest
    frame = pd.read_csv(manifest_path, dtype={"speaker_id": "string"})
    profiles = speaker_profiles(frame)
    main_assignment = assign_main_clients(profiles, args.main_clients)
    stress_assignment = assign_stress_clients(profiles)
    main_report = scenario_summary(profiles, main_assignment, MAIN_VERSION)
    stress_report = scenario_summary(profiles, stress_assignment, STRESS_VERSION)

    rows = []
    for speaker in sorted(profiles, key=int):
        rows.append({
            "speaker_id": speaker,
            "split_version": "speaker_disjoint_v2",
            "main_partition_version": MAIN_VERSION,
            "main_client_id": main_assignment[speaker],
            "stress_partition_version": STRESS_VERSION,
            "stress_client_id": stress_assignment[speaker],
            "accent": profiles[speaker]["accent"],
            "samples": profiles[speaker]["samples"],
        })
    report = {
        "status": "PASS",
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_samples": int((frame["split"] == "train").sum()),
        "train_speakers": len(profiles),
        "global_split_modified": False,
        "main": main_report,
        "stress": stress_report,
    }
    output_dir = ROOT / "data/manifests"
    report_dir = ROOT / "reports/tables/b5"
    assignment_path = output_dir / "federated_speaker_partitions_v1.csv"
    summary_path = report_dir / "b5_federated_scenarios_v1.json"
    table_path = report_dir / "b5_federated_scenarios_v1.csv"
    pd.DataFrame(rows).to_csv(assignment_path, index=False)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = []
    for scenario in (main_report, stress_report):
        for client in scenario["clients"]:
            flat.append({"scenario": scenario["version"], **client})
    pd.json_normalize(flat, sep="_").to_csv(table_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
