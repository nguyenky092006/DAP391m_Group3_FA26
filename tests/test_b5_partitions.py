import unittest
from collections import Counter

from fedecai.data.build_b5_partitions import (
    ACCENTS,
    EMOTIONS,
    assign_federated_clients,
    assign_grouped_cv,
)


class B5PartitionTests(unittest.TestCase):
    def setUp(self):
        self.profiles = {}
        speaker = 0
        for accent in ACCENTS:
            for offset in range(5):
                self.profiles[str(speaker)] = {
                    "samples": 12 + offset,
                    "accent": accent,
                    "gender": "female" if speaker % 2 else "male",
                    "emotions": Counter({emotion: 3 + (offset % 2) for emotion in EMOTIONS}),
                }
                speaker += 1

    def test_grouped_cv_is_deterministic_disjoint_and_marginally_complete(self):
        first, first_score = assign_grouped_cv(self.profiles, seed=391, folds=5, attempts=20)
        second, second_score = assign_grouped_cv(self.profiles, seed=391, folds=5, attempts=20)
        self.assertEqual(first, second)
        self.assertEqual(first_score, second_score)
        self.assertEqual(set(first), set(self.profiles))
        self.assertEqual(Counter(first.values()), Counter({f"fold_{i}": 3 for i in range(5)}))
        for fold in set(first.values()):
            members = [self.profiles[speaker] for speaker, value in first.items() if value == fold]
            self.assertEqual({profile["accent"] for profile in members}, set(ACCENTS))
            for emotion in EMOTIONS:
                self.assertTrue(any(profile["emotions"][emotion] for profile in members))

    def test_grouped_cv_allows_a_cross_stratum_with_fewer_speakers_than_folds(self):
        # Mirrors the real v2 train set: Central-Sad has only four speakers, so
        # full cross-stratum coverage in five folds is mathematically impossible.
        self.profiles["4"]["emotions"]["sad"] = 0
        assignment, _ = assign_grouped_cv(self.profiles, seed=391, folds=5, attempts=20)
        self.assertEqual(len(set(assignment.values())), 5)
        self.assertEqual(set(assignment), set(self.profiles))

    def test_federated_clients_only_use_training_speakers(self):
        split = {
            speaker: ("train" if int(speaker) < 10 else "validation")
            for speaker in self.profiles
        }
        clients = assign_federated_clients(self.profiles, split)
        self.assertEqual(set(clients), {str(index) for index in range(10)})
        for speaker, client in clients.items():
            self.assertEqual(client, f"client_{self.profiles[speaker]['accent']}")


if __name__ == "__main__":
    unittest.main()
