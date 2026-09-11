import unittest
from collections import Counter

from fedecai.data.build_b5_federated_scenarios import (
    MAIN_VERSION, STRESS_VERSION, assign_main_clients, assign_stress_clients,
    scenario_summary,
)


class B5FederatedScenarioTests(unittest.TestCase):
    def setUp(self):
        self.profiles = {}
        speaker = 0
        for accent in ("central", "north", "south"):
            for offset in range(8):
                self.profiles[str(speaker)] = {
                    "samples": 10 + offset,
                    "accent": accent,
                    "emotions": Counter({emotion: 2 + offset for emotion in ("angry", "happy", "neutral", "sad")}),
                }
                speaker += 1

    def test_main_is_deterministic_mixed_and_complete(self):
        first = assign_main_clients(self.profiles, clients=3)
        self.assertEqual(first, assign_main_clients(self.profiles, clients=3))
        report = scenario_summary(self.profiles, first, MAIN_VERSION)
        self.assertEqual(len(report["clients"]), 3)
        self.assertTrue(all(report["validation"].values()))

    def test_stress_is_accent_domain_and_complete(self):
        assignment = assign_stress_clients(self.profiles)
        report = scenario_summary(self.profiles, assignment, STRESS_VERSION)
        self.assertEqual(len(report["clients"]), 3)
        self.assertTrue(all(report["validation"].values()))


if __name__ == "__main__":
    unittest.main()
