import unittest
from collections import Counter

from fedecai.data.create_speaker_split_v2 import (
    ACCENTS,
    EMOTIONS,
    apply_assignment_v2,
    construct_feasible_assignment,
    coverage_gaps,
)


def profile(accent, samples=4):
    return {
        "samples": samples,
        "accent": accent,
        "gender": "female",
        "emotions": Counter({emotion: 1 for emotion in EMOTIONS}),
    }


class TestSpeakerSplitV2(unittest.TestCase):
    def test_constructed_assignment_has_complete_cross_strata(self):
        profiles = {
            str(index): profile(accent)
            for index, accent in enumerate(
                ["central", "north", "south"] * 4
            )
        }
        assignment = construct_feasible_assignment(
            profiles,
            {"train": 6, "validation": 3, "test": 3},
            seed=391,
            forced_train_speaker="0",
            attempts=100,
        )

        self.assertEqual(Counter(assignment.values()), Counter({"train": 6, "validation": 3, "test": 3}))
        self.assertEqual(assignment["0"], "train")
        self.assertEqual(coverage_gaps(profiles, assignment), [])

    def test_excluded_rows_remain_excluded_and_v1_is_not_reused(self):
        rows = [
            {
                "sample_id": "eligible",
                "speaker_id": "1",
                "is_eligible_for_split": "True",
            },
            {
                "sample_id": "excluded",
                "speaker_id": "2",
                "is_eligible_for_split": "False",
            },
        ]
        output = apply_assignment_v2(rows, {"1": "test"}, seed=391)

        self.assertEqual(output[0]["split"], "test")
        self.assertEqual(output[0]["split_version"], "speaker_disjoint_v2")
        self.assertTrue(output[0]["test_membership_frozen"])
        self.assertEqual(output[1]["split"], "excluded")
        self.assertFalse(output[1]["test_membership_frozen"])


if __name__ == "__main__":
    unittest.main()
