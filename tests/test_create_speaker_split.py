import unittest

from fedecai.data.create_speaker_split import apply_assignment, build_speaker_profiles


class TestSpeakerSplit(unittest.TestCase):
    def test_profile_rejects_multiple_clean_accents(self):
        rows = [
            {"is_eligible_for_split": "True", "speaker_id": "1", "accent_clean": "north", "gender_clean": "male", "emotion": "happy"},
            {"is_eligible_for_split": "True", "speaker_id": "1", "accent_clean": "south", "gender_clean": "male", "emotion": "sad"},
        ]
        with self.assertRaises(ValueError):
            build_speaker_profiles(rows)

    def test_excluded_row_never_receives_model_split(self):
        rows = [
            {"is_eligible_for_split": "True", "speaker_id": "1"},
            {"is_eligible_for_split": "False", "speaker_id": "1"},
        ]
        output = apply_assignment(rows, {"1": "train"}, 391)
        self.assertEqual(output[0]["split"], "train")
        self.assertEqual(output[1]["split"], "excluded")


if __name__ == "__main__":
    unittest.main()

