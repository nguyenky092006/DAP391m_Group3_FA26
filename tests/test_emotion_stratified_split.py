import unittest

import pandas as pd

from fedecai.data.create_emotion_stratified_split import build_split, validate_split


class EmotionStratifiedSplitTests(unittest.TestCase):
    def make_frame(self):
        rows = []
        accents = ("central", "north", "south")
        emotions = ("angry", "happy", "neutral", "sad")
        for index in range(80):
            rows.append({
                "source_row_index": index,
                "sample_id": f"sample_{index:03d}",
                "speaker_id": str(index % 8),
                "emotion": emotions[index % 4],
                "accent_clean": accents[index % 3],
                "audio_sha256": f"hash_{index:03d}",
                "audio_locator": f"audio_{index:03d}.wav",
                "is_eligible_for_split": True,
            })
        rows.append({
            "source_row_index": 80,
            "sample_id": "excluded",
            "speaker_id": "0",
            "emotion": "angry",
            "accent_clean": "south",
            "audio_sha256": "excluded_hash",
            "audio_locator": "excluded.wav",
            "is_eligible_for_split": False,
        })
        return pd.DataFrame(rows)

    def build(self):
        return build_split(
            self.make_frame(), seed=391, train_count=56,
            validation_count=12, test_count=12, cv_folds=5,
        )

    def test_split_is_deterministic_and_sample_level(self):
        first = self.build()
        second = self.build()
        self.assertEqual(first["split"].tolist(), second["split"].tolist())
        self.assertEqual(first["cv_fold"].astype(str).tolist(), second["cv_fold"].astype(str).tolist())
        self.assertEqual(first["split"].value_counts().to_dict(), {
            "train": 56, "validation": 12, "test": 12, "excluded": 1,
        })
        # The same synthetic speaker normally occurs across splits and is accepted.
        speaker_splits = first[first["split"].ne("excluded")].groupby("speaker_id")["split"].nunique()
        self.assertTrue(speaker_splits.gt(1).any())

    def test_emotion_stratification_and_cv(self):
        frame = self.build()
        eligible = frame[frame["split"].ne("excluded")]
        for split in ("train", "validation", "test"):
            self.assertEqual(set(eligible.loc[eligible["split"].eq(split), "emotion"]), {
                "angry", "happy", "neutral", "sad",
            })
        self.assertTrue(frame.loc[frame["split"].eq("train"), "cv_fold"].notna().all())
        self.assertTrue(frame.loc[~frame["split"].eq("train"), "cv_fold"].isna().all())

    def test_duplicate_waveform_is_rejected(self):
        source = self.make_frame()
        source.loc[1, "audio_sha256"] = source.loc[0, "audio_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate waveform"):
            build_split(source, seed=391, train_count=56, validation_count=12, test_count=12)

    def test_real_contract_validation_accepts_speaker_overlap(self):
        source = self.make_frame().iloc[:-1].copy()
        # Use default validation logic by repeating to the real target size.
        expanded = pd.concat([source.assign(
            source_row_index=block * len(source) + source["source_row_index"],
            sample_id=source["sample_id"] + f"_{block}",
            audio_sha256=source["audio_sha256"] + f"_{block}",
            audio_locator=source["audio_locator"] + f"_{block}",
        ) for block in range(63)], ignore_index=True).iloc[:4992]
        frame = build_split(expanded)
        report = validate_split(frame)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["speaker_overlap_status"], "INFO")
        self.assertEqual(report["duplicate_waveform_cross_split"], 0)


if __name__ == "__main__":
    unittest.main()
