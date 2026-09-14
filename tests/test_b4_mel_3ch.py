import unittest

import numpy as np

from fedecai.features.mel_3ch import build_mel_3ch, pad_or_crop_mel_3ch


class TestB4Mel3Channel(unittest.TestCase):
    def test_fixed_normalization_and_channel_repeat(self):
        source = np.linspace(-100.0, 5.0, 80 * 7, dtype=np.float32).reshape(80, 7)
        result = build_mel_3ch(source)
        self.assertEqual(result.shape, (3, 80, 7))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[1], result[2])

    def test_shared_padding_and_center_crop(self):
        source = np.arange(3 * 80 * 5, dtype=np.float32).reshape(3, 80, 5)
        cropped = pad_or_crop_mel_3ch(source, 3)
        np.testing.assert_array_equal(cropped, source[:, :, 1:4])
        padded = pad_or_crop_mel_3ch(source, 8)
        self.assertEqual(padded.shape, (3, 80, 8))
        np.testing.assert_array_equal(padded[:, :, :5], source)
        np.testing.assert_array_equal(
            padded[:, :, 5:], np.zeros((3, 80, 3), dtype=np.float32)
        )

    def test_invalid_mode_and_shape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mel_3ch_mode"):
            build_mel_3ch(np.zeros((80, 4), dtype=np.float32), mode="rgb_colormap")
        with self.assertRaisesRegex(ValueError, "Expected log-Mel shape"):
            build_mel_3ch(np.zeros((79, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
