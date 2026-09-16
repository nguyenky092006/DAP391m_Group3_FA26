import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RegressionControlContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/models/b6_regression_controls_v1.json").read_text(encoding="utf-8"))

    def test_wav2vec2_changes_only_budget_and_scheduler_horizon(self):
        old = json.loads((ROOT / "configs/models/b6_performance_v1.json").read_text(encoding="utf-8"))["wav2vec2_raw"]
        new = self.config["wav2vec2_raw"]
        allowed = {"experiment_id", "training_regime", "max_epochs"}
        for key in old:
            if key not in allowed:
                self.assertEqual(old[key], new[key], key)
        self.assertEqual(new["max_epochs"], 12)
        self.assertEqual(new["scheduler"], "cosine_annealing")

    def test_pitch_stable_recipe(self):
        item = self.config["pitch_fusion_paper_aligned"]
        self.assertEqual(item["trainable_transformer_layers"], 0)
        self.assertEqual(item["head_learning_rate"], 1e-5)
        self.assertEqual(item["scheduler"], "constant")
        self.assertEqual(item["max_epochs"], 10)
        self.assertEqual(item["patience"], 3)
        self.assertEqual(item["batch_size"] * item["gradient_accumulation_steps"], 16)

    def test_smoke_checkpoint_write_is_guarded(self):
        for name in ("run_b6_wav2vec2_finetune.py", "run_b6_pitch_fusion_finetune.py"):
            source = (ROOT / "src/fedecai/models" / name).read_text(encoding="utf-8")
            self.assertIn("if not args.smoke_test:\n                    checkpoint.parent.mkdir", source)

    def test_new_ids_do_not_collide(self):
        self.assertEqual(len({self.config[name]["experiment_id"] for name in ("wav2vec2_raw", "pitch_fusion_paper_aligned")}), 2)
        self.assertTrue(all("controlled" in self.config[name]["experiment_id"] for name in ("wav2vec2_raw", "pitch_fusion_paper_aligned")))


if __name__ == "__main__":
    unittest.main()
