import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FinalB6LockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/models/b6_final_lock_v1.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((ROOT / "reports/tables/b6_final/b6_final_lock_validation.json").read_text(encoding="utf-8"))

    def test_exactly_eight_unique_final_active_models(self):
        models = self.config["final_models"]
        self.assertEqual(len(models), 8)
        self.assertEqual(len({item["experiment_id"] for item in models}), 8)
        self.assertTrue(self.config["locked"])

    def test_all_final_paths_exist_and_are_outside_archive(self):
        for item in self.config["final_models"]:
            for key in ("checkpoint", "metrics", "predictions", "report"):
                self.assertNotIn("archive/", item[key].lower())
                self.assertTrue((ROOT / item[key]).is_file(), f"{key}: {item[key]}")

    def test_canonical_table_and_resnet_lock(self):
        with (ROOT / "reports/tables/b6_final/centralized_b6_final_locked.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["artifact_status"] for row in rows}, {"FINAL_ACTIVE"})
        resnet = next(row for row in rows if row["experiment_id"] == "b6_resnet18_mel3_finetune_seed391")
        self.assertEqual(float(resnet["overall_macro_f1"]), 0.6218351697860597)
        self.assertEqual(float(resnet["latency_median_ms"]), 3.513)

    def test_archive_and_test_gates_pass(self):
        self.assertEqual(self.validation["status"], "PASS")
        self.assertEqual(self.validation["archive_hashes_verified"], 294)
        self.assertEqual(self.validation["test_samples_loaded"], 0)
        self.assertEqual(self.validation["active_runner_archive_scan_matches"], 0)
        self.assertTrue(self.validation["smoke_checkpoint_overwrite_guard"])
        self.assertEqual(self.validation["unarchived_failures"], [])

    def test_backbone_decision_is_selected_but_fedecai_not_run(self):
        fedecai = json.loads((ROOT / "configs/models/b6_fedecai_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(fedecai["backbone_status"], "SELECTED")
        self.assertEqual(fedecai["selected_backbone"], "resnet18_mel3")
        self.assertEqual(fedecai["efficiency_fallback"], "mobilenet_v3_mel3")
        self.assertFalse(self.validation["federated_training_triggered"])

    def test_all_centralized_execution_entrypoints_enforce_lock(self):
        entrypoints = [
            ROOT / "scripts/run_centralized_infrastructure_validation.py",
            *(
                ROOT / "src/fedecai/models" / name
                for name in (
                    "run_centralized_smoke_tests.py",
                    "run_b6_all_models.py",
                    "run_b6_classical_pipeline.py",
                    "run_b6_deep_pipeline.py",
                    "run_b6_vit_cached.py",
                    "run_b6_wav2vec2_raw.py",
                    "run_b6_pitch_fusion_paper.py",
                    "run_b6_performance_image.py",
                    "run_b6_wav2vec2_finetune.py",
                    "run_b6_pitch_fusion_finetune.py",
                    "run_b6_performance_all.py",
                    "run_b6_regression_controls.py",
                )
            ),
        ]
        for path in entrypoints:
            self.assertIn("enforce_b6_training_lock", path.read_text(encoding="utf-8"), path.name)

    def test_final_report_figure_is_traceable_and_includes_accent_gap(self):
        figure = ROOT / "reports/figures/b6/final_centralized_b6_ranking.png"
        source = (ROOT / "src/fedecai/models/generate_b6_final_table_figure.py").read_text(encoding="utf-8")
        self.assertTrue(figure.is_file())
        self.assertGreater(figure.stat().st_size, 100_000)
        self.assertIn("Accent Gap ↓", source)
        self.assertNotIn('"Train time s ↓"', source)
        self.assertIn("centralized_b6_final_locked.csv", source)


if __name__ == "__main__":
    unittest.main()
