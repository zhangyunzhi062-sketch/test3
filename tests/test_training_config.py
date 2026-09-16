"""训练参数构造的无模型测试。"""

import argparse
import tempfile
import unittest
from pathlib import Path

from train import build_parser, build_training_arguments
from uav_yolo.config import load_project_config


class TrainingArgumentsTests(unittest.TestCase):
    def test_cli_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_project_root": root,
                "project": {"prepared_dataset": str(root / "prepared")},
                "training": {
                    "model": "yolo11n.pt",
                    "epochs": 100,
                    "imgsz": 640,
                    "batch": 8,
                    "device": "0",
                    "workers": 0,
                    "seed": 42,
                    "patience": 30,
                    "project": "runs/detect",
                    "name": "uav_tree_stone",
                },
            }
            args = argparse.Namespace(
                model=None,
                epochs=5,
                imgsz=None,
                batch=2,
                device="cpu",
                workers=None,
            )
            result = build_training_arguments(config, args)
            self.assertEqual(result["epochs"], 5)
            self.assertEqual(result["batch"], 2)
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["data"], str(root / "prepared" / "data.yaml"))
            self.assertTrue(result["val"])

    def test_100a_gpu_friendly_defaults(self):
        root = Path(__file__).resolve().parents[1]
        config = load_project_config(root / "config" / "project-a.yaml")
        args = build_parser().parse_args([])
        result = build_training_arguments(config, args)

        self.assertEqual(result["model"], "yolo11n.pt")
        self.assertEqual(result["epochs"], 100)
        self.assertEqual(result["imgsz"], 640)
        self.assertEqual(result["batch"], 8)
        self.assertEqual(result["workers"], 0)
        self.assertEqual(result["patience"], 30)
        self.assertEqual(result["name"], "uav_tree_stone")

    def test_100b_accuracy_profile_defaults(self):
        root = Path(__file__).resolve().parents[1]
        config = load_project_config(root / "config" / "project-b.yaml")
        args = build_parser().parse_args([])
        result = build_training_arguments(config, args)

        self.assertEqual(result["model"], "yolo11x.pt")
        self.assertEqual(result["epochs"], 1000)
        self.assertEqual(result["imgsz"], 832)
        self.assertEqual(result["batch"], -1)
        self.assertEqual(result["workers"], 48)
        self.assertEqual(result["patience"], 200)
        self.assertEqual(result["optimizer"], "AdamW")
        self.assertTrue(result["cos_lr"])
        self.assertEqual(result["multi_scale"], 0.25)
        self.assertEqual(result["cache"], "ram")


if __name__ == "__main__":
    unittest.main()
