"""验证两个版本使用同一组数据与一致的默认预测位置。"""

import unittest
from pathlib import Path

from uav_yolo.config import load_project_config, resolve_project_path
from uav_yolo.runtime import _version_numbers


class ProfileTests(unittest.TestCase):
    def test_default_profile_matches_version(self):
        root = Path(__file__).resolve().parents[1]
        version = (root / "VERSION").read_text(encoding="utf-8").strip()
        expected_name = {
            "1.0.0A": "project-a.yaml",
            "1.0.0B": "project-b.yaml",
        }[version]
        default = load_project_config(root / "config" / "project.yaml")
        expected = load_project_config(root / "config" / expected_name)
        self.assertEqual(default["training"], expected["training"])
        self.assertEqual(default["prediction"], expected["prediction"])
        self.assertEqual(default["datasets"], expected["datasets"])

    def test_profiles_use_uploaded_yolo11_data(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "Nature objects.v1i.yolov11",
            "Stone.v1-training-stone.yolov11",
            "Tree-Top-View.v1i.yolov11",
        }
        for name in ("project.yaml", "project-a.yaml", "project-b.yaml"):
            with self.subTest(name=name):
                config = load_project_config(root / "config" / name)
                actual = {
                    Path(item["root"]).name for item in config["datasets"]
                }
                self.assertEqual(actual, expected)
                self.assertTrue(
                    all(
                        resolve_project_path(config, item["root"]).is_dir()
                        for item in config["datasets"]
                    )
                )
                self.assertEqual(
                    config["prediction"]["model"],
                    "runs/detect/uav_tree_stone/weights/best.pt",
                )
                self.assertEqual(config["prediction"]["source"], "待检测图片")

    def test_compatibility_version_parser(self):
        self.assertEqual(_version_numbers("8.4.112"), (8, 4, 112))
        self.assertEqual(_version_numbers("1.13.1+cu116"), (1, 13, 1))


if __name__ == "__main__":
    unittest.main()
