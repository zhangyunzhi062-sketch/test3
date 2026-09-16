"""预测默认目录与结果目录打开行为测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from predict_image import build_parser
from uav_yolo.config import load_project_config, resolve_project_path
from uav_yolo.runtime import open_result_directory


class PredictionDefaultsTests(unittest.TestCase):
    def test_model_and_source_are_optional(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.model)
        self.assertIsNone(args.source)
        self.assertFalse(args.no_open)

    def test_default_source_is_inside_project(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "project.yaml"
        config = load_project_config(config_path)
        source = resolve_project_path(config, config["prediction"]["source"])
        model = resolve_project_path(config, config["prediction"]["model"])
        self.assertEqual(source.name, "待检测图片")
        self.assertEqual(source.parent, config["_project_root"])
        self.assertEqual(
            model,
            config["_project_root"]
            / "runs"
            / "detect"
            / "uav_tree_stone"
            / "weights"
            / "best.pt",
        )
        self.assertTrue(config["prediction"]["open_result"])


class OpenResultDirectoryTests(unittest.TestCase):
    def test_missing_directory_is_reported_without_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            opened, message = open_result_directory(missing)
        self.assertFalse(opened)
        self.assertIn("目录不存在", message)

    @unittest.skipUnless(os.name == "nt", "Windows 资源管理器测试")
    def test_windows_directory_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(os, "startfile", create=True) as startfile:
                opened, message = open_result_directory(Path(directory))
        self.assertTrue(opened)
        self.assertEqual(message, "")
        startfile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
