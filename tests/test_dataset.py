"""数据合并逻辑的无模型单元测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from uav_yolo.dataset import (
    MARKER_FILE,
    DatasetPreparationError,
    convert_label_line,
    polygon_to_box,
    prepare_dataset,
)


class LabelConversionTests(unittest.TestCase):
    def test_polygon_to_box(self):
        box = polygon_to_box([0.1, 0.2, 0.5, 0.2, 0.5, 0.8, 0.1, 0.8])
        self.assertAlmostEqual(box[0], 0.3)
        self.assertAlmostEqual(box[1], 0.5)
        self.assertAlmostEqual(box[2], 0.4)
        self.assertAlmostEqual(box[3], 0.6)

    def test_class_mapping_and_ignored_class(self):
        mapped = convert_label_line("2 0.5 0.5 0.2 0.4", {2: 0})
        self.assertEqual(mapped, "0 0.50000000 0.50000000 0.20000000 0.40000000")
        self.assertIsNone(convert_label_line("4 0.5 0.5 0.2 0.4", {2: 0}))

    def test_rejects_invalid_coordinates(self):
        with self.assertRaises(DatasetPreparationError):
            convert_label_line("0 1.2 0.5 0.2 0.2", {0: 1})

    def test_empty_and_unmapped_labels(self):
        self.assertIsNone(convert_label_line("", {0: 1}))
        self.assertIsNone(convert_label_line("4 0.5 0.5 0.1 0.1", {0: 1}))

    def test_rejects_invalid_polygon(self):
        with self.assertRaises(DatasetPreparationError):
            convert_label_line("0 0.1 0.2 0.5 0.2 0.5", {0: 1})


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = []
        for dataset_id in ("nature", "stone", "tree_top"):
            dataset_root = self.root / dataset_id
            self.sources.append((dataset_id, dataset_root))
            for split in ("train", "valid", "test"):
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "labels").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def _write_image_and_label(self, dataset_root, split, name, label):
        (dataset_root / split / "images" / name).write_bytes(b"not-decoded-in-test")
        label_name = "{}.txt".format(Path(name).stem)
        (dataset_root / split / "labels" / label_name).write_text(
            label, encoding="utf-8"
        )

    def _write_config(self, output):
        config = {
            "project": {
                "name": "test_project",
                "prepared_dataset": str(output),
                "classes": ["tree", "stone"],
            },
            "datasets": [
                {
                    "id": "nature",
                    "root": str(self.sources[0][1]),
                    "class_map": {1: 1, 2: 0, 3: 0},
                },
                {
                    "id": "stone",
                    "root": str(self.sources[1][1]),
                    "class_map": {0: 1},
                },
                {
                    "id": "tree_top",
                    "root": str(self.sources[2][1]),
                    "class_map": {0: 0},
                },
            ],
            "training": {},
            "prediction": {},
        }
        config_dir = self.root / "config"
        config_dir.mkdir(exist_ok=True)
        path = config_dir / "project.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def test_preserves_splits_and_converts_polygon(self):
        for split in ("train", "valid", "test"):
            self._write_image_and_label(
                self.sources[0][1], split, "nature.jpg", "2 0.5 0.5 0.2 0.2\n"
            )
            self._write_image_and_label(
                self.sources[1][1],
                split,
                "stone.png",
                "0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8\n",
            )
            self._write_image_and_label(
                self.sources[2][1], split, "tree.jpg", "0 0.4 0.4 0.1 0.1\n"
            )

        output = self.root / "prepared"
        manifest = prepare_dataset(self._write_config(output))

        self.assertTrue((output / MARKER_FILE).is_file())
        self.assertTrue((output / "data.yaml").is_file())
        for split in ("train", "valid", "test"):
            images = list((output / split / "images").iterdir())
            labels = list((output / split / "labels").iterdir())
            self.assertEqual(len(images), 3)
            self.assertEqual(len(labels), 3)
            self.assertEqual(manifest["splits"][split]["tree_objects"], 2)
            self.assertEqual(manifest["splits"][split]["stone_objects"], 1)
        saved_manifest = json.loads(
            (output / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_manifest["classes"], ["tree", "stone"])

    def test_refuses_to_overwrite_unmarked_directory(self):
        output = self.root / "unrelated"
        output.mkdir()
        (output / "important.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(DatasetPreparationError):
            prepare_dataset(self._write_config(output), overwrite=True)
        self.assertEqual(
            (output / "important.txt").read_text(encoding="utf-8"), "keep"
        )

    def test_overwrite_requires_flag_even_for_marked_directory(self):
        output = self.root / "prepared"
        config_path = self._write_config(output)
        prepare_dataset(config_path)
        with self.assertRaises(DatasetPreparationError):
            prepare_dataset(config_path)
        prepare_dataset(config_path, overwrite=True)
        self.assertTrue((output / MARKER_FILE).is_file())

    def test_rejects_unexpected_source_class_names(self):
        output = self.root / "prepared"
        config_path = self._write_config(output)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["datasets"][0]["source_names"] = ["tree"]
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )
        (self.sources[0][1] / "data.yaml").write_text(
            "names: [rock]\n", encoding="utf-8"
        )
        with self.assertRaises(DatasetPreparationError):
            prepare_dataset(config_path)

    def test_missing_label_and_same_stem_images_are_safe(self):
        nature_root = self.sources[0][1]
        self._write_image_and_label(
            nature_root, "train", "same.jpg", "2 0.5 0.5 0.2 0.2\n"
        )
        (nature_root / "train" / "images" / "same.png").write_bytes(b"second-image")
        (nature_root / "train" / "images" / "background.jpg").write_bytes(
            b"background"
        )

        output = self.root / "prepared"
        manifest = prepare_dataset(self._write_config(output))

        output_images = list((output / "train" / "images").iterdir())
        output_labels = list((output / "train" / "labels").iterdir())
        self.assertEqual(len(output_images), 3)
        self.assertEqual(len(output_labels), 3)
        self.assertEqual(len({path.name.lower() for path in output_images}), 3)
        self.assertEqual(manifest["splits"]["train"]["labels_missing"], 1)
        empty_labels = [
            path for path in output_labels if path.read_text(encoding="utf-8") == ""
        ]
        self.assertEqual(len(empty_labels), 1)


if __name__ == "__main__":
    unittest.main()
