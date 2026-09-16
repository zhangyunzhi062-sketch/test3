"""将多个 YOLO 数据集安全合并为统一的检测数据集。"""

import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from .config import load_project_config, resolve_project_path


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MARKER_FILE = ".uav_yolo_prepared.json"


class DatasetPreparationError(RuntimeError):
    """数据准备期间发现无法安全继续的问题。"""


def polygon_to_box(coordinates: Sequence[float]) -> Tuple[float, float, float, float]:
    """把归一化多边形坐标转换成 YOLO xywh 外接框。"""
    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        raise DatasetPreparationError("多边形至少需要 3 个点，且坐标数量必须为偶数。")
    xs = list(coordinates[0::2])
    ys = list(coordinates[1::2])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        raise DatasetPreparationError("多边形无法形成有效外接框。")
    return (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        width,
        height,
    )


def convert_label_line(
    line: str,
    class_map: Dict[int, int],
    source: str = "<label>",
    line_number: int = 1,
) -> Optional[str]:
    """转换一行检测框或多边形标签；被忽略的类别返回 None。"""
    fields = line.strip().split()
    if not fields:
        return None

    try:
        class_value = float(fields[0])
        class_id = int(class_value)
    except ValueError as exc:
        raise DatasetPreparationError(
            "{} 第 {} 行的类别编号不是数字。".format(source, line_number)
        ) from exc
    if class_value != class_id or class_id < 0:
        raise DatasetPreparationError(
            "{} 第 {} 行的类别编号必须是非负整数。".format(source, line_number)
        )

    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise DatasetPreparationError(
            "{} 第 {} 行含有非数字坐标。".format(source, line_number)
        ) from exc

    if any(not math.isfinite(value) for value in coordinates):
        raise DatasetPreparationError(
            "{} 第 {} 行含有无穷大或 NaN。".format(source, line_number)
        )
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise DatasetPreparationError(
            "{} 第 {} 行坐标必须位于 0 到 1。".format(source, line_number)
        )

    if class_id not in class_map:
        return None

    if len(coordinates) == 4:
        x_center, y_center, width, height = coordinates
        if width <= 0 or height <= 0:
            raise DatasetPreparationError(
                "{} 第 {} 行的检测框宽高必须大于 0。".format(source, line_number)
            )
    else:
        x_center, y_center, width, height = polygon_to_box(coordinates)

    if (
        x_center < 0
        or x_center > 1
        or y_center < 0
        or y_center > 1
        or width <= 0
        or width > 1
        or height <= 0
        or height > 1
    ):
        raise DatasetPreparationError(
            "{} 第 {} 行转换后得到非法检测框。".format(source, line_number)
        )

    return "{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
        class_map[class_id], x_center, y_center, width, height
    )


def convert_label_file(
    label_path: Optional[Path],
    class_map: Dict[int, int],
) -> Tuple[List[str], int, int]:
    """读取并转换标签文件，返回输出行、保留数量和忽略数量。"""
    if label_path is None or not label_path.is_file():
        return [], 0, 0

    output_lines = []
    ignored = 0
    with label_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            converted = convert_label_line(
                line,
                class_map,
                source=str(label_path),
                line_number=line_number,
            )
            if converted is None:
                ignored += 1
            else:
                output_lines.append(converted)
    return output_lines, len(output_lines), ignored


def prepare_dataset(
    config_path: Path,
    output_override: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """根据配置生成统一数据集，并返回统计清单。"""
    config = load_project_config(Path(config_path))
    output = (
        Path(output_override).expanduser().resolve()
        if output_override
        else resolve_project_path(config, config["project"]["prepared_dataset"])
    )
    sources = _validated_sources(config, output)
    _check_output_policy(output, overwrite)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ".{}-building-{}".format(output.name, uuid.uuid4().hex)
    manifest = _new_manifest(config, output)

    try:
        for split in SPLITS:
            (staging / split / "images").mkdir(parents=True, exist_ok=False)
            (staging / split / "labels").mkdir(parents=True, exist_ok=False)

        used_names: Set[str] = set()
        for source in sources:
            _merge_source(source, staging, manifest, used_names)

        _write_dataset_yaml(staging, output, config["project"]["classes"])
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        _write_json(staging / "dataset_manifest.json", manifest)
        _write_json(
            staging / MARKER_FILE,
            {
                "generator": "uav-yolo-tree-stone",
                "version": 1,
                "created_at": manifest["finished_at"],
            },
        )

        _install_staging_dataset(staging, output, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise

    return manifest


def _validated_sources(
    config: Dict[str, Any],
    output: Path,
) -> List[Dict[str, Any]]:
    sources = []
    for item in config["datasets"]:
        root = resolve_project_path(config, item["root"])
        if not root.is_dir():
            raise DatasetPreparationError("找不到数据集目录：{}".format(root))
        expected_names = item.get("source_names")
        if expected_names is not None:
            if not isinstance(expected_names, list) or not all(
                isinstance(name, str) for name in expected_names
            ):
                raise DatasetPreparationError(
                    "数据集 {} 的 source_names 必须是类别名称列表。".format(item["id"])
                )
            yaml_path = root / "data.yaml"
            if not yaml_path.is_file():
                raise DatasetPreparationError("找不到原始数据集配置：{}".format(yaml_path))
            try:
                with yaml_path.open("r", encoding="utf-8-sig") as file:
                    source_yaml = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise DatasetPreparationError(
                    "原始数据集配置格式错误：{}".format(yaml_path)
                ) from exc
            raw_names = source_yaml.get("names") if isinstance(source_yaml, dict) else None
            if isinstance(raw_names, dict):
                try:
                    actual_names = [
                        raw_names[key] for key in sorted(raw_names, key=lambda key: int(key))
                    ]
                except (ValueError, TypeError) as exc:
                    raise DatasetPreparationError(
                        "原始数据集类别编号无效：{}".format(yaml_path)
                    ) from exc
            else:
                actual_names = raw_names
            if actual_names != expected_names:
                raise DatasetPreparationError(
                    "数据集 {} 的类别与配置不符：期望 {}，实际 {}。".format(
                        item["id"], expected_names, actual_names
                    )
                )
        if _same_or_child(root, output) or _same_or_child(output, root):
            raise DatasetPreparationError(
                "派生数据目录不得与原始数据目录相同或互相包含：{}".format(root)
            )
        for split in SPLITS:
            image_dir = root / split / "images"
            label_dir = root / split / "labels"
            if not image_dir.is_dir():
                raise DatasetPreparationError("缺少图片目录：{}".format(image_dir))
            if not label_dir.is_dir():
                raise DatasetPreparationError("缺少标签目录：{}".format(label_dir))
        sources.append(
            {
                "id": item["id"],
                "root": root,
                "class_map": item["class_map"],
            }
        )
    return sources


def _check_output_policy(output: Path, overwrite: bool) -> None:
    if output.parent == output:
        raise DatasetPreparationError("拒绝将磁盘根目录作为输出目录。")
    if not output.exists():
        return
    if not output.is_dir():
        raise DatasetPreparationError("输出路径已经存在且不是目录：{}".format(output))

    has_content = any(output.iterdir())
    if not has_content:
        return
    if not overwrite:
        raise DatasetPreparationError(
            "输出目录非空。若确定重建，请添加 --overwrite：{}".format(output)
        )
    if not (output / MARKER_FILE).is_file():
        raise DatasetPreparationError(
            "输出目录没有本程序的安全标记，拒绝覆盖：{}".format(output)
        )


def _merge_source(
    source: Dict[str, Any],
    staging: Path,
    manifest: Dict[str, Any],
    used_names: Set[str],
) -> None:
    source_stats = {
        "id": source["id"],
        "root": str(source["root"]),
        "splits": {},
    }
    manifest["sources"].append(source_stats)

    for split in SPLITS:
        image_dir = source["root"] / split / "images"
        label_dir = source["root"] / split / "labels"
        images = sorted(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        split_stats = {
            "images": 0,
            "labels_found": 0,
            "labels_missing": 0,
            "objects_kept": 0,
            "objects_ignored": 0,
            "tree_objects": 0,
            "stone_objects": 0,
        }

        for image_path in images:
            if image_path.stat().st_size == 0:
                raise DatasetPreparationError("图片文件为空：{}".format(image_path))

            destination_name = _unique_image_name(
                source["id"], split, image_path, used_names
            )
            destination_image = staging / split / "images" / destination_name
            destination_label = (
                staging / split / "labels" / Path(destination_name).with_suffix(".txt")
            )
            label_path = label_dir / "{}.txt".format(image_path.stem)
            existing_label = label_path if label_path.is_file() else None
            lines, kept, ignored = convert_label_file(
                existing_label, source["class_map"]
            )

            shutil.copy2(str(image_path), str(destination_image))
            with destination_label.open("w", encoding="utf-8", newline="\n") as file:
                if lines:
                    file.write("\n".join(lines) + "\n")

            split_stats["images"] += 1
            split_stats["objects_kept"] += kept
            split_stats["objects_ignored"] += ignored
            if existing_label:
                split_stats["labels_found"] += 1
            else:
                split_stats["labels_missing"] += 1
            for line in lines:
                class_id = int(line.split(maxsplit=1)[0])
                if class_id == 0:
                    split_stats["tree_objects"] += 1
                elif class_id == 1:
                    split_stats["stone_objects"] += 1

        source_stats["splits"][split] = split_stats
        _add_split_totals(manifest["splits"][split], split_stats)


def _unique_image_name(
    source_id: str,
    split: str,
    image_path: Path,
    used_names: Set[str],
) -> str:
    extension_name = image_path.suffix.lower().lstrip(".")
    base_name = "{}__{}__{}{}".format(
        source_id, image_path.stem, extension_name, image_path.suffix.lower()
    )
    key = "{}/{}".format(split, base_name).lower()
    if key not in used_names:
        used_names.add(key)
        return base_name

    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    candidate = "{}__{}{}".format(Path(base_name).stem, digest, image_path.suffix.lower())
    key = "{}/{}".format(split, candidate).lower()
    if key in used_names:
        raise DatasetPreparationError("无法消除输出文件名冲突：{}".format(image_path))
    used_names.add(key)
    return candidate


def _new_manifest(config: Dict[str, Any], output: Path) -> Dict[str, Any]:
    return {
        "project": config["project"]["name"],
        "classes": config["project"]["classes"],
        "output": str(output),
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None,
        "sources": [],
        "splits": {
            split: {
                "images": 0,
                "labels_found": 0,
                "labels_missing": 0,
                "objects_kept": 0,
                "objects_ignored": 0,
                "tree_objects": 0,
                "stone_objects": 0,
            }
            for split in SPLITS
        },
    }


def _add_split_totals(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += value


def _write_dataset_yaml(
    staging: Path,
    final_output: Path,
    classes: List[str],
) -> None:
    data = {
        "path": final_output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(classes)},
    }
    with (staging / "data.yaml").open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _install_staging_dataset(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if any(output.iterdir()):
            if not overwrite:
                raise DatasetPreparationError("输出目录在生成期间变为非空。")
            backup = output.parent / ".{}-backup-{}".format(output.name, uuid.uuid4().hex)
            os.replace(str(output), str(backup))
            try:
                os.replace(str(staging), str(output))
            except Exception:
                os.replace(str(backup), str(output))
                raise
            shutil.rmtree(str(backup))
        else:
            output.rmdir()
            os.replace(str(staging), str(output))
    else:
        os.replace(str(staging), str(output))


def _same_or_child(path: Path, possible_parent: Path) -> bool:
    path = path.resolve()
    possible_parent = possible_parent.resolve()
    try:
        path.relative_to(possible_parent)
        return True
    except ValueError:
        return False
