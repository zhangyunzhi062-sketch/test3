"""项目 YAML 配置的读取与校验。"""

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class ConfigError(ValueError):
    """配置文件内容不完整或不合法。"""


def load_project_config(config_path: Path) -> Dict[str, Any]:
    """读取并校验项目配置。"""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError("找不到配置文件：{}".format(path))

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError("配置文件 YAML 格式错误：{}".format(exc)) from exc

    if not isinstance(config, dict):
        raise ConfigError("配置文件顶层必须是一个映射。")

    _require_mapping(config, "project")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("datasets 必须是非空列表。")
    _require_mapping(config, "training")
    _require_mapping(config, "prediction")

    project = config["project"]
    classes = project.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ConfigError("project.classes 必须是非空列表。")
    if classes != ["tree", "stone"]:
        raise ConfigError("第一版类别顺序必须固定为 ['tree', 'stone']。")
    if not project.get("prepared_dataset"):
        raise ConfigError("缺少 project.prepared_dataset。")

    ids = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ConfigError("datasets[{}] 必须是映射。".format(index))
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ConfigError("datasets[{}].id 不能为空。".format(index))
        if dataset_id in ids:
            raise ConfigError("数据集 id 重复：{}".format(dataset_id))
        ids.add(dataset_id)
        if not dataset.get("root"):
            raise ConfigError("数据集 {} 缺少 root。".format(dataset_id))
        class_map = dataset.get("class_map")
        if not isinstance(class_map, dict) or not class_map:
            raise ConfigError("数据集 {} 的 class_map 必须是非空映射。".format(dataset_id))
        normalized_map = {}
        for old_id, new_id in class_map.items():
            try:
                old_value = int(old_id)
                new_value = int(new_id)
            except (TypeError, ValueError) as exc:
                raise ConfigError("数据集 {} 的类别编号必须是整数。".format(dataset_id)) from exc
            if old_value < 0 or new_value < 0 or new_value >= len(classes):
                raise ConfigError("数据集 {} 存在越界类别映射。".format(dataset_id))
            normalized_map[old_value] = new_value
        dataset["class_map"] = normalized_map

    config["_config_path"] = path
    config["_project_root"] = path.parent.parent.resolve()
    return config


def resolve_project_path(config: Dict[str, Any], value: Any) -> Path:
    """将相对路径按项目根目录解析，绝对路径保持不变。"""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def require_keys(mapping: Dict[str, Any], keys: Iterable[str], section: str) -> None:
    """检查配置区块是否包含给定键。"""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError("{} 缺少配置项：{}".format(section, ", ".join(missing)))


def _require_mapping(config: Dict[str, Any], key: str) -> None:
    if not isinstance(config.get(key), dict):
        raise ConfigError("{} 必须是映射。".format(key))
