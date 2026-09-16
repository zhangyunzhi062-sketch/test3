"""使用统一数据集训练 YOLO11 检测模型。"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from uav_yolo.config import (
    ConfigError,
    load_project_config,
    require_keys,
    resolve_project_path,
)
from uav_yolo.runtime import configure_ultralytics_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练航拍树木与石头 YOLO11 模型。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "project.yaml",
        help="项目配置文件。",
    )
    parser.add_argument("--model", help="初始权重，如 yolo11n.pt。")
    parser.add_argument("--epochs", type=int, help="训练轮数。")
    parser.add_argument("--imgsz", type=int, help="输入图片尺寸。")
    parser.add_argument("--batch", type=int, help="批次大小。")
    parser.add_argument("--device", help="设备，如 0 或 cpu。")
    parser.add_argument("--workers", type=int, help="数据加载进程数。")
    parser.add_argument("--resume", type=Path, help="从 last.pt 继续训练。")
    return parser


def build_training_arguments(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    training = dict(config["training"])
    require_keys(
        training,
        (
            "model",
            "epochs",
            "imgsz",
            "batch",
            "device",
            "workers",
            "seed",
            "patience",
            "project",
            "name",
        ),
        "training",
    )
    for key in ("model", "epochs", "imgsz", "batch", "device", "workers"):
        value = getattr(args, key, None)
        if value is not None:
            training[key] = value

    training["project"] = str(resolve_project_path(config, training["project"]))
    training.update(
        {
            "data": str(
                resolve_project_path(
                    config, config["project"]["prepared_dataset"]
                )
                / "data.yaml"
            ),
            "val": True,
            "plots": True,
            "save": True,
            "exist_ok": False,
        }
    )
    return training


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_project_config(args.config)
        configure_ultralytics_dir(config["_project_root"])
        from ultralytics import YOLO

        if args.resume:
            checkpoint = args.resume.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError("找不到继续训练权重：{}".format(checkpoint))
            print("从检查点继续训练：{}".format(checkpoint))
            model = YOLO(str(checkpoint))
            model.train(resume=True)
            return 0

        training = build_training_arguments(config, args)
        data_yaml = Path(training["data"])
        if not data_yaml.is_file():
            raise FileNotFoundError(
                "找不到派生数据配置：{}。请先运行 prepare_dataset.py。".format(
                    data_yaml
                )
            )
        model_name = training.pop("model")
        local_model = resolve_project_path(config, model_name)
        if local_model.is_file():
            model_name = str(local_model)
        print("初始权重：{}".format(model_name))
        print("数据配置：{}".format(data_yaml))
        print("输出目录：{}\\{}".format(training["project"], training["name"]))
        model = YOLO(str(model_name))
        model.train(**training)
        print("训练完成，请在输出目录的 weights\\best.pt 中查找最佳权重。")
        return 0
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print("[训练未启动或已停止] {}".format(exc))
        return 1
    except Exception as exc:
        print("[训练失败] {}".format(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
