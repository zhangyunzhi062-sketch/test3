"""合并并统一三个原始 YOLO 数据集。"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from uav_yolo.dataset import DatasetPreparationError, prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并树木与石头数据集，不修改原始文件。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "project.yaml",
        help="项目配置文件，默认使用 config/project.yaml。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="覆盖配置中的派生数据输出目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重建带有本程序安全标记的已有输出目录。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_dataset(
            config_path=args.config,
            output_override=args.output,
            overwrite=args.overwrite,
        )
    except DatasetPreparationError as exc:
        print("[数据准备失败] {}".format(exc))
        return 1
    except Exception as exc:
        print("[意外错误] {}".format(exc))
        return 1

    print("数据准备完成：{}".format(manifest["output"]))
    print("类别：{}".format(", ".join(manifest["classes"])))
    for split, stats in manifest["splits"].items():
        print(
            "{}：{} 张图片，{} 个 tree，{} 个 stone，忽略 {} 个非目标标注".format(
                split,
                stats["images"],
                stats["tree_objects"],
                stats["stone_objects"],
                stats["objects_ignored"],
            )
        )
    print("训练配置：{}".format(Path(manifest["output"]) / "data.yaml"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
