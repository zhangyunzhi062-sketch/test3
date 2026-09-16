"""使用训练后的 YOLO11 权重检测图片或图片目录。"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from uav_yolo.config import ConfigError, load_project_config, resolve_project_path
from uav_yolo.dataset import IMAGE_EXTENSIONS
from uav_yolo.runtime import (
    configure_ultralytics_dir,
    next_available_directory,
    open_result_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检测指定图片或图片目录。")
    parser.add_argument(
        "--model",
        type=Path,
        help="训练得到的 best.pt；不填写时使用项目原始训练目录中的默认模型。",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="图片文件或图片目录；不填写时检测项目根目录的“待检测图片”。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "project.yaml",
        help="项目配置文件。",
    )
    parser.add_argument("--output", type=Path, help="结果输出目录。")
    parser.add_argument("--conf", type=float, help="置信度阈值，0 到 1。")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值，0 到 1。")
    parser.add_argument("--imgsz", type=int, help="推理图片尺寸。")
    parser.add_argument("--device", help="设备，如 0 或 cpu。")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="检测完成后不自动打开结果目录，适合远程或无桌面环境。",
    )
    return parser


def _validate_source(source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError("找不到图片或目录：{}".format(source))
    if source.is_file() and source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("不支持的图片格式：{}".format(source.suffix))
    if source.is_dir():
        has_images = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in source.iterdir()
        )
        if not has_images:
            raise ValueError("目录中没有支持的图片：{}".format(source))


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _collect_rows(
    results: Iterable[Any],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image_index, result in enumerate(results, start=1):
        source_path = Path(str(result.path))
        output_name = "{:04d}__{}".format(image_index, source_path.name)
        output_image = output_dir / "images" / output_name
        result.save(filename=str(output_image))

        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        for detection_index, (box, confidence, class_value) in enumerate(
            zip(xyxy, confidences, classes), start=1
        ):
            class_id = int(class_value)
            rows.append(
                {
                    "source": str(source_path),
                    "annotated_image": str(output_image),
                    "detection_index": detection_index,
                    "class_id": class_id,
                    "class_name": _class_name(result.names, class_id),
                    "confidence": round(float(confidence), 6),
                    "x1": round(float(box[0]), 3),
                    "y1": round(float(box[1]), 3),
                    "x2": round(float(box[2]), 3),
                    "y2": round(float(box[3]), 3),
                }
            )
    return rows


def _write_results(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    with (output_dir / "detections.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")

    fieldnames = [
        "source",
        "annotated_image",
        "detection_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with (output_dir / "detections.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_project_config(args.config)
        configure_ultralytics_dir(config["_project_root"])
        prediction = dict(config["prediction"])

        model_path = (
            args.model.expanduser().resolve()
            if args.model
            else resolve_project_path(config, prediction["model"])
        )
        source = (
            args.source.expanduser().resolve()
            if args.source
            else resolve_project_path(config, prediction["source"])
        )
        if not model_path.is_file():
            raise FileNotFoundError("找不到模型权重：{}".format(model_path))
        _validate_source(source)

        conf = prediction["conf"] if args.conf is None else args.conf
        iou = prediction["iou"] if args.iou is None else args.iou
        imgsz = prediction["imgsz"] if args.imgsz is None else args.imgsz
        device = prediction["device"] if args.device is None else args.device
        if not 0 <= float(conf) <= 1 or not 0 <= float(iou) <= 1:
            raise ValueError("--conf 和 --iou 必须位于 0 到 1。")

        output_base = (
            args.output.expanduser().resolve()
            if args.output
            else resolve_project_path(config, prediction["output"])
        )
        output_dir = next_available_directory(output_base)
        (output_dir / "images").mkdir(parents=True, exist_ok=False)

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.predict(
            source=str(source),
            stream=True,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=str(device),
            save=False,
            verbose=True,
        )
        rows = _collect_rows(results, output_dir)
        _write_results(output_dir, rows)
        print("识别完成：{}".format(output_dir))
        print("共保存 {} 条检测记录。".format(len(rows)))
        if bool(prediction.get("open_result", True)) and not args.no_open:
            opened, message = open_result_directory(output_dir)
            if not opened:
                print("[提示] 无法自动打开结果目录：{}".format(message))
        return 0
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print("[预测未启动或已停止] {}".format(exc))
        return 1
    except Exception as exc:
        print("[预测失败] {}".format(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
