"""Ultralytics 运行目录、环境检查和通用工具。"""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXPECTED_PYTHON = (3, 10)
EXPECTED_TORCH_PREFIX = "1.13.1+cu116"
MINIMUM_ULTRALYTICS = (8, 3, 0)


def _version_numbers(value: str) -> Tuple[int, ...]:
    """只提取版本号开头的数字段，用于给出易懂的兼容性提示。"""
    numbers = []
    for part in value.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def configure_ultralytics_dir(project_root: Path) -> Path:
    """把 Ultralytics 用户配置放到项目内的忽略目录。"""
    target = Path(project_root).resolve() / ".ultralytics"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(target))
    return target


def inspect_environment(run_cuda_test: bool = True) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """收集当前环境信息，返回信息、错误列表和警告列表。"""
    information: Dict[str, Any] = {
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": None,
        "ultralytics": None,
        "numpy": None,
        "opencv": None,
        "cuda_available": False,
        "gpu": None,
        "gpu_capability": None,
        "compiled_arches": [],
        "cuda_test": "未执行",
    }
    errors: List[str] = []
    warnings: List[str] = []

    if sys.version_info[:2] != EXPECTED_PYTHON:
        warnings.append(
            "计划环境为 Python 3.10，当前为 {}。".format(platform.python_version())
        )
    if information["conda_env"] != "yolov11":
        warnings.append(
            "当前 CONDA_DEFAULT_ENV 不是 yolov11，请先执行 conda activate yolov11。"
        )

    try:
        information["ultralytics"] = importlib.metadata.version("ultralytics")
        if _version_numbers(information["ultralytics"]) < MINIMUM_ULTRALYTICS:
            errors.append("Ultralytics 版本过旧，不支持 YOLO11；建议安装 8.4.112。")
    except importlib.metadata.PackageNotFoundError:
        errors.append("未安装 ultralytics。")

    try:
        information["numpy"] = importlib.metadata.version("numpy")
        if _version_numbers(information["numpy"])[:1] >= (2,):
            errors.append(
                "PyTorch 1.13.1 与 NumPy 2 不兼容；请在 yolov11 环境中执行 pip install -r requirements.txt。"
            )
    except importlib.metadata.PackageNotFoundError:
        errors.append("未安装 NumPy；请执行 pip install -r requirements.txt。")

    try:
        information["opencv"] = importlib.metadata.version("opencv-python")
        if _version_numbers(information["opencv"])[:2] >= (4, 12):
            errors.append(
                "当前 OpenCV 需要 NumPy 2；请执行 pip install -r requirements.txt 安装兼容版本。"
            )
    except importlib.metadata.PackageNotFoundError:
        errors.append("未安装 opencv-python；请执行 pip install -r requirements.txt。")

    try:
        import torch
    except Exception as exc:
        errors.append("无法导入 PyTorch：{}".format(exc))
        return information, errors, warnings

    information["torch"] = torch.__version__
    if not str(torch.__version__).startswith(EXPECTED_TORCH_PREFIX):
        warnings.append(
            "计划使用 PyTorch {}，当前为 {}。".format(
                EXPECTED_TORCH_PREFIX, torch.__version__
            )
        )

    information["cuda_available"] = bool(torch.cuda.is_available())
    if not information["cuda_available"]:
        warnings.append("CUDA 当前不可用；训练将非常慢，建议检查显卡驱动与 PyTorch。")
        return information, errors, warnings

    try:
        information["gpu"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        information["gpu_capability"] = "{}.{}".format(capability[0], capability[1])
        information["compiled_arches"] = list(torch.cuda.get_arch_list())
        current_arch = "sm_{}{}".format(capability[0], capability[1])
        if current_arch not in information["compiled_arches"]:
            warnings.append(
                "PyTorch 编译列表没有直接包含 {}；将通过最小 CUDA 运算确认兼容性。".format(
                    current_arch
                )
            )
        if run_cuda_test:
            value = (torch.ones(1, device="cuda") * 2).item()
            torch.cuda.synchronize()
            if value != 2.0:
                raise RuntimeError("CUDA 运算结果异常")
            information["cuda_test"] = "通过"
    except Exception as exc:
        information["cuda_test"] = "失败"
        errors.append("最小 CUDA 运算失败：{}".format(exc))

    return information, errors, warnings


def next_available_directory(base: Path) -> Path:
    """返回不存在的输出目录，避免覆盖旧结果。"""
    base = Path(base)
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.parent / "{}_{}".format(base.name, index)
        if not candidate.exists():
            return candidate
        index += 1


def open_result_directory(directory: Path) -> Tuple[bool, str]:
    """用系统文件管理器打开结果目录；失败时返回原因，不中断预测。"""
    target = Path(directory).resolve()
    if not target.is_dir():
        return False, "目录不存在：{}".format(target)

    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except (OSError, RuntimeError) as exc:
        return False, str(exc)
    return True, ""
