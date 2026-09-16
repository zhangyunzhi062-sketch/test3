"""检查当前 yolov11 Conda 环境是否适合本项目。"""

from pathlib import Path

from uav_yolo.runtime import configure_ultralytics_dir, inspect_environment


def main() -> int:
    project_root = Path(__file__).resolve().parent
    configure_ultralytics_dir(project_root)
    information, errors, warnings = inspect_environment(run_cuda_test=True)

    print("=== YOLO11 航拍项目环境检查 ===")
    print("Conda 环境：{}".format(information["conda_env"] or "未检测到"))
    print("Python：{} ({})".format(information["python"], information["python_executable"]))
    print("PyTorch：{}".format(information["torch"] or "未安装"))
    print("Ultralytics：{}".format(information["ultralytics"] or "未安装"))
    print("NumPy：{}".format(information["numpy"] or "未安装"))
    print("OpenCV：{}".format(information["opencv"] or "未安装"))
    print("CUDA 可用：{}".format("是" if information["cuda_available"] else "否"))
    if information["gpu"]:
        print("显卡：{}".format(information["gpu"]))
        print("计算能力：{}".format(information["gpu_capability"]))
        print("PyTorch 编译架构：{}".format(", ".join(information["compiled_arches"])))
    print("最小 CUDA 运算：{}".format(information["cuda_test"]))

    for warning in warnings:
        print("[警告] {}".format(warning))
    for error in errors:
        print("[错误] {}".format(error))

    if errors:
        print("检查未通过，请先根据错误信息修复环境。")
        return 1
    print("环境检查完成，可以继续准备数据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
