# Winter.Z 的图像识别 YOLO11 ver1.0

这是一个用 Python 和 Ultralytics YOLO11 训练航拍图片目标检测模型的入门项目。当前 main 分支默认配置为 **1.0.0B 威力加强版**；个人电脑建议下载 **1.0.0A 显卡友好版（Latest）**。两个版本共用代码与数据，只是训练默认参数不同。请选择 [Releases](https://github.com/zhangyunzhi062-sketch/test3/releases) 中适合自己的完整包。

模型目前只检测两类：tree（树木、树冠、树干）和 stone（石头、岩块）。所附数据没有独立的森林标注，因此目前不能直接识别 forest。项目专注于静态图片，未来可扩展到无人机摄像头或 RTSP。

## 下载后有什么

- database：随项目提供的三套 Roboflow YOLO11 原始数据，含 train、valid、test 和 data.yaml；无需更改盘符或用户名。详见[数据来源](数据来源.md)。
- 待检测图片：默认待识别图片目录。
- config/project.yaml：所下载版本的默认设置；config/project-a.yaml 和 config/project-b.yaml 可供手动切换。
- prepare_dataset.py：校验、合并三套数据，不修改原始文件。
- train.py：训练或从 last.pt 继续训练。
- predict_image.py：默认检测“待检测图片”，输出标注图与 JSON/CSV，并尝试自动打开结果目录。
- check_environment.py：检查 Python、PyTorch、Ultralytics、NumPy、OpenCV 与 CUDA。

完整发布包附带相应的官方初始训练权重 yolo11n.pt 或 yolo11x.pt；Git 仓库和 GitHub 自动生成的源码 ZIP 不包含权重，也不包含任何训练后的模型。源码 ZIP 已含三套数据；若下载源码 ZIP，首次训练需要联网获取官方权重，或按[使用手册](使用手册.md)手动下载。

## 最短流程

在 Windows CMD 中激活 yolov11 环境并进入项目解压目录，然后逐行执行：

~~~bat
conda activate yolov11
cd /d "项目解压目录"
pip install -r requirements.txt
python check_environment.py
python prepare_dataset.py
python train.py
~~~

训练完成后，把图片放进“待检测图片”，执行：

~~~bat
python predict_image.py
~~~

以上默认检测权重位置为 runs\detect\uav_tree_stone\weights\best.pt。B 版训练结果默认位于另一个目录，检测它时请用 --model 指明相应 best.pt，具体命令见[零基础中文使用手册](使用手册.md)。训练结果、生成数据和用户图片不会提交到仓库。

## 两版默认训练设置

| Release | 模型 | epochs | imgsz | batch | workers | patience |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.0.0A 显卡友好版（Latest） | yolo11n.pt | 100 | 640 | 8 | 0 | 30 |
| 1.0.0B 威力加强版 | yolo11x.pt | 1000 | 832 | 自动 | 48 | 200 |

B 版另外使用 AdamW、余弦学习率、多尺度训练与内存缓存。参数更大、训练更久并不保证在这三套数据上一定更准，请比较验证集和测试集指标。两版默认设备均为 GPU 0、随机种子 42，且都支持命令行覆盖。

环境配置、路径引号、续训、显存不足和权重下载故障，请阅读[使用手册](使用手册.md)。技术背景见[研究结论](研究结论.md)。
