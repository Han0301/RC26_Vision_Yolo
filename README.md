# RC26 Vision YOLO — YOLOv11 自定义模型训练工作区

> 基于 YOLOv11 的 12-ROI 方块检测模型训练与推理，用于 RC26 机器人竞赛视觉感知。

## 项目概述

本项目是针对 RC26 机器人竞赛的 **12-ROI 方块检测模型**训练工作区，基于 Ultralytics YOLOv11 进行自定义模型开发，迭代了 6 个版本，逐步引入 Z-buffer 遮挡处理、Point-size 加权、注意力机制等改进。

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLOv11
- OpenVINO（模型导出需要）

### 安装依赖

```bash
pip install torch ultralytics openvino opencv-python
```

### 训练

```bash
python train.py
```

### 推理

```bash
# 基础推理
python infer.py
```

### 模型导出

```bash
python pt_to_onnx_openvino.py
```

## 版本迭代日志

### yolo11_Custom_12roi — 基础 3 分类版
- **模型**: YOLO11ROIBackbone + Neck + Head
- **分类**: 3 类（0=无效ROI, 1=有效无方块, 2=有效有方块）
- **损失**: YOLO11ROIFocalLoss3C
- **文件**: model.py, train.py, inferencer.py, loss.py, dataset.py, create_real_datasets.py, pt_to_onnx_openvino.py

### yolo11_Custom_12roi_c2 — 2 分类 + 数量约束版
- **改为 2 分类**: 删除无效类，简化为"有无方块"二分类
- **新增** YOLO11ROICOUNTLOSS: 数量约束损失
- **新增** compare_model.py 模型对比工具
- **新增** infer.py 重构推理
- **对比实验**: count_loss_weight=[0.0, 0.1, 0.2, 0.25]，结论 0.0 最佳
