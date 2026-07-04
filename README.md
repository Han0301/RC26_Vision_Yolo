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
# 标准训练
python train.py
# Z-buffer 训练
python zb_train.py
```

### 推理

```bash
# 标准推理
python infer.py
# Z-buffer 推理
python zb_infer.py
```

## 版本迭代日志

### yolo11_Custom_12roi — 基础 3 分类版
- 3 分类 YOLOv11 模型, 12 ROI, FocalLoss

### yolo11_Custom_12roi_c2 — 2 分类 + 数量约束版
- 2 分类, 新增 YOLO11ROICOUNTLOSS, 对比实验

### yolo11_zb_12roi_c2 — Z-buffer 遮挡处理集成版
- **新增** zb_func.py: Z-buffer 核心函数
- **新增** zb_main.py: Z-buffer 主流程
- **新增** zb_train.py: Z-buffer 训练
- **新增** zb_infer.py: Z-buffer 推理
- **新增** zb_dataset.py: Z-buffer 数据集
- **新增** dataset_trans.py: 数据集转换
- **新增** transform.py: 数据增强
- 集成 Z-buffer 提升遮挡场景鲁棒性
