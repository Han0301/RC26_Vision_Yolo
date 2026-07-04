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
python inferencer.py
# 或
python infer.py
```

### 模型导出

```bash
python pt_to_onnx_openvino.py
```

## 版本迭代日志

### yolo11_Custom_12roi — 基础 3 分类版
- **模型架构**: YOLO11ROIBackbone + YOLO11ROINeck + YOLO11ROIHead
- **分类**: 3 类（0=无效ROI, 1=有效无方块, 2=有效有方块）
- **ROI**: 12 个固定位置 ROI
- **损失函数**: YOLO11ROIFocalLoss3C（Focal Loss 3 分类）
- **训练**: `train.py` 完整训练流程
- **推理**: `inferencer.py` 批量推理
- **数据集**: `create_real_datasets.py` 真实数据生成
- **导出**: `pt_to_onnx_openvino.py` 模型导出

### yolo11_Custom_12roi_c2 — 2 分类 + 数量约束版
- **改为 2 分类**: 删除无效类，简化为"有无方块"二分类
- **新增** `YOLO11ROICOUNTLOSS`: 数量约束损失，控制预测方块数量
- **新增** `compare_model.py`: 模型对比工具
- **新增** `infer.py`: 重构的推理脚本
- **新增** `readme.md`: 训练日志与实验记录
- 对比实验 count_loss_weight = [0.0, 0.1, 0.2, 0.25]
- 结论: count_loss_weight=0.0 效果最佳

### yolo11_zb_12roi_c2 — Z-buffer 遮挡处理集成版
- **新增** `zb_func.py`: Z-buffer 核心处理函数
- **新增** `zb_main.py`: Z-buffer 主流程
- **新增** `zb_train.py`: Z-buffer 训练流程
- **新增** `zb_infer.py`: Z-buffer 推理流程
- **新增** `zb_dataset.py`: Z-buffer 数据集处理
- **新增** `dataset_trans.py`: 数据集格式转换
- **新增** `transform.py`: 数据增强变换
- 集成 Z-buffer 遮挡处理，提升遮挡场景检测鲁棒性

### yolo11_Custom_pointsize — Point-size 加权版
- **新增** `conf_weight` 参数: 在损失函数中添加 point-size 置信度加权
- **新增** `YOLO11ROIBCEWithLogitsLoss2C`: BCE 损失变体
- **新增** `log_to_datasets.py`: 日志转数据集
- **新增** `post_processing.py`: 后处理模块
- **新增** `realdata_trans.py`: 真实数据格式转换
- 让模型对高 point-size（丰富图像信息）的图片给予高置信度
- 保持 Z-buffer 管线与 2 分类架构

### yolo11_Custom_atten — 注意力机制引入版
- **架构重构**: 模块化重写（dataset_func/dataset_main, infer_func/infer_main, train_func/train_main）
- **新增** `LocalGrid_Attention`: 局部网格注意力模块
- **新增** `show_atten.py`: 注意力可视化工具
- **新增** `load_model.py`: 模型加载封装
- **新增** `train_config.py`: 集中式训练配置
- **新增** `dataset_check.py` / `dataset_filter.py`: 数据集检查与过滤
- **损失函数**: 重命名为 `CountLoss` / `FocalLoss` / `BCELoss`
- **分类**: 2 类（二分类）

### yolo11_Custom_atten2 — 注意力联合训练版
- **新增** `main.py`: 统一入口，支持训练/推理/导出模式切换
- **保持** LocalGrid_Attention 注意力模块
- **改进训练策略**: 12ROI + 单张 ROI 共同训练
- 防止模型过于依赖注意力模块
- 不局限单张 ROI 场景，支持任意数量图片输入
- 代码结构进一步优化
