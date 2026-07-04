# RC26 Vision YOLO — YOLOv11 自定义模型训练工作区

> 基于 YOLOv11 的 12-ROI 方块检测模型训练与推理。

## 使用方法

```bash
pip install torch ultralytics openvino opencv-python
python main.py          # 统一入口 (训练/推理/导出)
python train_main.py    # 训练
python infer_main.py    # 推理
python show_atten.py    # 注意力可视化
python pt_to_onnx_openvino.py  # 模型导出
```

## 版本迭代日志

### yolo11_Custom_12roi — 基础 3 分类版
### yolo11_Custom_12roi_c2 — 2 分类 + 数量约束版
### yolo11_zb_12roi_c2 — Z-buffer 遮挡处理集成版
### yolo11_Custom_pointsize — Point-size 加权版

### yolo11_Custom_atten — 注意力机制引入版
- **架构重构**: 模块化重写, 15 个新文件
- **新增** LocalGrid_Attention: 局部网格注意力模块
- **新增** show_atten.py: 注意力热力图可视化
- **新增** main.py: 统一入口 (train/infer/export)
- **新增** load_model.py / train_config.py: 配置集中管理
- **新增** dataset_check.py / dataset_filter.py: 数据质量工具
- **新损失**: CountLoss / FocalLoss / BCELoss (重命名重构)
- **移除** Z-buffer 管线: zb_func/zb_main/zb_dataset 等
- **移除** evlate/ log_to_datasets/ post_processing/ 等
- **分类**: 2 类 (二分类, 保持)
