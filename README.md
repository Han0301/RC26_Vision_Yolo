# RC26 Vision YOLO — YOLOv11 自定义模型训练工作区

> 基于 YOLOv11 的 12-ROI 方块检测模型训练与推理。

## 使用方法

```bash
pip install torch ultralytics openvino opencv-python
python main.py          # 统一入口 (训练/推理/导出)
python train_main.py    # 训练
python infer_main.py    # 推理
python show_atten.py    # 注意力可视化
```

## 版本迭代日志

### yolo11_Custom_12roi — 基础 3 分类版
- YOLO11ROIBackbone + Neck + Head, 3 分类 FocalLoss

### yolo11_Custom_12roi_c2 — 2 分类 + 数量约束版
- 2 分类, 新增 YOLO11ROICOUNTLOSS 数量约束损失

### yolo11_zb_12roi_c2 — Z-buffer 遮挡处理集成版
- 集成 Z-buffer 管线 (zb_func/zb_main/zb_train/zb_infer)

### yolo11_Custom_pointsize — Point-size 加权版
- 损失函数增加 conf_weight 点云密度加权

### yolo11_Custom_atten — 注意力机制引入版
- 模块化重构, 新增 LocalGrid_Attention 注意力模块

### yolo11_Custom_atten2 — 注意力联合训练版
- **改进训练策略**: 12ROI + 单张 ROI 共同训练
- 防止模型过于依赖注意力模块
- 支持任意数量图片输入（不局限 12 ROI 场景）
- **移除** dataset_check.py / dataset_filter.py
- **优化** 代码结构与训练配置
- 保持 LocalGrid_Attention 注意力模块
