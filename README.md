# RC26 Vision YOLO — YOLOv11 自定义模型训练工作区

> 基于 YOLOv11 的 12-ROI 方块检测模型训练与推理。

## 使用方法

```bash
pip install torch ultralytics openvino opencv-python
python train.py      # 训练
python infer.py      # 推理
python zb_main.py    # Z-buffer 主流程
```

## 版本迭代日志

### yolo11_Custom_12roi — 基础 3 分类版
### yolo11_Custom_12roi_c2 — 2 分类 + 数量约束版
### yolo11_zb_12roi_c2 — Z-buffer 遮挡处理集成版

### yolo11_Custom_pointsize — Point-size 加权版
- **新增** conf_weight 参数: 损失函数中点云密度置信度加权
- **新增** YOLO11ROIBCEWithLogitsLoss2C: BCE 损失变体
- **新增** log_to_datasets.py: 日志文件→数据集转换
- **新增** post_processing.py: 推理结果后处理
- **新增** realdata_trans.py: 真实数据格式转换
- **新增** evlate/: 评估工具目录
- **移除** zb_infer.py / zb_train.py (功能合并至 zb_main.py)
- **修改** loss.py: 所有损失函数增加 conf_weight 输入
- **修改** model.py / train.py / infer.py: 适配 point-size 加权
