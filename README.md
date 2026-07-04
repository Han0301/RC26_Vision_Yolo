# RC26 Vision YOLO — YOLOv11 自定义模型训练工作区

> 基于 Ultralytics YOLOv11 的 12-ROI 方块检测模型，从 3 分类到注意力机制的完整迭代历程。

## 项目背景

RC26 机器人竞赛中，需要从仿真环境的 12 个固定 ROI（Region of Interest）中判断每个位置是否有方块。因为所有的 ROS 图像都是仿真图片，场景相对受限，所以采用自定义 YOLOv11 分类头的方式，在预训练 backbone 上接轻量级 Neck + Head，直接对每个 ROI 进行二分类/三分类。

整个项目经历了 6 个版本的迭代，每次迭代都针对实际测试中暴露的问题进行改进。

## 使用方法

```bash
pip install torch ultralytics openvino opencv-python

# 训练 (版本不同入口不同)
python train.py              # v1.x 系列
python train_main.py         # v2.x 系列

# 推理
python infer.py              # v1.x 系列
python infer_main.py         # v2.x 系列

# 注意力可视化 (v2.x)
python show_atten.py

# 模型导出
python pt_to_onnx_openvino.py
```

## 版本迭代日志

### yolo11_Custom_12roi (v1.0) — 基础 3 分类版

**问题背景**: 首次接入 YOLOv11 做 12-ROI 检测。每个 ROI 可能出现三种状态：该 ROI 无效（在视野外/被遮挡）、有效但无方块、有效且有方块。

**方案与实现**:
- 复用 YOLOv11 的 backbone 做特征提取，设计 `YOLO11ROIBackbone → YOLO11ROINeck → YOLO11ROIHead` 的流水线
- 分类头输出 `[B, 12, 3]` 的三分类概率（0=无效ROI, 1=有效无方块, 2=有效有方块）
- 使用 `Focal Loss` 缓解类别不平衡（更多 ROI 属于"无方块"类别）
- 支持 3 种模型尺寸（n/s/l）

**遗留问题**: 3 分类在实际测试中发现"无效 ROI"和"有效无方块"的边界模糊，影响模型收敛。

### yolo11_Custom_12roi_c2 (v1.1) — 2 分类 + 数量约束版

**发现问题**: 
1. 仿真环境中所有 ROI 都位于视野内，"无效 ROI"这个类别缺乏实际意义，反而增加了分类难度
2. 模型预测的方块数量与实际严重不符（比如实际 8 个方块预测出 12 个）

**解决方案**:
- **3 分类 → 2 分类**: 删除"无效 ROI"类，改为"无方块/有方块"二分类。这是更符合实际问题定义的决策边界
- **新增 `YOLO11ROICOUNTLOSS`**: 统计预测的正样本数量，与实际数量计算 MSE 损失，反向传播约束模型输出数量
- 对比实验 `count_loss_weight = [0.0, 0.1, 0.2, 0.25]`，结论是无数量约束（weight=0.0）时准确率最高（97.6%），因为 count_loss 会与 cls_loss 竞争

**效果**: 全局准确率 97.6%，高置信度（>0.9）准确率 99.7%

### yolo11_zb_12roi_c2 (v1.2) — Z-buffer 遮挡处理集成版

**发现问题**: 当方块被其他物体（如机械臂、其他方块）遮挡时，模型仅凭 RGB 图像容易漏检。仿真环境中可以获取深度信息，但之前没有利用起来。

**解决方案**:
- 利用仿真环境提供的深度图，引入 **Z-buffer（深度缓冲）遮挡处理**管线
- `zb_func.py`: Z-buffer 核心函数，判断 3D 空间中哪些区域被遮挡
- `zb_main.py`: 编排 Z-buffer 处理流程，与模型推理串联
- `zb_train.py` / `zb_infer.py`: 在训练和推理时注入深度信息辅助判断
- `zb_dataset.py`: 加载深度数据作为额外输入

**效果**: 遮挡场景的漏检率显著降低

### yolo11_Custom_pointsize (v1.3) — Point-size 加权版

**发现问题**: 
1. 不同 ROI 对应的点云密度（Point Size）差异很大——点云密的区域图像信息丰富，模型应该更有信心判断；点云稀的区域信息不足，预测应更保守
2. 但模型对所有 ROI 一视同仁，导致高密度区域的高置信度预测被浪费

**解决方案**:
- 在所有损失函数（FocalLoss、BCELoss、CountLoss）中新增 **`conf_weight`** 参数
- conf_weight 由点云密度归一化得到：`weighted_loss = loss * conf_weight`
- density ROI 的损失被放大 → 模型更关注这些区域的学习
- sparse ROI 的损失被缩小 → 模型不会过度拟合噪声

**效果**: 模型在高 point-size 区域给出更高置信度预测，整体预测质量提升

### yolo11_Custom_atten (v2.0) — 注意力机制引入版

**发现问题**: 之前的模型把 12 个 ROI 当作独立样本处理，但它们在空间上是关联的。例如：(1) 相邻 ROI 的检测结果应该互相影响（一个方块不会悬空）。(2) 全局上下文有助于理解整体布局。

**解决方案 — 架构大重构**:
- **新增 `LocalGrid_Attention`**: 局部网格注意力模块。ROI 特征通过多头自注意力交互，每个 ROI 的最终表示融合了相邻 ROI 的上下文信息
- **残差连接 + 可调权重**: `output = norm(roi_feat + atten_weight * attn_feat)`，atten_weight 控制注意力信息的融合强度
- **模块化重构**: 将庞大的单文件拆分为 func/main 分离架构：
  - `dataset_func.py` / `dataset_main.py` — 数据加载逻辑与流程解耦
  - `infer_func.py` / `infer_main.py` — 推理逻辑独立
  - `train_func.py` / `train_main.py` / `train_config.py` — 训练配置集中管理
  - `load_model.py` — 模型加载统一入口
  - `dataset_check.py` / `dataset_filter.py` — 数据质量保证
- **新增 `show_atten.py`**: 注意力热力图可视化，帮助 Debug 注意力模块是否正常工作
- **损失函数重命名**: `CountLoss` / `FocalLoss` / `BCELoss`（更清晰）

**好处**: 
- ROI 之间可以"交流"信息，上下文感知能力提升
- 模块化架构便于后续各个击破地改进
- 注意力可视化工具帮助理解模型行为

### yolo11_Custom_atten2 (v2.1) — 注意力联合训练版

**发现问题**: 
1. v2.0 只在 12 ROI 的固定网格上训练，但实际场景可能需要处理任意数量的 ROI（不一定是 12 个）
2. 模型过度依赖注意力模块，忽略了单张 ROI 自身的判别特征。比如两张距离很远的 ROI，应该主要靠自身特征判断，而不是靠邻居的信息

**解决方案**:
- **12 ROI + 单张 ROI 联合训练**：既保留 12 ROI 的注意力建模优势，又通过单张 ROI 训练强制 backbone 提取独立判别特征
- 训练时随机选择 12 ROI 模式或单 ROI 模式，防止模型过依赖注意力
- 因为训练数据包含了单 ROI 场景，推理时也支持任意数量 ROI 输入

**效果**: 
- 模型既具备上下文感知能力（注意力），又有独立判别能力
- 推理时支持灵活的 ROI 数量，部署更通用
- 注意力模块不会成为瓶颈
