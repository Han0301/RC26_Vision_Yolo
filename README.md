# RC26 Vision YOLO

## 简介

RC26 Vision YOLO 是一个面向固定 12 个感兴趣区域（ROI）的二分类工作空间。模型将同一样本的 12 张 ROI 图像合并到批次维度，在一次 Backbone 调用中并行提取特征；随后把特征恢复为 ROI 序列，通过多头自注意力建模不同位置之间的关系，最终为每个 ROI 输出“无目标 / 有目标”两类 logits。

项目基于 PyTorch 与 Ultralytics YOLO11 模块实现，包含数据加载、YOLO11 预训练权重迁移、混合训练、推理评估、注意力可视化，以及 ONNX / OpenVINO 导出代码。当前默认任务配置为 12 个 ROI、每个 ROI 为 `64 × 64` RGB 图像、2 个类别。

## 仓库结构

```text
RC26_Vision_Yolo-main/
├── README.md                   # 项目说明（唯一 README）
├── scripts/
│   ├── model.py               # Backbone、Neck、多头自注意力与分类头
│   ├── load_model.py          # YOLO11 预训练权重映射与断点加载
│   ├── dataset_main.py        # 12 ROI 数据集定义与标签读取
│   ├── dataset_func.py        # 数据增强、DataLoader 与混合数据集
│   ├── loss.py                # Focal、BCE 与数量约束损失
│   ├── train_config.py        # 模型、数据集、训练及损失配置
│   ├── train_main.py          # 12 ROI / 单 ROI 交替训练流程
│   ├── train_func.py          # 验证流程与二分类指标
│   ├── infer_main.py          # 单样本及数据集推理接口
│   ├── infer_func.py          # 置信度、位置、point_size 统计与结果保存
│   ├── infer_test.py          # 动态 ROI 数量推理示例
│   ├── show_atten.py          # 单样本及数据集平均注意力可视化
│   ├── pt_to_onnx_openvino.py # PyTorch → ONNX → OpenVINO IR
│   └── main.py                # 实验入口示例
├── .vscode/
├── .gitignore
└── update_payload.json
```

## 快速开始

### 1. 准备环境

建议使用 Python 3.9 及以上版本，并根据本机 CUDA 环境安装对应版本的 PyTorch。

```bash
git clone https://github.com/Han0301/RC26_Vision_Yolo.git
cd RC26_Vision_Yolo/scripts

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install torch torchvision ultralytics opencv-python numpy tqdm prompt-toolkit matplotlib
```

模型导出时还需要：

```bash
pip install onnx openvino
```

### 2. 准备数据集

每个样本由一个包含 `1.png` 到 `12.png` 的 ROI 文件夹，以及一个同编号 JSON 标签文件组成：

```text
dataset_root/
├── roi_images/
│   ├── roi_0/
│   │   ├── 1.png
│   │   ├── ...
│   │   └── 12.png
│   └── roi_1/
└── labels/
    ├── label_0.json
    └── label_1.json
```

标签文件必须包含长度为 12 的 `labels` 与 `point_size`：

```json
{
  "labels": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
  "point_size": [0.0, 12.4, 9.8, 0.0, 8.1, 7.5, 0.0, 11.2, 6.9, 0.0, 10.3, 8.8]
}
```

其中 `labels[i]` 是第 `i + 1` 个 ROI 的类别；`point_size` 会按 `[0,1,2]`、`[3,4,5]`、`[6,7,8]`、`[9,10,11]` 四组归一化为置信权重。

### 3. 配置并训练

先修改 `scripts/train_config.py` 中的以下路径：

- `model_config["YOLO_weight_path"]`：与 `MODEL_SIZE` 对应的 YOLO11 预训练权重；
- `dataset_config["DATASET_ROOTS"]`：一个或多个数据集根目录；
- `train_config["SAVE_DIR"]`：训练权重输出目录。

在 `scripts` 目录中运行：

```bash
python train_main.py
```

训练流程会交替使用完整 12 ROI 批次和全覆盖单 ROI 批次。默认采用 AdamW、CosineAnnealingLR、梯度裁剪、Focal Loss 与数量约束损失，并根据完整 / 单 ROI 验证集的平均正类 F1 保存 `best_model.pt`。

### 4. 推理

```python
from infer_main import YOLO11ROIInferencer

inferencer = YOLO11ROIInferencer(
    model_path="./yolo11_pt/best_model.pt",
    model_size="s",
    roi_size=64,
    num_roi=12,
    num_classes=2,
)

classes, probabilities = inferencer.infer_roi_folder(
    "./example_dataset/roi_images/roi_0",
    label_path="./example_dataset/labels/label_0.json",
    is_print=True,
)
```

`classes` 的形状为 `[12]`，`probabilities` 的形状为 `[12, 2]`。`infer_datasets()` 还支持按置信度阈值、ROI 位置和 `point_size` 权重统计结果，并输出 CSV。

> 当前部分示例配置仍保留原始 Windows 绝对路径。首次运行前请将其替换为本机路径；同级模块使用直接导入，因此建议从 `scripts/` 目录启动脚本。

## 核心模型架构设计

### 整体数据流

```text
输入 [B, N, 3, 64, 64]
        │
        ├─ 展平 ROI 维度 → [B×N, 3, 64, 64]
        │
        ├─ 共享 YOLO11-style Backbone（单次批量调用）
        │     Conv → Conv → C2f → Conv → C2f → Conv → C2f → SPPF
        │
        ├─ Neck：C2f → 1×1 Conv → Global Average Pooling
        │
        ├─ 恢复 ROI 序列 → [B, N, C]
        │
        ├─ Multi-Head Self-Attention
        │     LayerNorm(x + attention_weight × MHA(x, x, x))
        │
        └─ 共享分类头 → [B, N, 2]
```

`N` 通常为 12，但前向过程可接收动态 ROI 数量。所有 ROI 共享同一套 Backbone、Neck 与分类头参数；“一次前向 Backbone”指将 12 个 ROI 合并为 `B × 12` 的批次后，只调用一次共享 Backbone，而不是分别循环 12 次。

### Backbone 与 Neck

Backbone 复用 Ultralytics 的 `Conv`、`C2f` 和 `SPPF` 模块。对于默认 `s` 模型，`64 × 64` ROI 经四次步长为 2 的下采样后得到 `256 × 4 × 4` 特征图；Neck 再通过 C2f、`1 × 1` 卷积与全局平均池化压缩为每个 ROI 的 64 维向量。

项目提供三档规模：

| 配置 | Backbone 末端通道 | Neck 输出维度 | 注意力头数 | 分类头隐藏维度 |
| --- | ---: | ---: | ---: | ---: |
| `n` | 128 | 32 | 2 | 16 |
| `s` | 256 | 64 | 4 | 32 |
| `l` | 512 | 128 | 4 | 64 |

`load_YOLO_weights()` 会把官方 YOLO11 的第 0–7 层映射到自定义 Backbone、第 8–9 层映射到 Neck，并跳过检测头及形状不匹配的参数。

### 多头自注意力融合

Neck 输出恢复为 `[B, N, C]` 后，模型对 ROI 序列执行自注意力，让每个位置根据其他 ROI 的语义特征进行上下文融合：

```text
y = LayerNorm(x + α · MultiHeadAttention(x, x, x))
```

其中 `α` 对应 `atten_weight`，用于控制跨 ROI 信息对原始局部特征的影响。注意力权重保存在 `model.attn_weights`，可通过 `show_atten.py` 生成单样本 `12 × 12` 热力图或数据集平均热力图。当 `N = 1` 时，模型跳过注意力计算，仅执行 LayerNorm，以兼容单 ROI 训练和推理。

### 分类头与训练目标

分类头由 `1 × 1 Conv + Dropout + Linear` 组成，对每个 ROI 共享参数，输出 2 类 logits。训练目标包括：

- **Focal Loss**：使用类别权重与难样本调制因子，缓解类别不均衡；
- **Count Loss**：对 12 ROI 样本中正类概率总和施加约束，默认期望正类数量为 8；单 ROI 输入时自动关闭；
- **混合训练**：完整 12 ROI 批次学习跨位置关系，全覆盖单 ROI 批次强化局部分类能力。

## 推理评估与可视化

数据集推理支持以下分析：

- 不同置信度阈值下的类别覆盖率与准确率；
- 12 个固定位置各自的准确率；
- 基于 `point_size` 归一化权重的分段统计；
- 逐 ROI 预测概率、类别、标签和 point size 的 CSV 输出；
- 单样本及数据集平均注意力矩阵可视化。

## 模型导出

在 `scripts/pt_to_onnx_openvino.py` 中设置权重与输出路径后运行：

```bash
python pt_to_onnx_openvino.py
```

脚本依次完成 PyTorch 权重加载、ONNX 导出与校验、OpenVINO IR 转换和 CPU 编译验证。导出的输入、输出同时声明 batch 与 ROI 数量为动态轴，并使用 1、7、12 个 ROI 的随机输入进行形状验证。

## 注意事项

- 训练、推理和注意力可视化时，`model_size`、`num_classes`、`roi_size`、`atten_weight` 必须与权重保存时一致。
- 数据增强与推理预处理的归一化参数应保持一致；修改其中一处时请同步检查其他入口。
- `dataset_main.py` 当前按 `0–49999` 扫描样本编号，数据集编号超出该范围时需要同步调整。
- `infer_datasets()` 中仍有 Windows 路径分隔符写法；在 Linux/macOS 上批量推理前建议改为 `os.path.join()`。
- ONNX 导出中的 Python 条件分支会按示例输入路径进行追踪；若需要同一个导出模型同时严格覆盖单 ROI 与多 ROI 分支，请额外核对导出图和目标运行时行为。
