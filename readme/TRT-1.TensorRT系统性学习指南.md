# TensorRT 系统性学习指南

> 目标：从零到一掌握 TensorRT，能够将训练好的模型（PyTorch / ONNX）部署到 NVIDIA GPU 上实现推理加速。

---

## 目录

1. [前置知识](#1-前置知识)
2. [环境搭建](#2-环境搭建)
3. [核心概念（必须理解）](#3-核心概念必须理解)
4. [TensorRT 工作流](#4-tensorrt-工作流)
5. [学习路线图（按优先级排列）](#5-学习路线图按优先级排列)
6. [实践项目清单](#6-实践项目清单)
7. [常见问题与调试技巧](#7-常见问题与调试技巧)
8. [推荐学习资源](#8-推荐学习资源)

---

## 1. 前置知识

在开始 TensorRT 之前，确保你已经掌握以下内容：

| 知识领域 | 具体要求 |
|---------|---------|
| **Python** | 熟练使用，能看懂 Python/C++ 混合工程 |
| **PyTorch** | 会用 `torch.onnx.export()` 导出模型，理解 `torch.nn.Module` |
| **ONNX** | 理解计算图、节点、张量、动态轴概念 |
| **CUDA 基础** | 了解 GPU 内存模型、`cudaMemcpy`、stream、kernel 概念（不要求手写） |
| **深度学习基础** | 理解卷积、全连接、激活函数、BatchNorm、残差连接等常见算子 |

> 💡 如果 ONNX 还不熟，先花 1~2 天学习 ONNX 基础（[onnx.ai](https://onnx.ai/)）。

---

## 2. 环境搭建

### 2.1 硬件要求

- **必须**：NVIDIA GPU（Compute Capability ≥ 7.0，即 GTX 1060 6GB+ / RTX 20 系列以上）
- **推荐**：RTX 3060+ / Tesla T4+（支持 FP16 / INT8）

查看你的 GPU Compute Capability：
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### 2.2 软件安装

```bash
# 1. 确认 CUDA 版本
nvidia-smi                         # Driver 版本
nvcc --version                     # CUDA Toolkit 版本（需 ≥ 11.x）

# 2. 安装 TensorRT（推荐 pip，但需要先安装 TensorRT .deb/.exe）
#    或直接从 NVIDIA 官网下载后安装 .whl
pip install tensorrt               # 需系统已安装 TensorRT 库

# 3. 验证安装
python -c "import tensorrt as trt; print(trt.__version__)"

# 4. 安装 pycuda（Python 端推理时需要）
pip install pycuda

# 5. (可选) 安装 onnx / onnxruntime 用于调试
pip install onnx onnxruntime-gpu
```

> ⚠️ **常见坑**：`pip install tensorrt` 只装 Python 绑定，底层库需要从 NVIDIA 官网下载 TensorRT 安装包。推荐使用 **TensorRT Docker 镜像** 省去环境配置。

### 2.3 Docker 快速起步（推荐）

```bash
# TensorRT 官方 Docker（包含 CUDA + cuDNN + TensorRT）
docker pull nvidia/cuda:12.4.1-devel-ubuntu22.04
# 或在 NGC 上拉取专用镜像
docker pull nvcr.io/nvidia/tensorrt:24.12-py3
```

---

## 3. 核心概念（必须理解）

这是学习 TensorRT **最重要**的部分，不理解这些概念就无法正确使用 TensorRT。

### 3.1 构建期 (Build Time) vs 推理期 (Inference Time)

```
PyTorch模型 ──► ONNX ──► Builder ──► .engine ──► Runtime ──► 推理
                         构建期                   推理期
```

- **构建期**：慢，消耗大量显存和算力，生成优化后的 engine
- **推理期**：快，加载 engine 后直接推理，不依赖 TensorRT Builder

### 3.2 Engine / Plan

- `engine` 是 TensorRT 编译优化后的最终产物，二进制文件（`.engine` / `.plan`）
- 包含：网络结构 + 算子融合 + 内存优化 + kernel 选择
- **与 GPU 型号绑定**：不同 GPU 需重新 build（`compute_cap` 相同可以通用）

### 3.3 Builder / Config / Network

```
Builder ──► create_network() ──► Network (计算图)
    │
    └── create_builder_config() ──► Config (配置)
          │
          ├── workspace 大小
          ├── FP16 / INT8 精度
          └── Optimization Profile
```

### 3.4 精度模式

| 模式 | 速度提升 | 精度损失 | 硬件要求 |
|------|---------|---------|---------|
| **FP32** | 1x (baseline) | 无 | 所有 GPU |
| **FP16** | ~2x | 几乎无 | Compute Capability ≥ 7.0 |
| **INT8** | ~3-4x | 轻微（需校准） | Compute Capability ≥ 7.0 |
| **TF32** | ~1.5x | 几乎无 | Ampere (SM 8.0) 及以上 |

### 3.5 动态形状 (Dynamic Shapes) — **最高频坑**

```
Optimization Profile
├── min shape:  (1, 1, 3, 64, 64)
├── opt shape:  (1, 12, 3, 64, 64)  ← 以这个形状做最优优化
└── max shape:  (1, 24, 3, 64, 64)

推理时调用: context.set_binding_shape(0, actual_shape)
```

- 静态形状：形状固定，推理最快
- 动态形状：灵活但性能略低，需设置 profile
- 原则：**能用静态就不用动态**，如果 num_roi 变化不大，padding 到固定值

### 3.6 算子融合 (Operator Fusion)

TensorRT 的核心加速手段之一：

```
原始: Conv → BN → ReLU → Conv → BN → ReLU
融合: CBR → CBR (一个 kernel 完成 Conv+BN+ReLU)
```

常见的融合模式：
- Conv + Bias + ReLU → CBR
- Conv + BatchNorm + ReLU → CBR
- 多个小 Concat → 消除
- LayerNorm 融合

### 3.7 TensorRT 上下文 (ExecutionContext)

- `engine.create_execution_context()` 创建推理上下文
- 每个 context 占用独立的 GPU 内存
- 多线程推理时可以创建多个 context（共享 engine）

### 3.8 IExecutionContext / 绑定 (Binding)

```python
context = engine.create_execution_context()
# binding 0 = 输入, binding 1 = 输出
context.set_binding_shape(0, input.shape)
output_shape = tuple(context.get_binding_shape(1))
```

---

## 4. TensorRT 工作流

标准流程只有 5 步：

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 导出模型  │───►│ 创建Builder│───►│ 构建Engine │───►│ 序列化保存 │───►│ 加载推理  │
│ (ONNX)   │    │ + Network │    │           │    │ .engine  │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 4.1 典型 Python 代码骨架

```python
import tensorrt as trt

# 1. 日志 + Builder
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# 2. 创建 Network（动态形状必须用 EXPLICIT_BATCH）
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# 3. 解析 ONNX
parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 4. 配置
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

# 5. (可选) FP16
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# 6. (可选) 动态形状
profile = builder.create_optimization_profile()
profile.set_shape("input", (1,1,3,64,64), (1,12,3,64,64), (1,24,3,64,64))
config.add_optimization_profile(profile)

# 7. 构建
engine = builder.build_serialized_network(network, config)

# 8. 保存
with open("model.engine", "wb") as f:
    f.write(engine)
```

---

## 5. 学习路线图（按优先级排列）

按顺序学习，每个阶段都要动手实践。

### 🟢 阶段一：基础入门（1~2 天）

| 学习内容 | 目标 | 实践 |
|---------|------|------|
| 环境搭建 | 能成功 `import tensorrt` | 安装验证 |
| ONNX 导出 | 掌握 `torch.onnx.export()` 动态轴配置 | 导出你的 YOLO ROI 模型 |
| `trtexec` 使用 | 能用命令行转换和 benchmark | `trtexec --onnx=... --saveEngine=...` |
| Builder 基本流程 | 理解 Python API 的 builder→network→engine 流程 | 跑通官方 sample `sample_onnx.py` |

**关键练习**：
```bash
# 用 trtexec 转换你的 ONNX 模型
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
# trtexec 会自动 benchmark，看 latency 和 throughput
```

### 🟡 阶段二：核心 API 掌握（2~3 天）

| 学习内容 | 说明 |
|---------|------|
| **Network API** | `network.add_convolution()`, `network.add_activation()` 等（了解即可，通常用 ONNX Parser） |
| **Builder Config** | workspace 大小、精度标志、DLA 等 |
| **Optimization Profile** | 动态形状的 min/opt/max 配置 |
| **序列化与反序列化** | `serialize()` / `deserialize_cuda_engine()` |
| **ExecutionContext** | `set_binding_shape()`, `execute_async_v2()` |
| **Python 端推理** | 用 pycuda 管理 GPU 显存 |

**关键练习**：
- 编写完整的 Python 脚本：ONNX → engine → 推理验证
- 使用 `context.set_binding_shape()` 处理动态形状
- 对比 FP32 vs FP16 的精度和速度

### 🟠 阶段三：进阶优化（3~5 天）

| 学习内容 | 说明 |
|---------|------|
| **INT8 量化** | 需要校准数据集 `Int8Calibrator`，理解 KL 散度校准 |
| **Polygraphy** | NVIDIA 官方调试工具，查看网络层、比较精度 |
| **NVTools / Nsight** | profiling TensorRT kernel 性能 |
| **Layer 调试** | 用 `network.get_layer(i)` 查看每一层的名称/精度 |
| **Refit** | 不重新 build 只更新权重 |
| **多 stream 推理** | 利用 CUDA stream 并行推理 |

**关键练习**：
- 对模型做 INT8 量化，对比精度损失
- 用 Polygraphy 对比 ONNX 和 TensorRT 的输出差异
- 用 `trtexec --profilingVerbosity=detailed` 分析每层耗时

### 🔴 阶段四：工程化部署（5~7 天）

| 学习内容 | 说明 |
|---------|------|
| **C++ API** | 生产环境通常用 C++，性能更好 |
| **TRT 插件 (Plugin)** | 实现自定义算子，继承 `IPluginV2DynamicExt` |
| **TensorRT 与推理框架集成** | Triton Inference Server, TensorFlow-TRT, PyTorch-TRT |
| **多 batch 优化** | 动态 batch 和 padding 策略 |
| **显存优化** | `IExecutionContext` 复用、显存池管理 |
| **模型部署流水线** | 自动化 CI/CD 构建 engine |

---

## 6. 实践项目清单

> 动手实践是最好的学习方式。按顺序完成以下项目：

### 项目 1：简单分类模型（入门）

```
ResNet-18 / MobileNet 导出 ONNX → TensorRT
验证：top-1 accuracy 不变，速度提升 2-3x
```

### 项目 2：你的 YOLO ROI 模型（进阶）✅ (你已经完成)

```
已完成：ONNX → TensorRT 动态形状 engine
待完成：INT8 量化、Polygraphy 精度验证
```

### 项目 3：检测模型（YOLOv8/v11）

```
YOLOv8 导出 ONNX → TensorRT
包括：pre/post-processing 合并到 engine 中？
实践：EfficientNMS plugin
```

### 项目 4：自定义插件

```
实现一个 TensorRT Plugin（如 Swish activation）
注册 plugin → 在 ONNX 中用自定义 op → build engine
```

### 项目 5：生产级部署

```
C++ 加载 engine + 多线程推理 + Triton Server
包含：预热(warm-up)、动态 batch、延迟/吞吐量压测
```

---

## 7. 常见问题与调试技巧

### 7.1 ONNX Parser 解析失败

```python
# 查看具体错误
for i in range(parser.num_errors):
    print(parser.get_error(i))
```

常见原因：
- ONNX opset 版本太高（TensorRT 不支持最新 opset）
- 某些算子 TensorRT 不支持（如 `torch.where`, `torch.topk` 的某些变体）
- **解决**：简化模型或用 plugin

### 7.2 动态形状不生效

```
症状：推理时形状不对报错
原因：构建时没设 profile，或推理时没调用 set_binding_shape()
```

### 7.3 engine 构建失败

```bash
# 方法1: 减少 workspace 大小
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

# 方法2: 关闭 FP16 看是否 FP32 能过
# 方法3: 用 trtexec --verbose 看详细日志
trtexec --onnx=model.onnx --saveEngine=model.engine --verbose
```

### 7.4 精度对不齐（TensorRT 输出 vs ONNX 输出）

```python
# 用 Polygraphy 比较中间层
polygraphy run model.onnx --trt --onnxrt \
    --input-shapes roi_imgs:[1,12,3,64,64] \
    --check-error-stat median
```

常见原因：
- FP16 导致精度损失（对精度敏感的层应强制 FP32）
- INT8 校准数据不够代表性
- 某些算子实现差异

### 7.5 性能不如预期

```bash
# 1. trtexec benchmark
trtexec --loadEngine=model.engine --best

# 2. 查看 kernels 耗时
trtexec --loadEngine=model.engine --profilingVerbosity=detailed

# 3. 用 Nsight Systems 看 kernel 占用
nsys profile -o profile_output trtexec --loadEngine=model.engine --best
```

---

## 8. 推荐学习资源

### 官方文档（首要）

| 资源 | 链接 |
|------|------|
| TensorRT Developer Guide | https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/ |
| TensorRT Python API 文档 | https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/ |
| TensorRT C++ API 文档 | https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/ |
| TensorRT 官方 Sample | https://github.com/NVIDIA/TensorRT/tree/main/samples |
| TensorRT 官方 Python Sample | https://github.com/NVIDIA/TensorRT/tree/main/quickstart |

### 工具链

| 工具 | 用途 | 命令 |
|------|------|------|
| **trtexec** | 转换 + benchmark | TensorRT 自带 |
| **Polygraphy** | 调试 + 精度对比 | `pip install polygraphy` |
| **Nsight Systems** | 性能分析 | CUDA 工具包自带 |
| **Netron** | 可视化 ONNX 图 | `pip install netron` |
| **ONNX GraphSurgeon** | 修改/简化 ONNX 图 | `pip install nvidia-pyindex && pip install onnx-graphsurgeon` |

### 学习路径建议

```
第1周：环境搭建 + trtexec + Python API 基本流程（阶段一）
第2周：动态形状 + FP16 + 完整推理代码（阶段二）
第3周：INT8量化 + Polygraphy调试（阶段三）
第4周：C++部署 + 插件开发 + 生产环境优化（阶段四）
```

---

## 附录：快速命令速查表

```bash
# ─── trtexec ─────────────────────────────────────────
# ONNX → Engine（静态形状）
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# ONNX → Engine（动态形状）
trtexec --onnx=model.onnx --saveEngine=model.engine \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:32x3x224x224 \
        --fp16

# Benchmark engine
trtexec --loadEngine=model.engine --best

# 详细 profiling
trtexec --loadEngine=model.engine --profilingVerbosity=detailed

# ─── Polygraphy ──────────────────────────────────────
# 对比 ONNX Runtime 与 TensorRT 输出
polygraphy run model.onnx --trt --onnxrt \
    --input-shapes roi_imgs:[1,12,3,64,64]

# 查看网络层信息
polygraphy inspect model model.onnx

# ─── ONNX ────────────────────────────────────────────
# 验证 ONNX 模型
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# 简化 ONNX
python -m onnxsim model.onnx model_simplified.onnx
```

---

> **最后建议**：不要一次性看完所有资料。**先跑通一个最小 demo**（比如用 `trtexec` 转换一个分类模型），然后再逐步深入每个概念。带着问题去查官方文档，效率最高。
