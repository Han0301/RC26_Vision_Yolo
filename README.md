# RC26 Vision YOLO

本仓库整合了 RC26 视觉检测项目与 ONNX/TensorRT 学习教程。

## 仓库结构

```
RC26_Vision_Yolo/
├── legacy/                          # 原 RC26 项目源代码
│   ├── README.md                    # 原项目说明文档
│   ├── dataset_func.py              # 数据集处理函数
│   ├── dataset_main.py              # 数据集处理主程序
│   ├── infer_func.py                # 推理函数
│   ├── infer_main.py                # 推理主程序
│   ├── infer_test.py                # 推理测试
│   ├── load_model.py                # 模型加载
│   ├── loss.py                      # 损失函数
│   ├── main.py                      # 主入口
│   ├── model.py                     # 模型定义
│   ├── pt_to_onnx_openvino.py       # PyTorch 转 ONNX/OpenVINO
│   ├── show_atten.py                # 注意力可视化
│   ├── train_config.py              # 训练配置
│   ├── train_func.py                # 训练函数
│   └── train_main.py                # 训练主程序
│
├── tutorial/                        # ONNX/TensorRT 学习教程
│   ├── readme/                      # 教程文档
│   │   ├── ONNX-0.numpy基础与ndarray详解.md
│   │   ├── ONNX-1.计算图数据结构深度解析.md
│   │   ├── ONNX-2.推理引擎原理.md
│   │   ├── ONNX-3.图像检测应用实战.md
│   │   ├── ONNX-4.预处理原理与代码详解.md
│   │   ├── ONNX-5.模型推理与ONNX Runtime详解.md
│   │   ├── ONNX-6.后处理YOLO解码与NMS详解.md
│   │   ├── ONNX-7.检测结果可视化详解.md
│   │   └── TRT-1.TensorRT系统性学习指南.md
│   ├── src/                         # 示例代码
│   │   ├── export_pt_onnx.py        # PyTorch 模型导出为 ONNX
│   │   ├── onnx_1.py                # ONNX 基础操作示例
│   │   └── onnx_detect_demo.py      # ONNX 检测演示
│   └── model/                       # 预训练模型文件
│       ├── best.onnx
│       └── best.pt
│
├── .vscode/                         # VS Code 配置
└── README.md                        # 本文件
```

## 快速开始

### 原项目（legacy）

```bash
cd legacy
# 参见 legacy/README.md 获取详细使用说明
```

### 学习教程（tutorial）

```bash
cd tutorial/src
python onnx_1.py          # ONNX 基础操作
python onnx_detect_demo.py # ONNX 检测演示
python export_pt_onnx.py  # 模型导出
```
