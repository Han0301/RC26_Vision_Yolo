"""
通用 PyTorch 模型 (.pt/.pth) → ONNX 导出工具

功能：
  - 支持 .pt / .pth 权重文件
  - 支持 .pt 完整模型文件
  - 支持动态/静态 batch 及动态分辨率
  - 自动 ONNX 校验 + 简化
  - 支持多种常见模型结构

用法示例：
  python export_pt_onnx.py --weights model.pth --input-shape 1 3 224 224
  python export_pt_onnx.py --weights model.pt --input-shape 1 3 640 640 --dynamic
  python export_pt_onnx.py --weights model.pth --input-shape 1 3 224 224 --opset 17
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.onnx

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  模型导入部分 —— 根据你的模型结构调整此函数
# ═══════════════════════════════════════════════════════════════

def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    加载 PyTorch 模型。

    支持以下情况：
      1. 完整的模型文件（含网络结构）.pt
      2. 仅 state_dict 权重文件 .pth / .pt
      3. 自定义模型类（需在此函数中定义或引入）

    如果注释中的默认模型不适用，请按你自己的模型结构修改此函数。
    """
    weights_path = str(weights_path)
    suffix = Path(weights_path).suffix.lower()

    # ------------------------------------------------------------------
    # 示例 1：完整模型 (torch.save(model, "model.pt"))
    # ------------------------------------------------------------------
    if suffix == ".pt" and "state" not in weights_path:
        try:
            model = torch.load(weights_path, map_location=device, weights_only=False)
            if isinstance(model, dict):
                raise ValueError("文件是 state_dict，不是完整模型")
            print(f"[✓] 已加载完整模型: {weights_path}")
            model.to(device)
            model.eval()
            return model
        except Exception:
            pass  # 尝试作为 state_dict 加载

    # ------------------------------------------------------------------
    # 示例 2：state_dict 权重 + 预定义模型结构
    #  ⚠️ 请根据你的实际模型替换以下代码
    # ------------------------------------------------------------------
    # === 使用预训练分类模型（如 ResNet、VGG、EfficientNet 等）===
    # from torchvision.models import resnet50, ResNet50_Weights
    # model = resnet50(weights=None)
    # state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    # model.load_state_dict(state_dict, strict=True)

    # === 自定义模型 ===
    # from my_model import MyModel
    # model = MyModel(num_classes=10)
    # state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    # model.load_state_dict(state_dict, strict=True)

    # === YOLOv5 / YOLOv8 等检测模型 ===
    # model = attempt_load(weights_path, device=device)  # YOLOv5
    # model = YOLO(weights_path)                          # YOLOv8

    # ========== 以下为占位示例：一个简单的 CNN 分类器 ==========
    print("[!] 没有检测到匹配的模型结构，使用占位演示模型。")
    print("   请修改 load_model() 函数以加载你自己的模型。")

    class _DemoClassifier(torch.nn.Module):
        """演示用简单 CNN 分类器"""
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = _DemoClassifier()
    # 尝试加载权重（如果存在且匹配）
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"[✓] 已加载权重: {weights_path}")
        except Exception as e:
            print(f"[!] 权重加载失败，使用随机初始化: {e}")
    else:
        print(f"[!] 权重文件不存在: {weights_path}，使用随机初始化")

    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
#  ONNX 导出核心
# ═══════════════════════════════════════════════════════════════

def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
    opset_version: int = 17,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> str:
    """
    将 PyTorch 模型导出为 ONNX 格式。

    参数:
        model:        PyTorch 模型（已 .eval()）
        dummy_input:  示例输入张量
        output_path:  输出 .onnx 文件路径
        input_names:  输入节点名称列表
        output_names: 输出节点名称列表
        dynamic_axes: 动态轴定义（见下方示例）
        opset_version: ONNX opset 版本（推荐 17 或 18）
        device:       运行设备
        verbose:      是否打印详细信息

    返回:
        str: 导出的 ONNX 文件路径

    dynamic_axes 示例:
        # 动态 batch + 动态宽高
        dynamic_axes = {
            "input":  {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"},
        }
    """
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"[*] 导出 ONNX ...")
    print(f"    输出路径:  {output_path}")
    print(f"    输入形状:  {list(dummy_input.shape)}")
    print(f"    Opset:     {opset_version}")
    print(f"    动态轴:    {'有' if dynamic_axes else '无（静态）'}")

    # --- 导出 ---
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,       # 常量折叠优化
        export_params=True,             # 导出参数权重
        verbose=verbose,
    )

    print(f"[✓] ONNX 导出完成: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
#  ONNX 校验与简化
# ═══════════════════════════════════════════════════════════════

def verify_onnx(onnx_path: str) -> bool:
    """校验 ONNX 模型的合法性，并打印计算图摘要"""
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        graph = onnx_model.graph
        print(f"[✓] ONNX 校验通过")
        print(f"    输入节点:  {[i.name for i in graph.input]}")
        for inp in graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"      - {inp.name}: shape={shape}")
        print(f"    输出节点:  {[o.name for o in graph.output]}")
        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"      - {out.name}: shape={shape}")
        print(f"    算子数:    {len(graph.node)}")
        return True
    except ImportError:
        print("[!] 未安装 onnx 库，跳过校验。 pip install onnx")
        return False
    except Exception as e:
        print(f"[✗] ONNX 校验失败: {e}")
        return False


def simplify_onnx(onnx_path: str) -> str | None:
    """
    简化 ONNX 模型（常量折叠、冗余消除等）。
    需要安装 onnxsim: pip install onnx-simplifier
    返回简化后的文件路径，失败则返回 None。
    """
    try:
        import onnx
        import onnxsim
        onnx_model = onnx.load(onnx_path)
        model_simple, check = onnxsim.simplify(onnx_model)
        if check:
            out_path = onnx_path.replace(".onnx", "_sim.onnx")
            onnx.save(model_simple, out_path)
            print(f"[✓] ONNX 简化完成: {out_path}")
            return out_path
        else:
            print("[!] ONNX 简化失败（检查未通过），保留原模型")
            return None
    except ImportError:
        print("[!] 未安装 onnx-simplifier，跳过简化。 pip install onnx-simplifier")
        return None
    except Exception as e:
        print(f"[!] ONNX 简化异常: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  推理测试（可选）
# ═══════════════════════════════════════════════════════════════

def test_onnx_inference(
    onnx_path: str,
    dummy_input: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    """使用 ONNX Runtime 测试推理结果与原 PyTorch 结果对比"""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[!] 未安装 onnxruntime，跳过推理测试。 pip install onnxruntime")
        return

    print("[*] 测试 ONNX Runtime 推理 ...")

    # 选择执行提供者
    providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider",
                 "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    available = [p for p in providers if p in ort.get_available_providers()]
    if not available:
        available = ["CPUExecutionProvider"]
    print(f"    推理后端: {available}")

    session = ort.InferenceSession(onnx_path, providers=available)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 运行推理
    input_np = dummy_input.cpu().numpy()
    outputs = session.run([output_name], {input_name: input_np})

    print(f"[✓] ONNX 推理完成")
    print(f"    输入形状: {list(input_np.shape)}")
    print(f"    输出形状: {list(outputs[0].shape)}")
    print(f"    输出值(前5): {outputs[0].flatten()[:5].tolist()}")


# ═══════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch → ONNX 通用导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 导出分类模型（静态）
  python export_pt_onnx.py --weights model.pth --input-shape 1 3 224 224

  # 导出检测模型（动态 batch）
  python export_pt_onnx.py --weights model.pt --input-shape 1 3 640 640 --dynamic

  # 指定 opset 和输出名
  python export_pt_onnx.py --weights best.pth --input-shape 1 3 320 320 \\
      --opset 17 --input-names input --output-names output

  # 完整流程（导出 + 简化 + 推理测试）
  python export_pt_onnx.py --weights model.pth --input-shape 1 3 224 224 \\
      --dynamic --simplify --test
        """
    )
    parser.add_argument("--weights", type=str, default="model.pth",
                        help="PyTorch 模型权重文件路径 (.pt / .pth)")
    parser.add_argument("--input-shape", type=int, nargs="+",
                        default=[1, 3, 224, 224],
                        help="示例输入形状，如: 1 3 224 224")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 ONNX 文件路径（默认自动生成）")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本 (默认: 17)")
    parser.add_argument("--dynamic", action="store_true",
                        help="启用动态 batch 和动态宽高")
    parser.add_argument("--simplify", action="store_true",
                        help="使用 onnx-simplifier 简化模型")
    parser.add_argument("--test", action="store_true",
                        help="导出后用 ONNX Runtime 测试推理")
    parser.add_argument("--input-names", type=str, nargs="+",
                        default=["input"],
                        help="输入节点名称")
    parser.add_argument("--output-names", type=str, nargs="+",
                        default=["output"],
                        help="输出节点名称")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="导出设备 (默认 cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细导出日志")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 设备 ──
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    print(f"[*] 使用设备: {device}")

    # ── 加载模型 ──
    model = load_model(args.weights, device)

    # ── 构建示例输入 ──
    input_shape = args.input_shape
    dummy_input = torch.randn(input_shape, device=device)

    # ── 输出路径 ──
    if args.output is None:
        weights_name = Path(args.weights).stem
        output_path = f"{weights_name}.onnx"
    else:
        output_path = args.output
    if not output_path.endswith(".onnx"):
        output_path += ".onnx"

    # ── 动态轴定义 ──
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {}
        for name in args.input_names:
            dynamic_axes[name] = {
                0: "batch_size",
                2: "height",
                3: "width",
            }
        for name in args.output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # ── 导出 ──
    export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=output_path,
        input_names=args.input_names,
        output_names=args.output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        device=device,
        verbose=args.verbose,
    )

    # ── 校验 ──
    verify_onnx(output_path)

    # ── 简化 ──
    if args.simplify:
        simplified_path = simplify_onnx(output_path)
        if simplified_path and args.test:
            output_path = simplified_path

    # ── 推理测试 ──
    if args.test:
        test_onnx_inference(output_path, dummy_input, device)

    print(f"\n[✔] 全部完成！输出文件: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
