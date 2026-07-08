"""
ONNX 目标检测推理完整示例 (YOLOv5)
支持: 图片 / 视频 / 摄像头

用法:
    python onnx_detect_demo.py --image path/to/image.jpg
    python onnx_detect_demo.py --video path/to/video.mp4
    python onnx_detect_demo.py --webcam 0
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ═══════════════════════════════════════════════════════════════
#  常量定义 (根据你的模型调整)
# ═══════════════════════════════════════════════════════════════

MODEL_INPUT_SIZE = (640, 640)          # 模型输入尺寸 (w, h)
CONF_THRESHOLD = 0.25                   # 置信度阈值
NMS_THRESHOLD = 0.45                    # NMS IoU 阈值

# YOLOv5 的 anchor 和 stride (预定义)
STRIDES = [8, 16, 32]
ANCHORS = np.array([
    [[10, 13], [16, 30], [33, 23]],     # 小目标 (80x80 grid)
    [[30, 61], [62, 45], [59, 119]],    # 中目标 (40x40 grid)
    [[116, 90], [156, 198], [373, 326]], # 大目标 (20x20 grid)
], dtype=np.float32)

# 类别名称 (根据你的模型修改)
CLASS_NAMES = ["class1", "class2"]      # 当前模型是 2 类检测


# ═══════════════════════════════════════════════════════════════
#  1. 加载 ONNX 模型
# ═══════════════════════════════════════════════════════════════

def load_model(onnx_path: str, device: str = "cpu") -> tuple:
    """
    加载 ONNX 模型，支持 CPU / CUDA / TensorRT 后端

    参数:
        onnx_path: .onnx 文件路径
        device:    "cpu" / "cuda" / "tensorrt"

    返回:
        session, input_name, output_name
    """
    providers = []

    if device == "tensorrt":
        providers.append(("TensorrtExecutionProvider", {
            "device_id": 0,
            "trt_max_workspace_size": 4 << 30,
            "trt_fp16_enable": True,
        }))

    if device == "cuda":
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
        }))

    providers.append("CPUExecutionProvider")  # 最后回退

    session = ort.InferenceSession(onnx_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"[✓] 模型加载成功")
    print(f"    输入: {input_name} {session.get_inputs()[0].shape}")
    print(f"    输出: {output_name} {session.get_outputs()[0].shape}")
    print(f"    后端: {session.get_providers()}")

    return session, input_name, output_name


# ═══════════════════════════════════════════════════════════════
#  2. 图像预处理
# ═══════════════════════════════════════════════════════════════

def letterbox(
    image: np.ndarray,
    target_size: tuple = (640, 640),
    color: tuple = (114, 114, 114),
) -> tuple:
    """
    Letterbox 填充 — 保持宽高比，短边填充至目标尺寸。
    这是检测模型最常用的预处理方式。

    返回: (填充后的图像, 缩放比例, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例（取最小缩放，保证完整图像可见）
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算 padding
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2

    # 创建画布并填充
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    canvas[int(pad_h):int(pad_h) + new_h, int(pad_w):int(pad_w) + new_w] = resized

    return canvas, scale, (pad_w, pad_h)


def preprocess(image: np.ndarray) -> tuple:
    """
    完整预处理管线:
      图像 → LetterBox → Normalize → HWC→CHW → Add Batch
    """
    # 1. Letterbox 填充
    padded, scale, (pad_w, pad_h) = letterbox(image, MODEL_INPUT_SIZE)

    # 2. BGR → RGB + Normalize (÷255)
    rgb = padded[..., ::-1].astype(np.float32) / 255.0

    # 3. HWC → CHW (H,W,3) → (3,H,W)
    chw = np.transpose(rgb, (2, 0, 1))

    # 4. Add batch dimension (3,H,W) → (1,3,H,W)
    input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

    return input_tensor, scale, (pad_w, pad_h)


# ═══════════════════════════════════════════════════════════════
#  3. 后处理 — YOLO 输出解码
# ═══════════════════════════════════════════════════════════════

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolov5_output(
    output: np.ndarray,
    num_classes: int = 2,
) -> tuple:
    """
    将 YOLOv5 原始输出解码为边界框 + 置信度 + 类别。

    YOLOv5 输出格式:
        [batch, num_anchors, 5+num_classes]
        其中 num_anchors = 3 scales × 3 anchors × grid_size²
        对于 640 输入: 3×(80²+40²+20²) = 25200

    每个 anchor 的 7 个值:
        [tx, ty, tw, th, obj_conf, cls1_score, cls2_score]

    解码过程:
        bx = sigmoid(tx) + grid_x
        by = sigmoid(ty) + grid_y
        bw = exp(tw) × anchor_w
        bh = exp(th) × anchor_h
    """
    assert output.shape[-1] == 5 + num_classes, \
        f"预期最后一维为 {5 + num_classes}, 实际为 {output.shape[-1]}"

    all_boxes, all_scores, all_class_ids = [], [], []
    grid_idx = 0

    for stride_idx, stride in enumerate(STRIDES):
        grid_size = MODEL_INPUT_SIZE[0] // stride
        num_anchors = 3

        # 当前尺度的输出切片
        n = grid_size * grid_size * num_anchors
        scale_output = output[grid_idx:grid_idx + n]
        grid_idx += n

        # 重塑为 (grid_h, grid_w, num_anchors, 5+num_classes)
        scale_output = scale_output.reshape(grid_size, grid_size, num_anchors, -1)

        # 生成网格坐标
        grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        grid_x = grid_x[..., np.newaxis]
        grid_y = grid_y[..., np.newaxis]

        # 当前尺度的 anchor（归一化到网格尺寸）
        anchors = ANCHORS[stride_idx] / stride

        # 分离各分量
        cx_raw = scale_output[..., 0]
        cy_raw = scale_output[..., 1]
        w_raw = scale_output[..., 2]
        h_raw = scale_output[..., 3]
        obj_conf = scale_output[..., 4]
        cls_scores = scale_output[..., 5:]

        # 解码边界框
        cx = (sigmoid(cx_raw) + grid_x) * stride
        cy = (sigmoid(cy_raw) + grid_y) * stride
        w = np.exp(w_raw) * anchors[:, 0] * stride
        h = np.exp(h_raw) * anchors[:, 1] * stride

        # cx,cy,w,h → x1,y1,x2,y2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # 展平
        boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
        obj_conf_flat = obj_conf.reshape(-1)
        cls_scores_flat = cls_scores.reshape(-1, num_classes)

        # 最终置信度 = obj_conf × max(cls_score)
        max_cls = cls_scores_flat.max(axis=-1)
        final_scores = obj_conf_flat * max_cls
        class_ids = cls_scores_flat.argmax(axis=-1)

        all_boxes.append(boxes)
        all_scores.append(final_scores)
        all_class_ids.append(class_ids)

    # 合并所有尺度
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    return boxes, scores, class_ids


# ═══════════════════════════════════════════════════════════════
#  4. NMS (非极大值抑制)
# ═══════════════════════════════════════════════════════════════

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算一个框与一组框的 IoU"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter

    return np.where(union > 0, inter / union, 0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """
    非极大值抑制 — 去除重复检测框

    流程:
        1. 按置信度降序排列
        2. 取最高分框，加入保留列表
        3. 删除与该框 IoU > 阈值的框
        4. 重复 2-3 直到没有框剩余
    """
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        ious = compute_iou(boxes[i], boxes[order[1:]])
        mask = ious < iou_threshold
        order = order[1:][mask]

    return keep


def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple,
    image_shape: tuple,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 2,
) -> list[dict]:
    """
    完整后处理管线

    参数:
        output:       原始模型输出 [25200, 7]
        scale:        resize 缩放比例
        pad:          (pad_w, pad_h)
        image_shape:  原始图像尺寸 (h, w)
        ...

    返回:
        [{"box": [x1,y1,x2,y2], "conf": float, "class_id": int}, ...]
    """
    # 1. 解码
    boxes, scores, class_ids = decode_yolov5_output(output, num_classes)

    # 2. 置信度过滤
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # 3. 坐标映射回原始图像
    pad_w, pad_h = pad
    orig_h, orig_w = image_shape[:2]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

    # 裁剪到图像边界
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # 4. 按类别分别做 NMS
    final_detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_class_ids = class_ids[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_threshold)
        for idx in keep:
            final_detections.append({
                "box": cls_boxes[idx].tolist(),
                "conf": float(cls_scores[idx]),
                "class_id": int(cls_class_ids[idx]),
            })

    final_detections.sort(key=lambda x: x["conf"], reverse=True)
    return final_detections


# ═══════════════════════════════════════════════════════════════
#  5. 可视化
# ═══════════════════════════════════════════════════════════════

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
]


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    class_names: list[str] = None,
) -> np.ndarray:
    """在图像上绘制检测框和标签"""
    img = image.copy()
    if class_names is None:
        class_names = [f"cls_{i}" for i in range(100)]

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["conf"]
        cls_id = det["class_id"]
        label = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        color = COLORS[cls_id % len(COLORS)]

        # 边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)

        # 标签文字
        cv2.putText(img, text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# ═══════════════════════════════════════════════════════════════
#  6. 推理主函数
# ═══════════════════════════════════════════════════════════════

def inference_single_image(
    session, input_name, output_name,
    image_path: str, output_path: str = None,
) -> list[dict]:
    """对单张图片执行推理"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    orig_image = image.copy()
    input_tensor, scale, pad = preprocess(image)

    # 推理
    t0 = time.perf_counter()
    outputs = session.run([output_name], {input_name: input_tensor})
    infer_time = (time.perf_counter() - t0) * 1000

    # 后处理
    detections = postprocess(
        outputs[0][0], scale, pad, image.shape,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=NMS_THRESHOLD,
        num_classes=len(CLASS_NAMES),
    )

    # 可视化
    result_img = draw_detections(orig_image, detections, CLASS_NAMES)

    # 打印结果
    print(f"[✓] 推理完成 | 推理耗时: {infer_time:.1f}ms")
    print(f"    检测到 {len(detections)} 个物体:")
    for det in detections:
        cls_name = CLASS_NAMES[det["class_id"]]
        print(f"      {cls_name:10s} conf={det['conf']:.3f}  "
              f"box=[{det['box'][0]:.0f},{det['box'][1]:.0f},"
              f"{det['box'][2]:.0f},{det['box'][3]:.0f}]")

    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"    结果已保存: {output_path}")
    else:
        cv2.imshow("ONNX Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def inference_video(session, input_name, output_name,
                    video_path: str = None, output_path: str = None):
    """对视频或摄像头执行推理"""
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    source_name = "Webcam" if video_path is None else Path(video_path).name

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"[*] 开始处理: {source_name} ({w}×{h})")

    frame_count, total_time = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, scale, pad = preprocess(frame)

        t0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: input_tensor})
        total_time += (time.perf_counter() - t0) * 1000

        detections = postprocess(
            outputs[0][0], scale, pad, frame.shape,
            conf_threshold=CONF_THRESHOLD, iou_threshold=NMS_THRESHOLD,
            num_classes=len(CLASS_NAMES),
        )

        result_frame = draw_detections(frame, detections, CLASS_NAMES)
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ONNX Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if writer:
            writer.write(result_frame)
        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[✓] 处理完成: {frame_count} 帧, 平均 {total_time / max(frame_count, 1):.1f}ms/帧")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ONNX YOLOv5 目标检测")
    parser.add_argument("--model", type=str,
                        default=r"H:\pycharm\test_tensorrt\model\best.onnx")
    parser.add_argument("--image", type=str, default=None, help="输入图片路径")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径")
    parser.add_argument("--webcam", type=int, default=None, help="摄像头 ID")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "tensorrt"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    global CONF_THRESHOLD, NMS_THRESHOLD
    CONF_THRESHOLD, NMS_THRESHOLD = args.conf, args.iou

    session, input_name, output_name = load_model(args.model, args.device)

    if args.image:
        inference_single_image(session, input_name, output_name,
                               args.image, args.output)
    elif args.video:
        inference_video(session, input_name, output_name,
                        args.video, args.output)
    elif args.webcam is not None:
        inference_video(session, input_name, output_name,
                        video_path=None, output_path=args.output)
    else:
        print("请指定 --image, --video 或 --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
