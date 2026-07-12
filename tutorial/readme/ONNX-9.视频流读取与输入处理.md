# 视频流读取与输入处理

> 使用 OpenCV 读取摄像头、视频文件、RTSP/HTTP 网络流，以及高效处理视频帧的完整方案。

---

## 目录

1. [视频读取基础概念](#1-视频读取基础概念)
2. [OpenCV VideoCapture API](#2-opencv-videocapture-api)
3. [读取摄像头输入](#3-读取摄像头输入)
4. [读取视频文件](#4-读取视频文件)
5. [读取网络视频流（RTSP / HTTP）](#5-读取网络视频流rtsp--http)
6. [视频属性获取与设置](#6-视频属性获取与设置)
7. [高效帧读取循环](#7-高效帧读取循环)
8. [多线程/多进程加速](#8-多线程多进程加速)
9. [实用示例](#9-实用示例)
10. [常见问题](#10-常见问题)

---

## 1. 视频读取基础概念

```mermaid
flowchart LR
    subgraph 输入源
        A["摄像头 (Camera)"]
        B["视频文件 (*.mp4, *.avi)"]
        C["RTSP 网络流"]
        D["HTTP 视频流"]
    end

    subgraph OpenCV
        E["cv2.VideoCapture"]
        F["逐帧读取 .read()"]
        G["帧解码"]
    end

    subgraph 处理
        H["预处理 / 推理 / 显示"]
        I["帧率控制 / 丢帧策略"]
    end

    A & B & C & D --> E --> F --> G --> H --> I
```

**关键概念：**

| 概念 | 说明 |
|------|------|
| **帧 (Frame)** | 视频的每一张静态图像，通常为 `numpy.ndarray`，shape `(H, W, 3)` BGR 格式 |
| **FPS (Frames Per Second)** | 视频的帧率，决定每秒处理的帧数 |
| **编解码器 (Codec)** | 视频压缩/解压缩算法，如 H.264、H.265 |
| **延迟 (Latency)** | 从采集到处理完成的时间，实时应用的关键指标 |
| **丢帧 (Frame Drop)** | 当处理速度慢于采集速度时，选择性跳过部分帧 |

---

## 2. OpenCV VideoCapture API

OpenCV 提供统一的 `cv2.VideoCapture` 接口，支持多种输入源。

```python
import cv2

# 创建 VideoCapture 对象
cap = cv2.VideoCapture()

# 通过 open() 连接指定设备/文件
cap.open(0)                        # 摄像头 ID 0
# cap.open("video.mp4")            # 视频文件
# cap.open("rtsp://...")           # RTSP 流
# cap.open("http://...")           # HTTP 流

# 或者在构造时直接指定
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video.mp4")
```

### 2.1 检查是否成功打开

```python
if not cap.isOpened():
    print("无法打开视频源")
    exit()
```

### 2.2 释放资源

```python
cap.release()    # 关闭视频源
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
```

---

## 3. 读取摄像头输入

### 3.1 打开默认摄像头

```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = 第一个摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break

    # 显示帧
    cv2.imshow("Camera", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3.2 设置摄像头参数

```python
cap = cv2.VideoCapture(0)

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 设置 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# 设置编码格式（部分摄像头支持）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 读取当前值
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print(f"分辨率: {width} x {height}, FPS: {fps}")
```

> **注意：** `cap.set()` 不一定对所有摄像头生效，取决于摄像头驱动和硬件支持。

### 3.3 多个摄像头

```python
# 多个摄像头可以分别打开
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)  # 第二个摄像头（USB 摄像头等）
```

---

## 4. 读取视频文件

```python
import cv2

cap = cv2.VideoCapture("video.mp4")  # 支持 .mp4, .avi, .mov, .mkv 等

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取总帧数用于进度显示
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频播放完毕

    # 处理帧 ...

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4.1 跳转到指定帧

```python
# 跳转到第 100 帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

# 跳转到指定时间位置（毫秒）
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # 第 5 秒
```

---

## 5. 读取网络视频流（RTSP / HTTP）

### 5.1 RTSP 流（摄像头等）

```python
# RTSP 地址格式
# rtsp://[username:password@]host[:port]/path
rtsp_url = "rtsp://admin:123456@192.168.1.100:554/stream1"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法连接 RTSP 流")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("流断开，尝试重连...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)  # 重连
        continue

    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 5.2 HTTP 视频流（MJPEG / HLS）

```python
# MJPEG over HTTP（如网络摄像头网页界面）
http_url = "http://192.168.1.100:8080/video"

cap = cv2.VideoCapture(http_url)

# HLS 流（如直播）
# hls_url = "https://example.com/live/stream.m3u8"
# cap = cv2.VideoCapture(hls_url)
```

### 5.3 YouTube / 在线视频

需要先使用 `yt-dlp` 获取真实流地址：

```python
# 1. 安装 yt-dlp: pip install yt-dlp

import yt_dlp
import cv2

def get_video_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# 获取真实流地址
url = get_video_url("https://www.youtube.com/watch?v=...")

# 用 OpenCV 读取
cap = cv2.VideoCapture(url)
```

---

## 6. 视频属性获取与设置

### 6.1 常用属性

```python
# 获取属性
width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))        # 宽度
height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))       # 高度
fps       = cap.get(cv2.CAP_PROP_FPS)                     # 帧率
total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))        # 总帧数
codec     = int(cap.get(cv2.CAP_PROP_FOURCC))             # 编码格式
exposure  = cap.get(cv2.CAP_PROP_EXPOSURE)                # 曝光度
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)             # 亮度
contrast  = cap.get(cv2.CAP_PROP_CONTRAST)                # 对比度

# 将 FOURCC 转为可读字符串
codec_str = chr(codec & 0xFF) + chr((codec >> 8) & 0xFF) + \
            chr((codec >> 16) & 0xFF) + chr((codec >> 24) & 0xFF)
```

### 6.2 属性设置对照表

| 属性 | 含义 | 常见值 |
|------|------|--------|
| `CAP_PROP_FRAME_WIDTH` | 帧宽度 | 640, 1280, 1920 |
| `CAP_PROP_FRAME_HEIGHT` | 帧高度 | 480, 720, 1080 |
| `CAP_PROP_FPS` | 帧率 | 15, 30, 60 |
| `CAP_PROP_FOURCC` | 编码格式 | `'MJPG'`, `'YUYV'` |
| `CAP_PROP_BUFFERSIZE` | OpenCV 内部缓冲区大小 | — |
| `CAP_PROP_POS_MSEC` | 当前时间位置（ms） | 用于跳转 |
| `CAP_PROP_POS_FRAMES` | 当前帧号 | 用于跳转 |

---

## 7. 高效帧读取循环

### 7.1 基本循环

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 处理帧
```

### 7.2 控制帧率（固定 FPS）

```python
import time

target_fps = 30
frame_time = 1.0 / target_fps

while True:
    t0 = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # 处理帧 ...

    # 维持固定帧率
    elapsed = time.perf_counter() - t0
    sleep_time = frame_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

    # 显示
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 7.3 实时 FPS 显示

```python
import time
import cv2

fps_list = []
while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    # 处理帧 ...

    t1 = time.perf_counter()
    fps_list.append(t1 - t0)

    # 滑动窗口均值（最近 30 帧）
    if len(fps_list) > 30:
        fps_list.pop(0)
    current_fps = 1.0 / (sum(fps_list) / len(fps_list))

    # 在画面上叠加显示 FPS
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 7.4 丢帧策略（读最新帧）

当处理速度跟不上采集速度时，可丢弃中间帧，只处理最新帧：

```python
import cv2

# 减小 OpenCV 缓冲区，确保读到最新帧
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 或手动清空缓冲区
def read_latest_frame(cap):
    """丢弃缓冲区中的旧帧，只返回最新一帧。"""
    ret, frame = cap.read()
    if not ret:
        return False, None

    # 持续读取直到缓冲区清空
    max_drain = 10  # 防止死循环
    while max_drain > 0:
        ret_next, frame_next = cap.read()
        if not ret_next:
            break
        frame = frame_next  # 更新为最新帧
        max_drain -= 1

    return True, frame

while True:
    ret, frame = read_latest_frame(cap)
    if not ret:
        break
    # 处理帧 ...
```

---

## 8. 多线程/多进程加速

### 8.1 生产者-消费者模式（独立读帧线程）

将 I/O 密集的帧读取放在独立线程中，避免阻塞主处理逻辑：

```python
import cv2
import threading
import queue
from typing import Optional

class VideoStream:
    """在后台线程中持续读取帧的生产者。"""
    def __init__(self, src, queue_size=64):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            if not self.q.full():
                self.q.put(frame)

    def read(self) -> Optional[np.ndarray]:
        """获取最新一帧（非阻塞）。"""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.cap.release()

# 使用示例
vs = VideoStream(0)  # 或视频文件 / RTSP 流

while True:
    frame = vs.read()
    if frame is None:
        continue

    # 主线程专注于处理
    result = heavy_processing(frame)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
```

### 8.2 多进程方案（适合 GIL 密集型任务）

```python
import cv2
import multiprocessing as mp

def frame_reader(src, frame_queue: mp.Queue, stop_event):
    """独立进程读取帧。"""
    cap = cv2.VideoCapture(src)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def frame_processor(frame_queue: mp.Queue, stop_event):
    """独立进程处理帧。"""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except:
            continue
        # 处理帧 ...

if __name__ == '__main__':
    queue = mp.Queue(maxsize=128)
    stop = mp.Event()

    reader = mp.Process(target=frame_reader, args=(0, queue, stop))
    processor = mp.Process(target=frame_processor, args=(queue, stop))

    reader.start()
    processor.start()

    reader.join()
    processor.join()
```

---

## 9. 实用示例

### 9.1 从摄像头读取并实时推理

```python
import cv2
import numpy as np
import time
import onnxruntime as ort

# 初始化 ONNX Runtime
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps_history = []
while True:
    t0 = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    # 预处理
    blob = cv2.resize(frame, (640, 640))
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]  # (1, 3, 640, 640)

    # 推理
    outputs = session.run(None, {input_name: blob})

    # 后处理 & 绘制 ...

    # FPS 统计
    elapsed = time.perf_counter() - t0
    fps_history.append(elapsed)
    if len(fps_history) > 30:
        fps_history.pop(0)
    fps = 1.0 / (sum(fps_history) / len(fps_history))

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 9.2 读取 RTSP 并写入本地文件

```python
import cv2

rtsp_url = "rtsp://admin:123456@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

# 获取原始属性
fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1', 'XVID'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)  # 写入本地文件

    cv2.imshow("Recording...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

### 9.3 视频文件批处理（逐帧分析）

```python
import cv2
import os

def process_video(video_path, output_dir):
    """逐帧处理视频并保存结果。"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 分析处理 ...
        result_frame = frame  # 替换为处理后的帧

        # 保存
        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, result_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"进度: {frame_idx}/{total}")

    cap.release()
    print("处理完成")
```

---

## 10. 常见问题

### 10.1 无法打开摄像头

```python
# 尝试不同的后端（MSMF / DSHOW / VFW）
# Windows 上可指定后端
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # DirectShow
# cap = cv2.VideoCapture(0, cv2.CAP_MSMF)     # Microsoft Media Foundation
```

### 10.2 RTSP 延迟过高

```python
# 降低缓冲区大小以减少延迟
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 使用 TCP 传输（部分摄像头 RTSP 支持）
# rtsp://admin:123456@192.168.1.100:554/stream1?tcp
```

### 10.3 视频读取失败/重连机制

```python
import time
import cv2

def safe_video_capture(url, retry_interval=3):
    """带重连机制的 VideoCapture。"""
    while True:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        print(f"连接失败，{retry_interval}s 后重试...")
        time.sleep(retry_interval)
```

### 10.4 cv2.waitKey 说明

```python
# waitKey(1) 等待 1ms，返回按键的 ASCII 码
# 常用写法：
if cv2.waitKey(1) & 0xFF == ord('q'):   # 按 q 退出
    break
if cv2.waitKey(1) & 0xFF == ord('s'):   # 按 s 保存当前帧
    cv2.imwrite("snapshot.jpg", frame)
```

### 10.5 支持的视频格式

| 格式 | 说明 |
|------|------|
| `.mp4` | H.264/H.265 编码，通用格式 |
| `.avi` | 原始/未压缩格式，文件大 |
| `.mov` | Apple 生态 |
| `.mkv` | 开源容器，支持多流 |
| `.flv` | Flash 视频，直播常用 |
| RTSP | 实时流协议，安防摄像头 |
| RTMP | 实时消息协议，直播推流 |
| HLS | HTTP Live Streaming，Apple |
| MJPEG | 连续 JPEG 图片流，浏览器友好 |

---

> **总结：** OpenCV 的 `VideoCapture` 提供了统一的视频读取接口，覆盖摄像头、本地文件、网络流等场景。结合多线程读取和丢帧策略，可以构建高效的实时视频处理管线。
