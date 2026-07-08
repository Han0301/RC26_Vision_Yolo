import onnxruntime as ort
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class result:
    x1: int
    y1: int
    w: int
    h: int
    cls: int
    conf: np.float16

def resize_padding(image:np.ndarray, target_size: tuple = (640,640), padding_color: np.array = (114,114,114)) -> tuple:
    h,w= image.shape[:2]
    target_h,target_w = target_size
    scale = min(target_h / h, target_w / w)
    now_h, now_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (now_w, now_h), interpolation=cv2.INTER_LINEAR)
    pad_w = int((target_w - w) / 2)
    pad_h = int((target_h - h) / 2)
    now_image = np.full((target_w, target_h, 3), padding_color,dtype=np.uint8)
    now_image[pad_h:pad_h + now_h, pad_w:pad_w + now_w] = resized_image
    return now_image, (pad_w, pad_h), scale
    
    

def preprocess(image : np.ndarray, target_size : tuple = (640,640)) -> tuple:
    # 1 bgr -> rgb
    rgb_image = image[...,::-1]
    # 2 resize_with_padding
    now_image, (pad_w, pad_h), scale = resize_padding(rgb_image, target_size)
    now_image = now_image.astype(np.float32) / 255.0    # now_image.dtype: float32
    now_image = now_image.transpose((2,0,1))
    now_image = now_image.reshape(1,3,640,640)
    return now_image, (pad_w, pad_h), scale

def load_model(model_path : str, device : str = "cpu") -> tuple:
    privider = []
    if device == "tensorrt":
        privider.append("TensorrtExecutionProvider")
    if device == "cuda":
        privider.append("CUDAExecutionProvider")
    if device == "cpu":
        privider.append("CPUExecutionProvider")
    session = ort.InferenceSession(model_path, providers=privider)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name
    
def run(input_tensor : np.ndarray, session: ort.InferenceSession, input_name, output_name):
    output = session.run([output_name],  {input_name: input_tensor})[0]
    return output

def calIOU(out1: result, out2: result):
    x_start, y_start = max(out1.x1, out2.x1),max(out1.y1, out2.y1)
    x_end, y_end = min(out1.x1 + out1.w, out2.x1 + out2.w),  min(out1.y1 + out1.h, out2.y1 + out2.h)
    I = max(0, x_end - x_start) * max(0, y_end - y_start)
    U = out1.w * out1.h + out2.w * out2.h - I
    return I / U
  
def postprocess(output_tensor, pad_w, pad_h, scale, conf_threshold = 0.85, iou_threshold = 0.9 ,target_size = (640,640)):
    c, ancher_num, out_num = output_tensor.shape
    class_num = out_num - 5

    pred_result = output_tensor[0]     # [25200, 7]
    pred_conf = pred_result[:,4]             # [25200]
    pred_result = pred_result[pred_conf > conf_threshold]
    result_list = [[] for _ in range(class_num)]

    for i in range(len(pred_result)):
        x1,y1,w,h,conf = pred_result[i][0],pred_result[i][1],pred_result[i][2], pred_result[i][3], pred_result[i][4]
        cls_conf = pred_result[i][5:]
        cls = cls_conf.argmax()

        x1 = ((x1 - pad_w) / scale).clip(0,target_size[0])
        y1 = ((y1 - pad_h) / scale).clip(0,target_size[1])
        w = w / scale
        h = h / scale
        out = result(x1,y1,w,h,cls,conf)
        result_list[out.cls].append(out)

    for cls in range(len(result_list)):
        # 该类别的所有结果
        cls_result = result_list[cls]
        if len(cls_result) == 0:
            continue
        cls_result.sort(key=lambda x: x.conf, reverse=True)

        # 计算iou, 保留框
        result_save = [True for _ in range(len(cls_result))]
        for result_i in range(cls_result):
            for result_j in range(cls_result):
                if result_save[result_i] == False:
                    continue
                if (calIOU(result_i, result_j) > iou_threshold):




def main():
    image = cv2.imread(r"F:\datasets_blue_kfs\images\1.png")
    session, input_name, output_name = load_model(r"H:\pycharm\test_tensorrt\model\best.onnx", "cpu")

    input_tensor, (pad_w, pad_h), scale = preprocess(image)
    output_tensor = run(input_tensor, session, input_name, output_name)
    postprocess(output_tensor, pad_w, pad_h, scale, 0.88)

if __name__ == "__main__":
    main()