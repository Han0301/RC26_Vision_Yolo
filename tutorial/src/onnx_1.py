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
    pad_w = int((target_w - now_w) / 2)
    pad_h = int((target_h - now_h) / 2)
    now_image = np.full((target_h, target_w, 3), padding_color,dtype=np.uint8)
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

# cls_result 排序完的当前列表 result
def nms(cls_result:list, iou_threshold = 0.9) -> list:
    keep_lists = []
    for i, box_i in enumerate(cls_result):
        if box_i is None:
            continue
        keep_lists.append(i)

        for j in range(i + 1, len(cls_result)):
            if cls_result[j] is None:
                continue
            if calIOU(box_i, cls_result[j]) > iou_threshold:
                cls_result[j] = None
    return [cls_result[t] for t in keep_lists]

def postprocess(output_tensor, pad_w, pad_h, scale, conf_threshold = 0.85, iou_threshold = 0.9 ,target_size = (640,640)):
    _, _, out_num = output_tensor.shape
    class_num = out_num - 5

    pred_result = output_tensor[0]     # [25200, 7]
    pred_conf = pred_result[:,4]             # [25200]
    pred_result = pred_result[pred_conf > conf_threshold]
    result_list = [[] for _ in range(class_num)]

    for i in range(len(pred_result)):
        xc,yc,w,h,conf = pred_result[i][0],pred_result[i][1],pred_result[i][2], pred_result[i][3], pred_result[i][4]
        cls_conf = pred_result[i][5:]
        cls = cls_conf.argmax()

        x1 = ((xc - pad_w - w/2) / scale).clip(0,target_size[0])
        y1 = ((yc - pad_h - h/2) / scale).clip(0,target_size[1])
        w = int(w / scale)
        h = int(h / scale)
        out = result(int(x1),int(y1),w,h,cls,conf)
        result_list[out.cls].append(out)

    for cls in range(len(result_list)):
        # 该类别的所有结果
        cls_result = result_list[cls]
        if len(cls_result) <= 1:
            continue
        # 当前类别的所有框
        cls_result.sort(key=lambda x: x.conf, reverse=True)

        # nms
        cls_result = nms(cls_result, iou_threshold)
    return result_list

def drew_result(image: np.ndarray, result_list: list, cls_to_name: dict = {}):
    drew_image = image
    list = []
    for cls_idx in range(len(result_list)):
        if result_list[cls_idx] is [] or None:
            continue
        for result_idx in range(len(result_list[cls_idx])):
            if result_list[cls_idx][result_idx] is not None:
                list.append(result_list[cls_idx][result_idx])
    for result in list:
        print(result)
        cv2.rectangle(drew_image, 
                      (result.x1, result.y1), 
                      (result.x1 + result.w, result.y1 + result.h),
                      (255,0,0),
                      2)
        text = "cls: " + str(result.cls) +" : conf: " + f"{result.conf:.3f}"
        cv2.putText(drew_image, text, (result.x1 - 5, result.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6,
                    (255, 255, 255),  # 白色文字
                    1,
                    lineType=cv2.LINE_AA)
    cv2.imshow("result", drew_image)
    cv2.waitKey(100000000)

def process_frame(iamge: np.ndarray, session, input_name, output_name) -> list:
    try:
        input_tensor, (pad_w, pad_h), scale = preprocess(iamge)
        output_tensor = run(input_tensor, session, input_name, output_name)
        return postprocess(output_tensor, pad_w, pad_h, scale, 0.88)
    except Exception as e:
        print(f"process_frame error: {e}")
        return []

def process_video(open_source: any,  is_drew:bool = True):
    try:
        cap =  cv2.VideoCapture(open_source)
        if not cap.isOpened():
            raise RuntimeError("process_video: not cap.isOpened()")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"process_video: cap info: w: {width}, h: {height}, fps: {fps}")

        while True:
            is_read_ok, frame = cap.read()
            if not is_read_ok:
                print("process_video: not is_read_ok")
                continue
            result_list = process_frame(frame, True)
            if is_drew:
                drew_result(frame, result_list)

    except Exception as e:
        print(f"process_video error:  {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows() 

def main():
    image = cv2.imread(r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179\images\image_3.png")
    model_path = r"H:\pycharm\test_tensorrt\model\best.onnx"
    session, input_name, output_name = load_model(model_path, "cpu")
    
    result_list = process_frame(image, session, input_name, output_name)
    drew_result(image, result_list)

if __name__ == "__main__":
    main()