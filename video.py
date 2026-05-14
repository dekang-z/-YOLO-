import cv2  # OpenCV库，用于图像处理和视频流操作
import numpy as np  # 数值计算库
from ultralytics import YOLO  # 引入YOLO目标检测模型
from paddleocr import PaddleOCR  # 引入PaddleOCR文本识别模型
from PIL import Image, ImageDraw, ImageFont  # 用于图像处理和中文文本渲染
import logging  # 日志记录

# 初始化OCR引擎
ocr = PaddleOCR(use_angle_cls=True,  # 启用角度分类器（针对倾斜文本）
                lang="ch")  # 设置识别语言为中文

# 加载YOLO车牌检测模型
model = YOLO(
    r"C:\Users\16430\Desktop\Vehicle_License_Plate_Recognition-main\ultralytics\runs\train\exp3\weights\best.pt",
    task='detect')  # 指定模型路径和任务类型（检测）


# 透视变换函数：将任意四边形区域变换为矩形
def four_point_transform(image, pts):
    """执行四点透视变换"""
    # 将输入点转换为浮点格式，OpenCV要求
    rect = np.array(pts, dtype="float32")

    # 计算变换后矩形的宽度（第一个点和第二个点的欧氏距离）
    width = int(np.linalg.norm(rect[1] - rect[0]))
    # 计算变换后矩形的高度（第一个点和第四个点的欧氏距离）
    height = int(np.linalg.norm(rect[3] - rect[0]))

    # 定义目标矩形四个角的坐标（0宽高-1确保边界包含）
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # 获取透视变换矩阵（原始点 → 目标点）
    M = cv2.getPerspectiveTransform(rect, dst)
    # 应用透视变换，输出矩形图像
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


# 中文字体渲染函数（解决OpenCV不支持中文问题）
def add_chinese_text(img, text, position, textColor=(255, 255, 255), textSize=30):
    """ 在图像上添加中文文字 """
    try:
        # 将OpenCV图像转为PIL格式并转为RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)  # 创建绘图对象

        try:
            # 尝试使用中文字体（宋体）
            fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
        except IOError:
            # 备用字体（英文）
            fontStyle = ImageFont.truetype("arial.ttf", textSize)

        # 在指定位置绘制文本（fill=文本颜色）
        draw.text(position, text, fill=textColor, font=fontStyle)
        # 转回OpenCV格式并返回
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"文字渲染失败: {str(e)}")  # 错误日志记录
        return img  # 出错时返回原始图像


# 视频/摄像头输入设置
video_path = 0  # 0表示默认摄像头，改为文件路径如"test.mp4"可处理视频
cap = cv2.VideoCapture(video_path)  # 创建视频捕获对象

# 摄像头检查
if not cap.isOpened():
    print("无法打开摄像头或视频")
    exit()

# 创建显示窗口并调整大小（提高显示效果）
cv2.namedWindow("车牌识别", cv2.WINDOW_NORMAL)  # 可调整窗口
cv2.resizeWindow("车牌识别", 1280, 720)  # 初始分辨率

# 主循环：处理每一帧视频
while cap.isOpened():
    ret, frame = cap.read()  # 读取下一帧
    if not ret:
        break  # 视频结束或读取失败时退出

    # 提升分辨率（优化检测效果）
    frame = cv2.resize(frame, (1280, 720))

    # YOLO车牌检测
    """关键"""
    results = model(frame,
                    imgsz=640,  # 推理尺寸（平衡速度与精度） imgsz 指定模型输入图像的尺寸（如 640 表示 640×640 像素），需为 32的倍数（因YOLO网络下采样要求）。
                    conf=0.25,  # 置信度阈值（过滤低置信检测） 过滤检测框的置信度阈值（范围 [0, 1]），仅保留高于此值的预测框。
                    iou=0.45)  # IoU阈值（抑制重叠框，防止车牌遮挡重叠或漏检） 在NMS（非极大值抑制）中，合并重叠框的IoU阈值（范围 [0, 1]）。重叠高于此值的框仅保留最高置信度框。

    # 遍历检测到的所有车牌框
    for result in results[0].boxes.xyxy:
        # 解析边界框坐标（左上右下）
        x1, y1, x2, y2 = map(int, result[:4])

        # 增加边界框padding（确保完整包含车牌）
        padding = 10
        x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)  # 左上角扩展
        x2, y2 = min(x2 + padding, frame.shape[1] - 1), min(y2 + padding, frame.shape[0] - 1)  # 右下角扩展

        # 获取车牌区域（实际未使用，下方透视变换直接使用全图）
        cropped_plate = frame[y1:y2, x1:x2]  # 截取区域

        # 构造透视变换四角坐标（基于带padding的边界框）
        pts = np.array([
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x2, y2],  # 右下
            [x1, y2]  # 左下
        ], dtype="float32")

        # 执行透视变换（校正倾斜车牌）
        warped_plate = four_point_transform(frame, pts)

        # OCR识别车牌文字
        result = ocr.ocr(warped_plate, cls=True)  # cls=True启用角度分类

        # 处理OCR结果
        plate_text = ""
        if result and isinstance(result, list):  # 检查有效结果
            try:
                # 拼接所有检测到的文字
                plate_text = "".join([word[1][0] for line in result for word in line if word])
            except Exception as e:
                print(f"车牌识别异常: {e}")
                plate_text = "识别失败"
        else:
            plate_text = "未检测到车牌"

        # 在原图上绘制车牌边界框（绿色矩形）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在车牌上方添加中文识别结果（绿色文字）
        frame = add_chinese_text(frame,
                                 f"车牌: {plate_text}",
                                 (x1, y1 - 30),  # 文本位置（框上方）
                                 textColor=(0, 255, 0),  # 绿色（参数对应rgb）
                                 textSize=30)  # 字号

    # 显示处理后的帧
    cv2.imshow("车牌识别", frame)

    # 退出检测（按'q'键）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()  # 释放视频流
cv2.destroyAllWindows()  # 关闭所有窗口