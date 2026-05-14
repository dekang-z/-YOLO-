import warnings  # 导入警告处理模块
import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from ultralytics import YOLO  # 导入YOLO目标检测模型
from paddleocr import PaddleOCR  # 导入PaddleOCR文字识别引擎
import matplotlib  # Python的2D绘图库
matplotlib.use('TkAgg')  # 设置matplotlib使用TkAgg作为后端，确保在非交互环境中正常显示图形
import matplotlib.pyplot as plt  # 导入Matplotlib绘图库

# ========================
# 中文显示配置
# ========================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体（解决中文乱码）
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号"-"显示为方块的问题

# ========================
# OCR引擎初始化
"""核心代码1，核心组件初始化 20~29行"""
# ========================
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 创建PaddleOCR实例：
# - use_angle_cls=True 启用文字方向分类
# - lang="ch" 指定中文识别模型

# ========================
# 车牌检测模型初始化
# ========================
path =   r".\ultralytics\runs\train\exp3\weights\best.pt"# YOLO模型权重文件路径（训练好的车牌检测模型）
img_path = r".\test.jpg"  # 待识别的测试图片路径
model = YOLO(path, task='detect')  # 加载YOLO模型，指定任务为目标检测

# ========================
# 图像读取与预处理
# ========================
original_image = cv2.imread(img_path)  # 读取原始图像（BGR格式）

# ========================
# 车牌检测
# ========================
results = model(img_path)  # 使用YOLO模型检测图像中的车牌，返回检测结果对象


# ========================
# 透视变换函数（用于车牌矫正）
# ========================
def four_point_transform(image, pts):
    """ 对图像进行透视变换矫正
    Args:
        image: 原始图像
        pts: 车牌四个顶点的坐标（左上、右上、右下、左下顺序）
    Returns:
        warped: 矫正后的车牌图像
    """
    rect = np.array(pts, dtype="float32")  # 将顶点坐标转为NumPy数组

    # 计算车牌宽度和高度（使用欧氏距离）
    width = int(np.linalg.norm(rect[1] - rect[0]))  # 上边宽度
    height = int(np.linalg.norm(rect[3] - rect[0]))  # 左边高度

    # 定义目标矩形顶点（矫正后的标准矩形）
    dst = np.array([
        [0, 0],  # 左上
        [width - 1, 0],  # 右上
        [width - 1, height - 1],  # 右下
        [0, height - 1]  # 左下
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)  # 计算透视变换矩阵
    warped = cv2.warpPerspective(image, M, (width, height))  # 执行透视变换

    return warped


# ========================
# 车牌识别流程
# ========================
for result in results[0].boxes.xyxy:  # 遍历检测到的每个车牌边界框
    # 解析边界框坐标（左上角和右下角）
    x1, y1, x2, y2 = map(int, result[:4])  # 将坐标转为整数

    # 增加边界框padding（扩大检测范围避免截断字符）
    """核心代码 83~104行"""
    padding = 10  # 扩展像素值，定义边界扩展的像素大小，这里设置为10像素。
    x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)  # (x1,y1)是图片左上角坐标，确保移动后的x1不会小于0（即不会超出图像左边界），同样，y1向上移动padding个像素，且不会小于0（即不会超出图像上边界）。
    x2, y2 = min(x2 + padding, original_image.shape[1] - 1), min(y2 + padding, original_image.shape[0] - 1)  #确保移动后的x2不会超过图像的宽度（即图像的最大列索引，因为列索引从0到shape[1]-1）。同样，y2向下移动padding个像素，且不会超过图像的高度（shape[0]-1）。

    # 裁剪车牌区域
    cropped_image = original_image[y1:y2, x1:x2]  # 根据坐标截取车牌区域图像

    # 构建透视变换所需的四个顶点坐标
    pts = np.array([
        [x1, y1],  # 左上
        [x2, y1],  # 右上
        [x2, y2],  # 右下
        [x1, y2]  # 左下
    ], dtype="float32")

    # 执行透视变换（矫正倾斜车牌）
    warped_image = four_point_transform(original_image, pts)  # 返回矫正后的车牌图像

    # ========================
    # OCR文字识别
    # ========================
    result = ocr.ocr(warped_image, cls=True)  # 识别矫正后的车牌：
    # - cls=True 启用方向分类器
    # 拼接识别结果（车牌号码）
    plate_text = "".join([word[1][0] for line in result for word in line])  # 提取每行每个字符的识别结果并拼接

    # ========================
    # 结果可视化
    # ========================
    plt.figure(figsize=(8, 4))  # 创建8x4英寸的画布

    # 显示原始检测车牌
    plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # 将BGR转为RGB格式显示
    plt.title("检测到的车牌")  # 设置子图标题

    # 显示矫正后的车牌和识别结果
    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))  # 显示矫正后的车牌
    plt.title(f"识别结果: {plate_text}")  # 显示识别出的车牌号

    plt.show()  # 显示完整图像