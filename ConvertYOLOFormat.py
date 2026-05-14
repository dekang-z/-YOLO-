import os  # 提供操作系统相关功能，如文件路径操作
import re  # 正则表达式模块，用于从文件名中提取坐标信息
import cv2  # OpenCV库，用于图像处理
import shutil  # 文件操作工具，用于移动文件
import logging  # 日志记录模块，用于记录程序运行信息
from tqdm import tqdm  # 进度条显示库，用于可视化处理进度

# 配置日志：设置日志级别为INFO，格式为时间-级别-消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOFormatConverter:
    def __init__(self, data_path, save_path):
        """
        初始化转换器
        :param data_path: 原始数据集路径（包含图片的根目录）
        :param save_path: 转换后数据保存路径（YOLO格式数据集将保存到这里）
        """
        self.data_path = data_path  # 存储原始数据集路径
        self.save_path = save_path  # 存储转换后数据集保存路径

        # 创建保存路径所需的目录结构
        self._create_directories()

    def _create_directories(self):
        """
        创建YOLO格式所需的目录结构
        在保存路径下创建test/train/val三个子集，每个子集包含images和labels文件夹
        """
        try:
            # 遍历训练集、验证集和测试集
            for subset in ["test", "train", "val"]:
                # 构建图片保存路径：保存路径/子集/images
                images_save_path = os.path.join(self.save_path, subset, "images")
                # 构建标签保存路径：保存路径/子集/labels
                labels_save_path = os.path.join(self.save_path, subset, "labels")
                # 如果图片目录不存在则创建
                if not os.path.exists(images_save_path):
                    os.makedirs(images_save_path)
                # 如果标签目录不存在则创建
                if not os.path.exists(labels_save_path):
                    os.makedirs(labels_save_path)
        except Exception as e:
            # 记录创建目录时的错误
            logging.error(f"创建文件夹时出错: {e}")
            raise  # 抛出异常

    @staticmethod
    def list_path_all_files(dirname):
        """
        递归遍历指定目录下的所有文件
        :param dirname: 要遍历的目录路径
        :return: 包含所有文件完整路径的列表
        """
        result = []  # 存储文件路径的列表
        # os.walk遍历目录树：maindir-当前目录, subdir-子目录列表, file_name_list-文件列表
        for maindir, subdir, file_name_list in os.walk(dirname):
            for filename in file_name_list:
                # 拼接完整文件路径
                apath = os.path.join(maindir, filename)
                result.append(apath)  # 添加到结果列表
        return result

    def convert(self):
        """
        核心转换方法：将原始数据集转换为YOLO格式
        处理流程：
        1. 遍历所有图片文件
        2. 从文件名解析车牌位置信息
        3. 计算YOLO格式的归一化边界框坐标
        4. 保存标签文件(.txt)
        5. 移动图片到新位置并重命名
        """
        try:
            # 获取原始数据集中的所有文件路径
            images_files = self.list_path_all_files(self.data_path)
            # 记录找到的文件数量
            logging.info(f"找到 {len(images_files)} 个文件")

            # 初始化计数器：为每个子集(test/train/val)的图片编号
            cnt = {"test": 1, "train": 1, "val": 1}

            # 使用tqdm包装文件列表，显示进度条
            for name in tqdm(images_files, desc="转换进度"):
                # 只处理jpg和png格式的图片文件
                if name.endswith(".jpg") or name.endswith(".png"):
                    # 从文件路径中提取子集名称（test/train/val）
                    # 例如："./ccpd_green/train/xxx.jpg" -> "train"
                    subset = os.path.basename(os.path.dirname(name))
                    # 使用OpenCV读取图片
                    img = cv2.imread(name)
                    if img is None:
                        # 如果图片读取失败，记录警告
                        logging.warning(f"无法读取图片: {name}")
                        continue  # 跳过此文件

                    # 获取图片的高度和宽度（用于归一化坐标）
                    height, width = img.shape[0], img.shape[1]

                    # 使用正则表达式从文件名中提取坐标信息
                    """关键代码1，文件名解析（正则表达式提取坐标）
                    关键作用：使用正则表达式-\d+\&\d+_\d+\&\d+-匹配文件名中-x0&y0_x1&y1-格式的坐标串，提取车牌边界框的绝对坐标"""
                    try:
                        # 匹配文件名中的坐标模式：-数字&数字_数字&数字-
                        # 例如：文件名中的"-123&456_789&1011-"
                        str1 = re.findall('-\d+\&\d+_\d+\&\d+-', name)[0][1:-1]
                        # 分割字符串：使用&和_作为分隔符
                        str2 = re.split('\&|_', str1)
                        # 解析四个坐标值：左上角(x0,y0)和右下角(x1,y1)
                        x0 = int(str2[0])
                        y0 = int(str2[1])
                        x1 = int(str2[2])
                        y1 = int(str2[3])
                    except Exception as e:
                        # 如果解析失败，记录错误
                        logging.error(f"解析文件名时出错: {name}, 错误: {e}")
                        continue  # 跳过此文件

                    # 计算YOLO格式的归一化边界框坐标：
                    """关键代码2，YOLO格式坐标转换（核心算法）
                    关键作用：
                    执行YOLO格式的核心转换逻辑：
                    中心点坐标：(x0+x1)/2/width
                    宽高相对值：(x1-x0)/width
                    保留6位小数确保精度"""
                    # 中心点x坐标 = (x0+x1)/2 / 图片宽度，保存小数后六位
                    x = round((x0 + x1) / 2 / width, 6)
                    # 中心点y坐标 = (y0+y1)/2 / 图片高度
                    y = round((y0 + y1) / 2 / height, 6)
                    # 边界框宽度 = (x1-x0) / 图片宽度
                    w = round((x1 - x0) / width, 6)
                    # 边界框高度 = (y1-y0) / 图片高度
                    h = round((y1 - y0) / height, 6)

                    # 构建保存路径
                    images_save_path = os.path.join(self.save_path, subset, "images")
                    labels_save_path = os.path.join(self.save_path, subset, "labels")

                    # 构建标签文件名：green_plate_000001.txt（使用6位数字补零）
                    txtfile = os.path.join(labels_save_path, f"green_plate_{str(cnt[subset]).zfill(6)}.txt")
                    # 构建图片文件名：保留原始图片格式
                    imgfile = os.path.join(
                        images_save_path,
                        f"green_plate_{str(cnt[subset]).zfill(6)}.{os.path.basename(name).split('.')[-1]}"
                    )

                    # 写入标签文件（YOLO格式）
                    """关键代码3. YOLO标签生成
                    关键作用：
                    生成YOLO标准标注文件：
                    0表示车牌类别ID
                    x y w h为归一化相对值"""
                    with open(txtfile, "w") as f:
                        # 格式：类别ID 中心x 中心y 宽度 高度
                        # 这里假设所有车牌都是同一类别（0）
                        f.write(" ".join(["0", str(x), str(y), str(w), str(h)]))

                    # 移动图片到新位置（同时重命名）
                    shutil.move(name, imgfile)

                    # 更新当前子集的计数器
                    cnt[subset] += 1

            # 转换完成，记录处理的总图片数（减去初始值3，因为一开始设置的初始值3项都是1）
            logging.info(f"转换完成，共处理 {sum(cnt.values()) - 3} 张图片")
        except Exception as e:
            # 转换过程中出现异常，记录错误
            logging.error(f"转换过程中出错: {e}")
            raise  # 抛出异常

if __name__ == '__main__':
    # 原始数据集路径（包含子文件夹test/train/val）
    data_path = "./CCPD2020/ccpd_green"
    # 转换后数据保存路径（将创建YOLO格式的数据集）
    save_path = "./dataset"

    # 创建格式转换器实例
    converter = YOLOFormatConverter(data_path, save_path)
    # 执行转换操作
    converter.convert()