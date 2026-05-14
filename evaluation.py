import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import time
import yaml
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class LicensePlateEvaluator:
    def __init__(self, config_path, test_image_dir, model_path=None, ocr_model=None):
        """
        初始化车牌识别评估器

        参数:
        config_path: 数据集配置文件路径 (data.yaml)
        test_image_dir: 测试集图片目录
        model_path: 指定的YOLO模型路径 (可选)
        ocr_model: 可选的预加载OCR模型
        """
        # 存储路径
        self.config_path = config_path
        self.test_image_dir = test_image_dir
        self.model_path = model_path  # 存储用户指定的模型路径

        # 设置标签目录（从images替换为labels）
        self.test_label_dir = test_image_dir.replace("images", "labels")

        # 创建输出目录
        self.output_dir = "evaluation_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'evaluation.log')),
                logging.StreamHandler()
            ]
        )

        # 记录路径信息
        logging.info(f"配置文件路径: {os.path.abspath(config_path)}")
        logging.info(f"测试图片目录: {os.path.abspath(test_image_dir)}")
        logging.info(f"测试标签目录: {os.path.abspath(self.test_label_dir)}")

        # 检查路径是否存在
        if not os.path.exists(config_path):
            logging.error(f"配置文件不存在: {config_path}")
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        if not os.path.exists(test_image_dir):
            logging.error(f"测试图片目录不存在: {test_image_dir}")
            raise FileNotFoundError(f"测试图片目录不存在: {test_image_dir}")

        if not os.path.exists(self.test_label_dir):
            logging.error(f"测试标签目录不存在: {self.test_label_dir}")
            raise FileNotFoundError(f"测试标签目录不存在: {self.test_label_dir}")

        # 加载数据集配置
        try:
            self.config = self.load_config_safely(config_path)
            logging.info("成功加载数据集配置")
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            raise

        # 初始化模型
        self.model = self._load_detection_model()
        self.ocr = ocr_model or PaddleOCR(use_angle_cls=True, lang="ch")

        # 存储评估结果
        self.results = []
        self.evaluation_metrics = {}
        self.error_analysis = {}

    @staticmethod
    def load_config_safely(config_path):
        """安全加载YAML配置文件，自动检测编码"""
        encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16']

        for encoding in encodings:
            try:
                with open(config_path, 'r', encoding=encoding) as f:
                    return yaml.safe_load(f)
            except UnicodeDecodeError:
                continue

        # 如果所有编码都失败，使用chardet检测
        try:
            import chardet
            with open(config_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(config_path, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"无法确定文件编码: {config_path}, 错误: {e}")
            raise

    def _load_detection_model(self):
        """加载训练好的YOLO检测模型"""
        try:
            # 优先使用用户指定的模型路径
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
                logging.info(f"加载用户指定的检测模型: {os.path.abspath(model_path)}")
                return YOLO(model_path, task='detect')

            # 如果未指定模型路径，则查找最新训练模型
            train_dir = "runs/train"
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"训练目录不存在: {os.path.abspath(train_dir)}")

            train_dirs = sorted([
                d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d)) and os.path.exists(
                    os.path.join(train_dir, d, "weights", "best.pt"))
            ], reverse=True)

            if not train_dirs:
                raise FileNotFoundError("未找到训练模型，请先训练模型")

            # 使用最新的训练模型
            latest_train = train_dirs[0]
            model_path = os.path.join(train_dir, latest_train, "weights", "best.pt")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            logging.info(f"加载检测模型: {os.path.abspath(model_path)}")
            return YOLO(model_path, task='detect')
        except Exception as e:
            logging.error(f"加载检测模型失败: {e}")
            raise

    def load_test_data(self):
        """加载测试数据集"""
        image_files = []
        label_files = []

        # 检查目录是否存在
        if not os.path.exists(self.test_image_dir):
            logging.error(f"测试图片目录不存在: {self.test_image_dir}")
            return []

        if not os.path.exists(self.test_label_dir):
            logging.error(f"测试标签目录不存在: {self.test_label_dir}")
            return []

        # 只处理有对应标注文件的图像
        for filename in os.listdir(self.test_image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.test_image_dir, filename)
                base_name = os.path.splitext(filename)[0]
                label_path = os.path.join(self.test_label_dir, f"{base_name}.txt")

                if os.path.exists(label_path):
                    image_files.append(img_path)
                    label_files.append(label_path)
                else:
                    logging.warning(f"未找到标签文件: {label_path}，跳过图片: {img_path}")

        logging.info(f"找到 {len(image_files)} 个带标注的测试图像")
        return list(zip(image_files, label_files))

    def parse_yolo_label(self, label_path, img_width, img_height):
        """解析YOLO格式的标注文件"""
        boxes = []

        if not os.path.exists(label_path):
            logging.warning(f"标签文件不存在: {label_path}")
            return boxes

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # 解析YOLO格式的归一化坐标
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                # 转换为边界框坐标
                x_min = max(0, int(x_center - width / 2))
                y_min = max(0, int(y_center - height / 2))
                x_max = min(img_width - 1, int(x_center + width / 2))
                y_max = min(img_height - 1, int(y_center + height / 2))

                # 确保边界框有效
                if x_min < x_max and y_min < y_max:
                    boxes.append((x_min, y_min, x_max, y_max))
                else:
                    logging.warning(f"无效边界框: {x_min},{y_min},{x_max},{y_max} in {label_path}")

            return boxes
        except Exception as e:
            logging.error(f"解析标签文件失败: {label_path}, 错误: {e}")
            return []

    def run_inference(self, visualize_samples=10):
        """
        在测试集上运行推理并收集结果
        """
        test_data = self.load_test_data()
        self.results = []
        visualization_count = 0

        if not test_data:
            logging.error("没有可用的测试数据")
            return []

        for img_path, label_path in tqdm(test_data, desc="运行推理", unit="img"):
            try:
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"无法读取图像: {img_path}")
                    continue

                # 获取图像尺寸
                height, width = img.shape[:2]

                # 解析真实标注
                true_boxes = self.parse_yolo_label(label_path, width, height)

                # 运行YOLO检测
                start_time = time.time()
                det_results = self.model(img)  # 返回Results对象列表
                detection_time = (time.time() - start_time) * 1000  # 转换为毫秒

                # 处理检测结果
                detected_boxes = []

                # 遍历检测结果 (每张图一个Results对象)
                for result in det_results:
                    # 获取检测框
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()

                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box[:4])
                            detected_boxes.append((x1, y1, x2, y2))

                            # 裁剪车牌区域
                            plate_img = img[y1:y2, x1:x2]

                            if plate_img.size == 0:
                                logging.warning(f"裁剪的车牌区域为空: {img_path}")
                                plate_text = ""
                            else:
                                # 运行OCR识别
                                try:
                                    ocr_result = self.ocr.ocr(plate_img, cls=True)
                                    plate_text = ""
                                    if ocr_result is not None and len(ocr_result) > 0:
                                        # 安全处理OCR结果
                                        text_lines = []
                                        for line in ocr_result:
                                            if line and len(line) > 0:
                                                for word_info in line:
                                                    if len(word_info) > 1 and word_info[1] and len(word_info[1]) > 0:
                                                        text_lines.append(str(word_info[1][0]))
                                        plate_text = "".join(text_lines)
                                except Exception as e:
                                    logging.error(f"OCR处理失败: {img_path}, 错误: {e}")
                                    plate_text = ""

                            # 检查检测是否有效
                            is_correct_detection = False
                            for true_box in true_boxes:
                                iou_value = self.iou((x1, y1, x2, y2), true_box)
                                if iou_value > 0.5:
                                    is_correct_detection = True
                                    break

                            # 保存结果
                            self.results.append({
                                "image_path": img_path,
                                "detection_box": [x1, y1, x2, y2],
                                "plate_text": plate_text,
                                "detection_time": detection_time,
                                "is_correct_detection": is_correct_detection
                            })

                # 可视化部分样本
                if visualization_count < visualize_samples and detected_boxes:
                    self.visualize_detection(
                        img.copy(),
                        detected_boxes,
                        true_boxes,
                        os.path.basename(img_path)
                    )
                    visualization_count += 1

            except Exception as e:
                logging.error(f"处理图像时出错: {img_path}, 错误: {e}")

        # 保存结果到文件
        with open(os.path.join(self.output_dir, 'inference_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logging.info(f"推理完成，处理了 {len(self.results)} 个车牌区域")
        return self.results

    @staticmethod
    def iou(box1, box2):
        """计算两个边界框的交并比(IoU)"""
        # 解包坐标
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 计算相交区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 计算相交区域面积
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # 计算各自区域面积
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # 计算并集面积
        union_area = box1_area + box2_area - inter_area

        # 计算IoU
        return inter_area / union_area if union_area > 0 else 0

    def visualize_detection(self, img, detected_boxes, true_boxes, filename):
        """可视化检测结果，包含预测框和真实框"""
        try:
            # 转换为PIL图像用于绘制中文
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 绘制真实框（绿色）
            for box in true_boxes:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x1, y1 - 10), "真实框", fill="green", font=ImageFont.load_default())

            # 绘制检测框（红色）
            for box in detected_boxes:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # 计算与真实框的最大IoU
                max_iou = max([self.iou(box, true_box) for true_box in true_boxes]) if true_boxes else 0
                iou_text = f"交并比: {max_iou:.2f}"

                # 在框上方显示IoU
                draw.text((x1, y1 - 30), iou_text, fill="blue", font=ImageFont.load_default())

            # 保存可视化结果
            output_path = os.path.join(self.output_dir, f"detection_{filename}")
            img_pil.save(output_path)
            logging.info(f"保存可视化结果: {output_path}")

        except Exception as e:
            logging.error(f"可视化失败: {e}")

    def calculate_metrics(self):
        """计算评估指标"""
        if not self.results:
            self.run_inference()
        # 1. 检测指标
        """关键代码400~405行"""
        detection_times = [r['detection_time'] for r in self.results]
        avg_detection_time = np.mean(detection_times) if detection_times else 0
        fps = 1000 / avg_detection_time if avg_detection_time > 0 else 0 # FPS计算
        # 检测准确率
        correct_detections = sum(1 for r in self.results if r['is_correct_detection'])
        detection_accuracy = correct_detections / len(self.results) if self.results else 0

        # 保存指标
        self.evaluation_metrics = {
            "detection_accuracy": detection_accuracy,
            "avg_detection_time": avg_detection_time,
            "fps": fps,
            "detection_times": detection_times
        }

        # 保存指标到文件
        with open(os.path.join(self.output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=2, ensure_ascii=False)

        return self.evaluation_metrics

    def plot_detection_performance(self):
        """绘制检测性能指标"""
        if not self.evaluation_metrics:
            self.calculate_metrics()

        # 提取YOLO指标
        yolo_metrics = self.evaluation_metrics.get("yolo_metrics", {})

        # 创建图表
        plt.figure(figsize=(15, 10))

        # 3. 检测时间分布
        plt.subplot(2, 2, 3)
        if self.evaluation_metrics["detection_times"]:
            plt.hist(self.evaluation_metrics["detection_times"], bins=20, color='salmon', alpha=0.7)
            plt.title('检测时间分布')
            plt.xlabel('检测时间 (ms)')
            plt.ylabel('频率')
        else:
            plt.text(0.5, 0.5, '无检测时间数据', ha='center', va='center')
            plt.title('检测时间分布')

        # 4. F1-置信度曲线
        f1_path = os.path.join("runs/detect/val", 'F1_curve.png')
        if os.path.exists(f1_path):
            plt.subplot(2, 2, 4)
            f1_img = plt.imread(f1_path)
            plt.imshow(f1_img)
            plt.axis('off')
            plt.title('F1-置信度曲线')
        else:
            logging.warning(f"未找到F1曲线: {f1_path}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detection_performance.png'))
        plt.close()
        logging.info("检测性能图表已保存")

    def plot_confusion_matrix(self):
        """绘制混淆矩阵（来自YOLO评估）"""
        conf_matrix_path = os.path.join("runs/detect/val", 'confusion_matrix.png')
        if os.path.exists(conf_matrix_path):
            try:
                conf_matrix_img = plt.imread(conf_matrix_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(conf_matrix_img)
                plt.axis('off')
                plt.title('混淆矩阵')
                plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
                plt.close()
                logging.info("混淆矩阵图表已保存")
            except Exception as e:
                logging.error(f"加载混淆矩阵失败: {e}")
        else:
            logging.warning(f"未找到混淆矩阵: {conf_matrix_path}")

    def plot_error_analysis(self):
        """绘制错误分析图表"""
        if not self.results:
            self.run_inference()

        # 计算错误类型分布
        error_types = {
            "检测失败": 0,
            "检测正确": 0,
            "OCR失败": 0
        }

        for r in self.results:
            if not r['is_correct_detection']:
                error_types["检测失败"] += 1
            else:
                error_types["检测正确"] += 1
                if not r['plate_text']:
                    error_types["OCR失败"] += 1

        # 转换为百分比
        total = len(self.results) if self.results else 1
        for key in error_types:
            error_types[key] = error_types[key] / total

        # 保存错误分析结果
        self.error_analysis = error_types

        # 绘制错误类型分布
        plt.figure(figsize=(10, 6))
        bars = plt.bar(error_types.keys(), error_types.values(), color=['red', 'green', 'orange'])

        # 在柱子上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1%}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.title('错误类型分布')
        plt.ylabel('比例')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'error_analysis.png'))
        plt.close()
        logging.info("错误分析图表已保存")

    def plot_speed_analysis(self):
        """绘制速度性能分析"""
        if not self.evaluation_metrics:
            self.calculate_metrics()

        # 性能指标
        metrics = {
            '平均检测时间(ms)': self.evaluation_metrics.get("avg_detection_time", 0),
            'FPS': self.evaluation_metrics.get("fps", 0)
        }

        # 创建图表
        plt.figure(figsize=(12, 5))

        # 柱状图 - 平均检测时间
        plt.subplot(1, 2, 1)
        bars = plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen'])
        plt.title('模型性能指标')
        plt.ylabel('数值')
        plt.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')

        # 箱线图 - 检测时间分布
        plt.subplot(1, 2, 2)
        if self.evaluation_metrics.get("detection_times"):
            plt.boxplot(self.evaluation_metrics["detection_times"])
            plt.title('检测时间分布')
            plt.ylabel('时间(ms)')
            plt.xticks([1], ['检测时间'])
        else:
            plt.text(0.5, 0.5, '无检测时间数据', ha='center', va='center')
            plt.title('检测时间分布')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'speed_analysis.png'))
        plt.close()
        logging.info("速度分析图表已保存")

    def generate_full_report(self):
        """生成完整评估报告"""
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 50)
        print("车牌识别模型综合评估报告")
        print("=" * 50)

        # 计算指标
        if not self.evaluation_metrics:
            self.calculate_metrics()

        # 显示关键指标
        yolo_metrics = self.evaluation_metrics.get("yolo_metrics", {})
        print(f"\n关键性能指标:")
        print(f"  检测准确率: {self.evaluation_metrics.get('detection_accuracy', 0):.2%}")
        print(f"  平均检测时间: {self.evaluation_metrics.get('avg_detection_time', 0):.2f} ms")
        print(f"  处理速度: {self.evaluation_metrics.get('fps', 0):.2f} FPS")

        # 绘制所有图表
        print("\n生成可视化图表...")
        self.plot_detection_performance()
        self.plot_confusion_matrix()
        self.plot_error_analysis()
        self.plot_speed_analysis()



        print(f"\n评估报告已生成，结果保存在: {os.path.abspath(self.output_dir)}")
        print("=" * 50)



if __name__ == "__main__":
    # 配置评估参数
    BASE_DIR = r"C:\Users\16430\Desktop\Vehicle_License_Plate_Recognition-main"
    CONFIG_PATH = os.path.join(BASE_DIR, "ultralytics", "data.yaml")
    TEST_IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")

    # 用户指定的模型路径
    MODEL_PATH = r"C:\Users\16430\Desktop\Vehicle_License_Plate_Recognition-main\ultralytics\runs\train\exp3\weights\best.pt"

    # 打印路径信息
    print("=" * 50)
    print(f"项目根目录: {BASE_DIR}")
    print(f"配置文件路径: {CONFIG_PATH}")
    print(f"测试图片目录: {TEST_IMAGE_DIR}")
    print(f"模型路径: {MODEL_PATH}")
    print("=" * 50)

    # 初始化评估器 - 使用指定的模型路径
    evaluator = LicensePlateEvaluator(
        config_path=CONFIG_PATH,
        test_image_dir=TEST_IMAGE_DIR,
        model_path=MODEL_PATH  # 传递指定的模型路径
    )

    # 生成完整评估报告
    evaluator.generate_full_report()