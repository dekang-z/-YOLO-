# 基于YOLOv11的车牌识别系统

这是一个面向本科大学生的车牌识别课程作业项目，基于YOLOv11目标检测和PaddleOCR文字识别技术实现。项目代码结构清晰，注释详细，适合作为深度学习入门实践的参考案例。

##  项目简介
本项目实现了一个端到端的车牌识别系统，能够自动检测图片和视频中的车牌区域，并识别出车牌号码。系统集成了倾斜车牌矫正、动态边界框扩展等实用技术，在复杂场景下也能保持较好的识别效果。

**主要功能：**
-  单张图片车牌识别
-  实时视频流/摄像头车牌识别
-  倾斜车牌自动矫正
-  中文车牌（含新能源）识别
-  批量图片处理
-  识别结果可视化

##  技术栈
| 技术 | 版本 | 说明 |
|------|------|------|
| Python | 3.8-3.10 | 主开发语言 |
| PyTorch | 1.13.1 | 深度学习框架 |
| YOLOv11 | 8.0.0 | 车牌检测模型 |
| PaddleOCR | 2.6.1.3 | 字符识别引擎 |
| OpenCV | 4.7.0.72 | 图像处理库 |
| CUDA | ≥11.3 | GPU加速支持（可选） |

##  环境配置（Windows系统）
### 1. 基础环境准备
1. 安装Anaconda（推荐）或Python官方版本
2. 打开Anaconda Prompt，创建虚拟环境：
```bash
conda create -n license-plate python=3.9
conda activate license-plate
```

### 2. 安装依赖库
```bash
# 安装PyTorch（GPU版本，如有NVIDIA显卡）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装其他依赖
pip install opencv-python==4.7.0.72
pip install ultralytics==8.0.0
pip install paddlepaddle==2.4.2
pip install paddleocr==2.6.1.3
pip install numpy==1.24.3
```

### 3. 验证环境
```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

##  数据集准备
本项目使用CCPD2020数据集，这是一个专门针对中国车牌的公开数据集。

1. **下载数据集**：从[CCPD官方仓库](https://github.com/detectRecog/CCPD)下载CCPD2020子集
2. **解压数据集**：将下载的压缩包解压到项目根目录下的`data`文件夹
3. **格式转换**：运行数据转换脚本，将CCPD格式转换为YOLO格式：
```bash
python ConvertYOLOFormat.py --data_path ./data/CCPD2020 --save_path ./data/yolo_dataset
```
4. **配置数据集**：修改`data.yaml`文件，指向转换后的数据集路径

##  快速开始
### 1. 直接使用预训练模型（推荐）
项目已提供训练好的权重文件，可直接运行识别：

```bash
# 单张图片识别
python picture.py --image_path ./test_images/test1.jpg

# 实时摄像头识别
python video.py --source 0

# 视频文件识别
python video.py --source ./test_videos/test.mp4
```

### 2. 重新训练模型（可选）
如果需要在自己的数据集上训练：
```bash
python train.py --epochs 50 --batch_size 8 --data ./data.yaml
```
训练完成后，最佳权重会保存在`runs/train/exp/weights/best.pt`

##  项目结构
```
license-plate-recognition/
├── data/                  # 数据集文件夹
├── runs/                  # 训练和推理结果
├── test_images/           # 测试图片
├── test_videos/           # 测试视频
├── ConvertYOLOFormat.py   # 数据集格式转换脚本
├── train.py              # 模型训练脚本
├── picture.py            # 图片识别脚本
├── video.py              # 视频/摄像头识别脚本
├── evaluation.py         # 模型评估脚本
├── data.yaml             # 数据集配置文件
└── README.md             # 项目说明文档
```

##  性能指标
在CCPD2020测试集上的测试结果（GTX 3050Ti）：
- 车牌检测mAP@0.5：87.2%
- 字符识别整体准确率：91.4%
- 单张图片平均处理时间：42ms
- 实时视频处理速度：~24 FPS

##  注意事项
1. **本项目仅用于学习参考**，请勿用于商业用途
2. 建议使用NVIDIA显卡以获得更好的性能，CPU模式下速度会较慢
3. 识别效果受光照、角度、遮挡等因素影响，复杂场景下准确率会有所下降
4. 项目中使用的预训练模型仅在CCPD数据集上训练，对特殊车牌（如军牌、警牌）识别效果有限

##  学习心得
通过这个项目，我主要学到了：
1. 目标检测与OCR技术的集成方法
2. 工业级AI系统的完整开发流程
3. 数据预处理在深度学习项目中的重要性
4. 透视变换等传统图像处理技术的实际应用
5. 如何评估和优化深度学习模型的性能

##  免责声明
本项目是本科课程作业的成果，仅供学习交流使用。项目中使用的所有数据集和开源库均归原作者所有。如有侵权，请联系删除。

##  交流与反馈
如果在复刻过程中遇到问题，欢迎提交Issue交流讨论。
