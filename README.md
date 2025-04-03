# 水瓶检测项目

基于PyTorch实现的水瓶目标检测项目，使用YOLOv5架构进行实现。

## 项目结构

```
.
├── config.yaml          # 配置文件
├── requirements.txt     # 项目依赖
├── train.py            # 训练脚本
├── predict.py          # 预测脚本
└── src/                # 源代码
    ├── data/           # 数据处理模块
    │   ├── dataset.py  # 数据集类
    │   └── transforms.py # 数据转换
    └── models/         # 模型定义
        └── yolo.py     # YOLO模型实现
```

## 环境要求

- Python 3.7+
- PyTorch 1.7.0+
- CUDA（可选，用于GPU加速）

## 安装步骤

1. 克隆项目：
```bash
git clone [项目地址]
cd Detection-AiProject
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

1. 准备数据集，按以下结构组织：
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. 标注格式：使用YOLO格式（每行：class_id x_center y_center width height）

## 训练模型

1. 修改 `config.yaml` 配置文件，设置合适的参数

2. 运行训练脚本：
```bash
python train.py
```

## 预测

使用训练好的模型进行预测：
```bash
python predict.py
```

## 模型评估

- 训练过程中会自动保存最佳模型
- 可以通过日志文件查看训练过程
- 支持验证集评估

## 注意事项

1. 确保数据集格式正确
2. 根据实际情况调整配置参数
3. 建议使用GPU进行训练

## 许可证

MIT License