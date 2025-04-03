import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

class BottleDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """初始化数据集
        Args:
            root_dir (str): 数据集根目录
            transform: 数据增强转换
            mode (str): 训练/验证/测试模式
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # 图像和标注文件的路径
        self.img_dir = os.path.join(root_dir, 'images', mode)
        self.label_dir = os.path.join(root_dir, 'labels', mode)
        
        # 获取所有图像文件
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 加载标注信息
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # YOLO格式：class x_center y_center width height
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([class_id, x_center, y_center, width, height])
        
        boxes = torch.tensor(boxes)
        
        # 应用数据增强
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'boxes': boxes,
            'img_path': img_path
        }