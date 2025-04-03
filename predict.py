import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from src.models.yolo import BottleDetector
from src.data.transforms import get_transforms
import matplotlib.pyplot as plt

class BottlePredictor:
    def __init__(self, model_path, config_path='config.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = BottleDetector(num_classes=self.config['num_classes'])
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 获取数据转换
        self.transform = get_transforms('test')
    
    def predict(self, image_path, conf_threshold=0.5, nms_threshold=0.4):
        """预测单张图片
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        Returns:
            boxes: 预测框
            scores: 置信度分数
        """
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size
        
        # 转换图片
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model.predict(
                image_tensor,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
        
        return predictions, orig_size
    
    def visualize(self, image_path, predictions, orig_size):
        """可视化预测结果
        Args:
            image_path: 原始图片路径
            predictions: 模型预测结果
            orig_size: 原始图片尺寸
        """
        # 加载原始图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制预测框
        for box, score in zip(predictions['boxes'], predictions['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # 转换回原始图片尺寸
            x1 = int(x1 * orig_size[0] / self.config['input_size'][0])
            y1 = int(y1 * orig_size[1] / self.config['input_size'][1])
            x2 = int(x2 * orig_size[0] / self.config['input_size'][0])
            y2 = int(y2 * orig_size[1] / self.config['input_size'][1])
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加置信度标签
            label = f'Bottle: {score:.2f}'
            cv2.putText(image, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def main():
    # 设置参数
    model_path = 'checkpoints/best_model.pth'
    image_path = 'data/test/sample.jpg'  # 测试图片路径
    
    # 创建预测器
    predictor = BottlePredictor(model_path)
    
    # 预测并可视化
    predictions, orig_size = predictor.predict(image_path)
    predictor.visualize(image_path, predictions, orig_size)

if __name__ == '__main__':
    main()