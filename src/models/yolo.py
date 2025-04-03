import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, kernel_size=1),
            ConvBlock(channels // 2, channels, kernel_size=3)
        )
    
    def forward(self, x):
        return x + self.block(x)

class BottleDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # 主干网络
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ResBlock(64),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResBlock(128),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            ResBlock(256),
            ConvBlock(256, 512, kernel_size=3, stride=2),
            ResBlock(512)
        )
        
        # 检测头
        self.head = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1),
            nn.Conv2d(256, (num_classes + 5) * 3, kernel_size=1)  # 每个anchor预测(num_classes + 5)个值
        )
        
        # 3种尺度的anchor boxes
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ])
    
    def forward(self, x):
        batch_size = x.shape[0]
        features = self.backbone(x)
        output = self.head(features)
        
        # 重塑输出维度 [batch, anchors * (num_classes + 5), grid, grid]
        output = output.view(batch_size, 3, -1, output.shape[-2], output.shape[-1])
        return output

    def predict(self, x, conf_threshold=0.5, nms_threshold=0.4):
        """模型推理
        Args:
            x: 输入图像张量
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        Returns:
            预测框和对应的置信度
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            # 在这里实现后处理逻辑，包括：
            # 1. 将预测转换为边界框坐标
            # 2. 应用置信度阈值
            # 3. 执行非极大值抑制(NMS)
            # 具体实现根据实际需求调整
        return predictions