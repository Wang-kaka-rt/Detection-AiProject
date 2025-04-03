import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.data import BottleDataset, get_transforms
from src.models.yolo import BottleDetector
from tqdm import tqdm
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def train(config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 创建数据集和数据加载器
    train_dataset = BottleDataset(
        root_dir=config['data_dir'],
        transform=get_transforms('train'),
        mode='train'
    )
    val_dataset = BottleDataset(
        root_dir=config['data_dir'],
        transform=get_transforms('val'),
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    model = BottleDetector(num_classes=config['num_classes']).to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.1
    )
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for batch in train_bar:
            images = batch['image'].to(device)
            targets = batch['boxes'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失（这里需要实现具体的损失计算逻辑）
            loss = compute_loss(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        # 验证
        val_loss = validate(model, val_loader, device)
        logging.info(f'Validation Loss: {val_loss:.4f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            logging.info(f'Saved best model to {save_path}')

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['boxes'].to(device)
            
            outputs = model(images)
            loss = compute_loss(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def compute_loss(predictions, targets):
    # 这里需要实现具体的损失计算逻辑
    # 包括边界框回归损失和分类损失
    # 暂时返回一个虚拟的损失值
    return torch.tensor(1.0, requires_grad=True)

if __name__ == '__main__':
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 设置日志
    setup_logging()
    
    # 开始训练
    train(config)