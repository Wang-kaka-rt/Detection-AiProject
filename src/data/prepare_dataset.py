import os
import shutil
import random
from pathlib import Path

def prepare_dataset(source_dir, target_dir, split_ratio=(0.7, 0.2, 0.1)):
    """将数据集按比例分配到训练集、验证集和测试集
    Args:
        source_dir: Bottle Images文件夹路径
        target_dir: 目标数据集路径
        split_ratio: 训练集、验证集、测试集的比例，默认为(0.7, 0.2, 0.1)
    """
    # 确保目标目录存在
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(target_dir, subdir, split), exist_ok=True)
    
    # 获取所有图片文件
    image_files = []
    class_names = ['Beer Bottles', 'Plastic Bottles', 'Soda Bottle', 'Water Bottle', 'Wine Bottle']
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        if os.path.exists(class_dir):
            for img in os.listdir(class_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append((os.path.join(class_dir, img), class_mapping[class_name]))
    
    # 随机打乱数据集
    random.shuffle(image_files)
    
    # 计算分割点
    total = len(image_files)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    # 分配数据集
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # 复制图片并生成标注文件
    for split_name, files in splits.items():
        for img_path, class_id in files:
            # 处理图片
            img_name = os.path.basename(img_path)
            target_img_path = os.path.join(target_dir, 'images', split_name, img_name)
            shutil.copy2(img_path, target_img_path)
            
            # 生成标注文件
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(target_dir, 'labels', split_name, label_name)
            
            # 获取图片尺寸（这里假设所有目标占据整个图片）
            with open(label_path, 'w') as f:
                # YOLO格式：<class> <x_center> <y_center> <width> <height>
                # 这里简单处理，假设目标占据整个图片的80%
                f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

def main():
    source_dir = os.path.join(os.getcwd(), 'Bottle Images')
    target_dir = os.path.join(os.getcwd(), 'data')
    
    prepare_dataset(source_dir, target_dir)
    print("数据集准备完成！")

if __name__ == '__main__':
    main()