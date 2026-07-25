import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class LeafDatasets(Dataset):
    def __init__(self,data_dir,mode='train',transform =None):
        """
        参数:
            data_dir (str): 数据根目录 (如 './data/classify-leaves')
            mode (str): 'train' 或 'test'
            transform: 图像预处理/数据增强操作
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        if mode =='train':
            self.csv_path = os.path.join(data_dir,'train.csv')
            self.df = pd.read_csv(self.csv_path)
            self.image_col = 'image_id'
            self.label_col = 'label'
            self.classes = sorted(self.df[self.label_col].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        else: # test mode
            self.csv_path = os.path.join(data_dir, 'test.csv')
            self.df = pd.read_csv(self.csv_path)
            self.image_col = 'image_id' # 测试集列名可能是 'image_id'

        # 设定图像文件夹路径
        self.img_folder = os.path.join(data_dir, f'{mode}_images')
        print(f"Initialized {mode} dataset with {len(self.df)} samples.")
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # 获取图像文件名和标签
        row = self.df.iloc[index]
        img_name = row[self.image_col]
        if not img_name.endswith(('.jpg','.jpeg','.png')):
            img_name = f"{img_name}.jpg"
        img_path = os.path.join(self.img_folder, img_name)
        return super().__getitem__(index)
    
        # 加载并转换图片
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)
        
        if self.mode =="train":
            label_str = row[self.label_col]
            label_idx = self.class_to_idx[label_str]
            return image, label_idx
        else:
            # 对于测试集，通常返回图像和原始ID，方便提交
            return image, img_name         


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # nn.AdaptiveAvgPool2d((6, 6))
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 使用示例
if __name__ == "__main__":
    # 定义数据预处理和增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]) # ImageNet 标准归一化
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 实例化数据集
    train_dataset = LeafDatasets(
        data_dir='./datasets/classify-leaves', 
        mode='train', 
        transform=train_transform
    )
    
    test_dataset = LeafDatasets(
        data_dir='./datasets/classify-leaves', 
        mode='test', 
        transform=test_transform
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")


    torchvision.models.AlexNet()