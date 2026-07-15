import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device:{device}')

batch_size = 128
epochs = 10
lr = 1e-3
image_size = 64

# 数据增强
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
# 训练 测试数据
train_data = datasets.FashionMNIST(
    root="./datasets",
    train=True,
    download=True,
    transform=transform
)
test_data = datasets.FashionMNIST(
    root="./datasets",
    train=False,
    download=True,
    transform=transform
)
# 数据加载器
train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False
)
print(f'train data len:{len(train_data)},test data len:{test_data}')

# ============ 3. VGG 模块 ============
def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*layers)

VGG11_ARCH = ((1,64),(1,128),(2,256),(2,512),(2,512))

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        conv_blks = []
        in_channels = 1
        for num_convs,out_channels in VGG11_ARCH:
            conv_blks.append(vgg_block(num_convs=num_convs,in_channels=in_channels,out_channels=out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_blks)
        
        # 计算全连接层输入
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            dummy = self.features(dummy)
            fc_input = dummy.view(1, -1).size(1)
        print(f'FC input size: {fc_input}')

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# ============ 5. 创建模型 ============
model = VGG11().to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ============ 6. 训练准备 ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)