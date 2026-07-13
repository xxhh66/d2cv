import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')
# 超参数
batch_size = 64
epochs = 10

# 数据处理transforms
transform = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)) # 遍历数据集，计算均值和方差
])

# 数据集
train_dataset = datasets.FashionMNIST(
    root="./datasets",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./datasets",
    train=False,
    download=True,
    transform=transform
)

# 数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False
)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6,4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )
        
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = AlexNet().to(device)
print(model)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optim = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

def train(model,loader):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0

    for images,labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optim.zero_grad()
        loss.backward()

        optim.step()
        total_loss += loss.item()
        _,pred = torch.max(outputs,1)
        total += labels.size(0)

        correct += (pred==labels).sum().item()

    return total_loss/len(loader),correct/total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images,labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _,pred = torch.max(outputs,1)
            total += labels.size(0)

            correct = (pred==labels).sum(0).item()
    return correct/total


for epoch in range(epochs):

    train_loss, train_acc = train(model, train_loader)

    # test_acc = evaluate(model, test_loader)

    print(
        f"Epoch [{epoch+1:2d}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        # f"Test Acc: {test_acc:.4f}"
    )