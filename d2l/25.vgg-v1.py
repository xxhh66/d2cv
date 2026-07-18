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


# 训练函数
def train_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs,labels) #模型预测结果（outputs）和真实标签（labels）之间的差距
        
        # 反向传播
        optimizer.zero_grad()
        # 计算模型所有参数的梯度
        loss.backward()
        # 优化参数
        optimizer.step()

        # 统计，loss.item()转换为python 数值
        total_loss += loss.item()*images.size(0)

        # 0：图片 batch 1:类别 class
        _,predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss/total
    acc = correct/total
    return avg_loss,acc

# 评估模型
def evaluate():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


train_losses = []
test_losses = []

train_accs = []
test_accs = []

start_time = time.time()

for epoch in range(epochs):

    train_loss, train_acc = train_epoch()
    test_loss, test_acc = evaluate()

    scheduler.step()

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_acc:.4f}"
    )

end_time = time.time()

print(f"\nTraining Time: {end_time-start_time:.2f}s")

    