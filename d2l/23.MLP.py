import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==========================
# 1. 设置运行设备
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# 2. 数据预处理
# ==========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# ==========================
# 3. 下载数据集
# ==========================
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

# ==========================
# 4. DataLoader
# ==========================
batch_size = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ==========================
# 5. 定义MLP网络
# ==========================
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.net = nn.Sequential(

            # 展平图片
            nn.Flatten(),

            # 第一层
            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            # 第二层
            nn.Linear(256, 128),
            nn.ReLU(),

            # 输出层
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# 创建模型
model = MLP().to(device)

print(model)

# ==========================
# 6. 损失函数
# ==========================
criterion = nn.CrossEntropyLoss()

# ==========================
# 7. 优化器
# ==========================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================
# 8. 训练函数
# ==========================
def train(model, dataloader):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        loss = criterion(outputs, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ==========================
# 9. 测试函数
# ==========================
def evaluate(model, dataloader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    return correct / total


# ==========================
# 10. 开始训练
# ==========================
epochs = 10

for epoch in range(epochs):

    train_loss, train_acc = train(model, train_loader)

    test_acc = evaluate(model, test_loader)

    print(
        f"Epoch [{epoch+1:2d}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Test Acc: {test_acc:.4f}"
    )

# ==========================
# 11. 保存模型
# ==========================
# torch.save(model.state_dict(), "mlp_fashionmnist.pth")

print("模型已保存！")