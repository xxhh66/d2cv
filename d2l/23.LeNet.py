import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ==========================
# 1. 设置运行设备
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# 2. 下载 FashionMNIST 数据集
# ==========================
train_dataset = datasets.FashionMNIST(
    root="./datasets",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="./datasets",
    train=False,
    download=True,
    transform=ToTensor()
)

# ==========================
# 3. 创建 DataLoader
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
# 4. 定义 LeNet 网络
# ==========================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            # 输入：1×28×28
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                padding=2
            ),
            # nn.Sigmoid(),
            nn.ReLU(),

            # 输出：6×14×14
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),

            # 输出：16×10×10
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),
            # nn.Sigmoid(),
            nn.ReLU(),


            # 输出：16×5×5
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),

            # 展平
            nn.Flatten(),

            # 全连接层
            nn.Linear(16 * 5 * 5, 120),
            # nn.Sigmoid(),
            nn.ReLU(),


            nn.Linear(120, 84),
            # nn.Sigmoid(),
            nn.ReLU(),


            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)

# ==========================
# 5. 创建模型
# ==========================
model = LeNet().to(device)

print(model)

# ==========================
# 6. 定义损失函数
# ==========================
criterion = nn.CrossEntropyLoss()

# ==========================
# 7. 定义优化器
# ==========================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

# ==========================
# 8. 定义训练函数
# ==========================
def train(model, dataloader, criterion, optimizer):

    model.train()

    running_loss = 0.0
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

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# ==========================
# 9. 定义测试函数
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

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    return correct / total

# ==========================
# 10. 开始训练
# ==========================
epochs = 10

for epoch in range(epochs):

    train_loss, train_acc = train(
        model,
        train_loader,
        criterion,
        optimizer
    )

    test_acc = evaluate(
        model,
        test_loader
    )

    print(
        f"Epoch [{epoch+1:2d}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Test Acc: {test_acc:.4f}"
    )

# ==========================
# 11. 保存模型
# ==========================
torch.save(
    model.state_dict(),
    "lenet_fashionmnist.pth"
)

print("\n模型已保存为 lenet_fashionmnist.pth")