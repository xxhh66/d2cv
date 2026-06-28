import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 超参数设置
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
INPUT_SIZE = 28 * 28   # MNIST 图像展平后的大小
HIDDEN_SIZES = [256, 128]  # 两个隐藏层
NUM_CLASSES = 10

# 2. 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值标准差
])

train_dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 定义 MLP 模型

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        
        # 使用 nn.ModuleList 或直接定义每一层
        # 方式一：直接定义所有层（适用于固定层数）
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        
        # 方式二：使用 Sequential 但手动展开（更清晰）
        # self.network = nn.Sequential(
        #     nn.Linear(input_size, hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_sizes[1], num_classes)
        # )
    
    def forward(self, x):
        # 将图像展平: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        
        # 方式一的前向传播
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
        
        # 方式二的前向传播（使用 Sequential）
        # return self.network(x)

# 4. 初始化模型、损失函数、优化器
model = MLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. 训练函数
def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 6. 测试函数
def test():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 7. 训练循环
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train()
    test_loss, test_acc = test()
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | '
          f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%')

# 8. 可视化训练过程
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
ax1.plot(range(1, EPOCHS+1), test_losses, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), train_accs, label='Train Accuracy')
ax2.plot(range(1, EPOCHS+1), test_accs, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curves')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()