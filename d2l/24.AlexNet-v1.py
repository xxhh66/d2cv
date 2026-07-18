import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing

# ============ 1. Windows 多进程设置 ============
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ============ 2. 设备配置 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'PyTorch 版本: {torch.__version__}')
print("=" * 60)

# ============ 3. 超参数配置 ============
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
IMAGE_SIZE = 64
NUM_WORKERS = 0

# ============ 4. 数据预处理 ============
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# ============ 5. 加载数据集 ============
print("加载 Fashion-MNIST 数据集...")
train_dataset = datasets.FashionMNIST(
    root="./datasets",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.FashionMNIST(
    root="./datasets",
    train=False,
    download=True,
    transform=transform_test
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# ============ 6. 数据加载器 ============
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"数据加载器配置: num_workers={NUM_WORKERS}, batch_size={BATCH_SIZE}")
print("=" * 60)

# ============ 7. AlexNet 模型定义（使用全局平均池化） ============
class AlexNet_GAP(nn.Module):
    """
    AlexNet + 全局平均池化（极简版，无自定义初始化）
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ============ 8. 创建模型 ============
print("创建 AlexNet_GAP 模型...")
model = AlexNet_GAP(num_classes=10).to(device)

# 打印模型结构（可选）
# print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print("=" * 60)

# ============ 9. 损失函数和优化器 ============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ============ 10. 混合精度训练 ============
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda') if device.type == 'cuda' else None
use_amp = scaler is not None
print(f"混合精度训练: {'启用' if use_amp else '禁用（CPU模式）'}")

# ============ 11. 训练和评估函数 ============
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False, ncols=80)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{current_acc:.1f}%'
        })
    
    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, loader, device, scaler=None):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False, ncols=80)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            current_acc = 100. * correct / total
            pbar.set_postfix({'acc': f'{current_acc:.1f}%'})
    
    return 100. * correct / total

# # ============ 12. 训练循环 ============
# print("\n" + "=" * 60)
# print("开始训练 AlexNet_GAP")
# print("=" * 60)

# start_time = time.time()
# history = {
#     'train_loss': [],
#     'train_acc': [],
#     'test_acc': [],
#     'epoch_time': []
# }

# best_test_acc = 0.0

# for epoch in range(1, EPOCHS + 1):
#     epoch_start = time.time()
    
#     if device.type == 'cuda':
#         torch.cuda.empty_cache()
    
#     # 训练
#     train_loss, train_acc = train_epoch(
#         model, train_loader, criterion, optimizer, device, scaler
#     )
    
#     # 更新学习率
#     scheduler.step()
    
#     # 评估
#     test_acc = evaluate(model, test_loader, device, scaler)
    
#     # 记录
#     history['train_loss'].append(train_loss)
#     history['train_acc'].append(train_acc)
#     history['test_acc'].append(test_acc)
    
#     epoch_time = time.time() - epoch_start
#     history['epoch_time'].append(epoch_time)
    
#     # 保存最佳模型
#     if test_acc > best_test_acc:
#         best_test_acc = test_acc
#         torch.save(model.state_dict(), 'best_alexnet_gap_model.pth')
#         print("  ✓ 保存最佳模型")
    
#     # 显存使用
#     if device.type == 'cuda':
#         allocated = torch.cuda.memory_allocated() / 1024**3
#         cached = torch.cuda.memory_reserved() / 1024**3
#         memory_info = f", GPU Mem: {allocated:.2f}/{cached:.2f}GB"
#     else:
#         memory_info = ""
    
#     # 打印结果
#     print(f"\nEpoch [{epoch:2d}/{EPOCHS}]")
#     print(f"  Loss: {train_loss:.4f}")
#     print(f"  Train Acc: {train_acc:.2f}%")
#     print(f"  Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
#     print(f"  Time: {epoch_time:.1f}s{memory_info}")
#     print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
#     print("-" * 60)

# total_time = time.time() - start_time

# print("\n" + "=" * 60)
# print("训练完成！")
# print("=" * 60)
# print(f"总训练时间: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
# print(f"平均每个 Epoch: {total_time/EPOCHS:.1f}s")
# print(f"最佳测试准确率: {best_test_acc:.2f}%")
# print("=" * 60)

# # ============ 13. 可视化训练结果 ============
# print("\n绘制训练曲线...")

# plt.figure(figsize=(14, 5))

# # 损失曲线
# plt.subplot(1, 2, 1)
# plt.plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('AlexNet_GAP Training Loss')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 准确率曲线
# plt.subplot(1, 2, 2)
# plt.plot(history['train_acc'], 'r-', linewidth=2, label='Train Acc')
# plt.plot(history['test_acc'], 'g-', linewidth=2, label='Test Acc')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('AlexNet_GAP Accuracy')
# plt.ylim(0, 100)
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('alexnet_gap_training_results.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("训练结果图已保存为 'alexnet_gap_training_results.png'")

