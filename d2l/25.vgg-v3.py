import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
import time

# ============================================================
# 1. VGG 模型定义（使用全局平均池化）
# ============================================================

def vgg_block(num_convs, in_channels, out_channels):
    """
    VGG 模块：多个卷积层 + 1个池化层
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  # 添加 BN 加速收敛
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# VGG 架构配置（可以使用更小的通道数）
VGG11_ARCH = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# 轻量版：VGG11_ARCH = ((1, 32), (1, 64), (2, 128), (2, 256), (2, 256))


class VGG11_GAP(nn.Module):
    """
    VGG11 网络 + 全局平均池化（替代全连接层）
    """
    def __init__(self, num_classes=10, in_channels=1, image_size=64):
        super().__init__()
        
        # 构建特征提取层
        conv_blks = []
        for num_convs, out_channels in VGG11_ARCH:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_blks)
        
        # ============ 关键修改：使用全局平均池化 ============
        # 替代原来的全连接层
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出 1x1
        
        # 只需要一个简单的全连接层（或者直接用 1x1 卷积）
        # 方式1：使用全连接层（参数少）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_channels, num_classes)  # in_channels = 512
        )
        
        # 方式2：使用 1x1 卷积（更少参数，但需要调整维度）
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(in_channels, num_classes, kernel_size=1),
        #     nn.Flatten()
        # )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)          # [B, C, 1, 1]
        x = self.classifier(x)   # [B, num_classes]
        return x


# ============================================================
# 2. 数据加载函数
# ============================================================

def get_data_loaders(batch_size=128, image_size=64, num_workers=0):
    """
    加载 Fashion-MNIST 数据集
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
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
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, len(train_data), len(test_data)


# ============================================================
# 3. 训练函数（与原代码相同）
# ============================================================

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if scaler is not None:
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, correct / total


def evaluate(model, test_loader, criterion, device, scaler=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, epochs, lr, device, save_path='best_vgg11_gap.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    start_time = time.time()
    
    print('\n' + '=' * 60)
    print('开始训练 (VGG11 + GAP)')
    print('=' * 60)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, scaler
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
        
        print(f'Epoch [{epoch:2d}/{epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print(f'  Time: {time.time()-epoch_start:.1f}s')
        print('-' * 60)
    
    total_time = time.time() - start_time
    print(f'\n完成！总时间: {total_time:.1f}s, 最佳测试准确率: {best_acc:.4f}')
    
    return history, best_acc


# ============================================================
# 4. 主函数
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # ---------- 超参数 ----------
    BATCH_SIZE = 128
    EPOCHS = 15
    LR = 1e-3
    IMAGE_SIZE = 64
    NUM_WORKERS = 0
    
    # ---------- 加载数据 ----------
    print('\n加载 Fashion-MNIST 数据集...')
    train_loader, test_loader, train_size, test_size = get_data_loaders(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f'训练集: {train_size}, 测试集: {test_size}')
    
    # ---------- 创建模型 ----------
    print('\n创建 VGG11_GAP 模型...')
    model = VGG11_GAP(
        num_classes=10,
        in_channels=1,
        image_size=IMAGE_SIZE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {total_params:,}')
    
    # ---------- 训练 ----------
    history, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        save_path='best_vgg11_gap.pth'
    )
    
    # ---------- 最终测试 ----------
    print('\n加载最佳模型进行最终评估...')
    model.load_state_dict(torch.load('best_vgg11_gap.pth'))
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f'最终测试准确率: {test_acc:.4f}')
    
    return model, history


# ============================================================
# 5. 程序入口
# ============================================================

if __name__ == '__main__':
    model, history = main()