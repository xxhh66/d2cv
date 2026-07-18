'''
① 在原始lenet网络结构上，将全连接层修改为全局池化层。
② 增加 Batch Normalization 层。
'''
import os 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,transforms
import time
from tqdm import tqdm
# 1. 运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# 2. 数据集及数据加载
train_dataset = datasets.FashionMNIST(
    root='./datasets',
    train=True,
    download=True,
    transform=ToTensor()
)
test_dataset = datasets.FashionMNIST(
    root='./datasets',
    train=False,
    download=True,
    transform=ToTensor()
)
batch_size = 256

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)
# 3. LeNet 网络结构
class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),

            nn.Flatten(),

            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes)
        )
    def forward(self,x):
        return self.net(x)

class LeNet_BN(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet_BN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),

            nn.Flatten(),

            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes)
        )
    def forward(self,x):
        return self.net(x)

class LeNet_GAP(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet_GAP, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),

            nn.AdaptiveAvgPool2d((1,1)),
            
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16,num_classes)
        )
    def forward(self,x):
        return self.net(x) 

# 4.模型参数对比
# ============================================================
# 对比函数
# ============================================================
def compare_models():
    """对比四个版本的参数量"""
    models = {
        'V1: 原始 LeNet': LeNet(),
        'V2: +BN': LeNet_BN(),
        'V3: +GAP': LeNet_GAP(),
    }
    
    # 详细参数分布
    print("\n详细参数分布:")
    print("-" * 70)

    for name,model in models.items():
        conv_params = 0
        bn_params = 0
        fc_params = 0

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv_params += sum(p.numel() for p in m.parameters())
            elif isinstance(m, nn.BatchNorm2d):
                bn_params += sum(p.numel() for p in m.parameters())
            elif isinstance(m, nn.Linear):
                fc_params += sum(p.numel() for p in m.parameters())
        
        total = conv_params + bn_params + fc_params
        
        print(f"\n{name}:")
        print(f"  卷积层参数: {conv_params:,}")
        print(f"  BN层参数:   {bn_params:,}")
        print(f"  全连接层:   {fc_params:,}")
        print(f"  总计:       {total:,}")

    print("=" * 70)


# 5. 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader,desc="Training",leave=False,ncols=80)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        current_acc = 100. * correct / total
        pbar.set_postfix(
            {
                'loss':f'{loss.item():.3f}',
                'acc':f'{current_acc:.1f}%'
            }
        )
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / total

    return avg_loss, accuracy
# 6. 评估函数
def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / total

    return avg_loss, accuracy
# 7. 对比训练
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    """完整训练流程"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    best_acc = 0.0
    start_time = time.time()

    model_name = model.__class__.__name__
    print(f"\n{'='*50}")
    print(f"训练: {model_name}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 保存最佳模型
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(model.state_dict(), f'best_{model_name}.pth')

        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:2d}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

    total_time = time.time() - start_time
    print(f"\n训练完成!")
    print(f"  最佳测试准确率: {best_acc:.2f}%")
    print(f"  总用时: {total_time:.1f}s")

    return history, best_acc

def train_all_versions(lr=1e-3,epochs=10):
    """训练所有版本并对比结果"""
    models = {
        'V1_Original': LeNet(),
        'V2_BN': LeNet_BN(),
        'V3_GAP': LeNet_GAP()
        # 'V4_BN_GAP': LeNet_V4_BN_GAP(),
    }

    results = {}

    for name, model in models.items():
        history, best_acc = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )
        
        results[name] = {
            'best_acc': best_acc,
            'params': sum(p.numel() for p in model.parameters()),
            'history': history
        }

    # 打印汇总结果
    print("\n" + "=" * 70)
    print("训练结果汇总")
    print("=" * 70)
    print(f"{'版本':<20} {'参数量':<15} {'最佳准确率':<15}")
    print("-" * 70)

    for name, data in results.items():
        print(f"{name:<20} {data['params']:<15,} {data['best_acc']:.2f}%")

    print("=" * 70)

    # 绘制对比图
    # plot_comparison(results)

    return results
# 9. 主程序

if __name__ == '__main__':
    # 对比模型参数量
    compare_models()

    # 训练
    train_all_versions()
