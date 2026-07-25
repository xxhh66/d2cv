import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. 设备配置
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================
# 2. 数据加载（FashionMNIST）
# ============================================================
# FashionMNIST 是灰度图，需要转换为 RGB（3通道）才能用 AlexNet
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),          # AlexNet 要求 224x224
    transforms.Grayscale(num_output_channels=3),  # 灰度图转 RGB
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860],
                        std=[0.3530, 0.3530, 0.3530])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860, 0.2860, 0.2860],
                        std=[0.3530, 0.3530, 0.3530])
])

# 加载数据集
train_dataset = datasets.FashionMNIST('./datasets', train=True, download=True, transform=transform_train)
test_dataset = datasets.FashionMNIST('./datasets', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")
print(f"类别数: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)")

# ============================================================
# 3. 创建 AlexNet 并展示各种冻结策略
# ============================================================

# ============================================================
# 3.1 方法一：加载预训练模型，查看结构
# ============================================================
print("\n" + "=" * 60)
print("AlexNet 原始结构")
print("=" * 60)

model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
print(model)

# ============================================================
# 3.2 方法二：冻结所有层，只训练最后一层
# ============================================================
print("\n" + "=" * 60)
print("方法二：冻结所有层，只训练最后一层（分类层）")
print("=" * 60)

def freeze_all_layers(model):
    """冻结所有层的参数"""
    for param in model.parameters():
        param.requires_grad = False
    return model

model_freeze_all = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# 冻结所有层
freeze_all_layers(model_freeze_all)

# 修改最后一层（新层默认 requires_grad=True）
num_features = model_freeze_all.classifier[6].in_features
model_freeze_all.classifier[6] = nn.Linear(num_features, 10)

# 统计可训练参数
trainable_params = sum(p.numel() for p in model_freeze_all.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_freeze_all.parameters())
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"冻结比例: {(1 - trainable_params/total_params)*100:.2f}%")

# # ============================================================
# # 3.3 方法三：冻结特征提取层，训练分类器
# # ============================================================
# print("\n" + "=" * 60)
# print("方法三：冻结特征提取层，只训练分类器")
# print("=" * 60)

# def freeze_features_only(model):
#     """只冻结特征提取层（features），分类器可训练"""
#     for param in model.features.parameters():
#         param.requires_grad = False
#     return model

# model_freeze_features = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# # 冻结特征提取层
# freeze_features_only(model_freeze_features)

# # 修改分类器
# num_features = model_freeze_features.classifier[6].in_features
# model_freeze_features.classifier[6] = nn.Linear(num_features, 10)

# # 统计可训练参数
# trainable_params = sum(p.numel() for p in model_freeze_features.parameters() if p.requires_grad)
# total_params = sum(p.numel() for p in model_freeze_features.parameters())
# print(f"总参数量: {total_params:,}")
# print(f"可训练参数量: {trainable_params:,}")
# print(f"冻结比例: {(1 - trainable_params/total_params)*100:.2f}%")

# # ============================================================
# # 3.4 方法四：逐层冻结（冻结前N层）
# # ============================================================
# print("\n" + "=" * 60)
# print("方法四：逐层冻结（冻结前5个卷积层）")
# print("=" * 60)

# def freeze_layers_by_index(model, num_layers_to_freeze):
#     """冻结指定数量的层"""
#     layers = list(model.features.children())
#     for i, layer in enumerate(layers):
#         if i < num_layers_to_freeze:
#             for param in layer.parameters():
#                 param.requires_grad = False
#     return model

# model_freeze_layers = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# # 冻结前5层
# freeze_layers_by_index(model_freeze_layers, 5)

# # 修改最后一层
# num_features = model_freeze_layers.classifier[6].in_features
# model_freeze_layers.classifier[6] = nn.Linear(num_features, 10)

# # 统计可训练参数
# trainable_params = sum(p.numel() for p in model_freeze_layers.parameters() if p.requires_grad)
# total_params = sum(p.numel() for p in model_freeze_layers.parameters())
# print(f"总参数量: {total_params:,}")
# print(f"可训练参数量: {trainable_params:,}")
# print(f"冻结比例: {(1 - trainable_params/total_params)*100:.2f}%")

# # ============================================================
# # 3.5 方法五：修改分类器结构
# # ============================================================
# print("\n" + "=" * 60)
# print("方法五：修改分类器结构（添加/删除层）")
# print("=" * 60)

# def modify_classifier(model, num_classes=10):
#     """
#     修改分类器结构
#     可以添加层、删除层、改变维度
#     """
#     # 获取原始分类器
#     old_classifier = list(model.classifier.children())
    
#     # 创建新的分类器
#     # 保留前5层（FC1, ReLU, Dropout, FC2, ReLU, Dropout）
#     # 修改最后一层
#     new_classifier = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(9216, 4096),  # 输入维度 9216（6x6x256）
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.5),
#         nn.Linear(4096, 4096),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.3),         # 可以调整 dropout 率
#         nn.Linear(4096, 1024),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.3),
#         nn.Linear(1024, num_classes)
#     )
    
#     model.classifier = new_classifier
#     return model

# model_modify = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
# model_modify = modify_classifier(model_modify, num_classes=10)

# # 查看修改后的分类器
# print("修改后的分类器结构:")
# print(model_modify.classifier)

# # 统计参数量
# total_params = sum(p.numel() for p in model_modify.parameters())
# trainable_params = sum(p.numel() for p in model_modify.parameters() if p.requires_grad)
# print(f"总参数量: {total_params:,}")
# print(f"可训练参数量: {trainable_params:,}")

# # ============================================================
# # 3.6 方法六：使用不同学习率（特征层和分类器不同学习率）
# # ============================================================
# print("\n" + "=" * 60)
# print("方法六：特征层和分类器使用不同学习率")
# print("=" * 60)

# model_diff_lr = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# # 修改分类器
# num_features = model_diff_lr.classifier[6].in_features
# model_diff_lr.classifier[6] = nn.Linear(num_features, 10)

# # 设置不同学习率
# optimizer = optim.Adam([
#     {'params': model_diff_lr.features.parameters(), 'lr': 0.0001},  # 特征层：小学习率
#     {'params': model_diff_lr.classifier.parameters(), 'lr': 0.001}  # 分类器：大学习率
# ])

# print("优化器配置:")
# for i, param_group in enumerate(optimizer.param_groups):
#     print(f"  组 {i}: lr={param_group['lr']}, 参数数量={len(param_group['params'])}")

# # ============================================================
# # 4. 训练函数
# # ============================================================
# def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
#     """训练模型"""
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     print("\n" + "=" * 60)
#     print("开始训练")
#     print("=" * 60)
    
#     history = {'train_acc': [], 'test_acc': []}
    
#     for epoch in range(1, epochs + 1):
#         # 训练
#         model.train()
#         correct = 0
#         total = 0
        
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
            
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             _, pred = outputs.max(1)
#             total += labels.size(0)
#             correct += (pred == labels).sum().item()
        
#         train_acc = 100 * correct / total
        
#         # 测试
#         model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, pred = outputs.max(1)
#                 total += labels.size(0)
#                 correct += (pred == labels).sum().item()
        
#         test_acc = 100 * correct / total
        
#         history['train_acc'].append(train_acc)
#         history['test_acc'].append(test_acc)
        
#         print(f"Epoch {epoch:2d}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
#     return history

# # ============================================================
# # 5. 选择一种策略进行训练
# # ============================================================
# print("\n" + "=" * 60)
# print("选择策略：冻结特征层，训练分类器")
# print("=" * 60)

# # 选择一种策略
# model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# # 冻结特征层
# for param in model.features.parameters():
#     param.requires_grad = False

# # 修改分类器
# num_features = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_features, 10)

# # 统计
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"总参数量: {total_params:,}")
# print(f"可训练参数量: {trainable_params:,}")
# print(f"冻结比例: {(1 - trainable_params/total_params)*100:.2f}%")

# # 训练
# history = train_model(model, train_loader, test_loader, epochs=5, lr=0.001)

# # ============================================================
# # 6. 可视化训练结果
# # ============================================================
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history['train_acc'], label='Train Acc')
# plt.plot(history['test_acc'], label='Test Acc')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('训练曲线')
# plt.legend()
# plt.grid(True)

# # 显示一些预测示例
# plt.subplot(1, 2, 2)
# model.eval()
# images, labels = next(iter(test_loader))
# images_gpu = images[:8].to(device)
# with torch.no_grad():
#     outputs = model(images_gpu)
#     _, preds = outputs.max(1)

# class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     img = images[i].numpy().transpose(1, 2, 0)
#     img = (img - img.min()) / (img.max() - img.min())
#     plt.imshow(img)
#     plt.title(f'P: {class_names[preds[i]]}')
#     plt.axis('off')

# plt.tight_layout()
# # plt.savefig('alexnet_fashionmnist_results.png', dpi=300)
# plt.show()