import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

# 基本用法
transforms.Resize(244)              # 最短边缩放到244，保持宽高比
transforms.Resize((244, 244))       # 精确缩放到244x244（可能拉伸）
transforms.Resize(244, max_size=300) # 最短边244，最长边不超过300

# 实际效果示例
from PIL import Image
import torchvision.transforms as transforms

# 假设原始图像尺寸：400x600 (宽x高)
img = Image.open('test.jpg')
print(f"原始尺寸: {img.size}")  # (400, 600)

# 方式1：指定最短边
transform1 = transforms.Resize(244)
img1 = transform1(img)
print(f"缩放后: {img1.size}")  # (244, 366) - 保持宽高比

# 方式2：指定精确尺寸
transform2 = transforms.Resize((244, 244))
img2 = transform2(img)
print(f"精确缩放: {img2.size}")  # (244, 244) - 可能拉伸