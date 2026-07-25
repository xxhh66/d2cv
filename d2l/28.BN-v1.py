import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from d2l import torch as d2l

def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    """  
        X为输入，
        gamma、beta为学的参数。
        moving_mean、moving_var为全局的均值、方差。
        moving_mean = \mu 
        moving_var : \sigma^2
        eps为避免除0的参数。
        momentum为更新moving_mean、moving_var的。 

        1. 标准化（Normalize）：     x̂ = (x - μ) / √(σ² + ε)
        2. 缩放和平移（Scale & Shift）： y = \gamma * x̂ + beta
    """
    # 'is_grad_enabled' 来判断当前模式是训练模式还是预测模式。就是在做推理的时候，推理不需要反向传播，所以不需要计算梯度    
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
    else:
         # 批量数+通道数+图片高+图片宽=4
        assert len(X.shape) in (2,4)
        # 2 表示2表示有两个维度，样本和特征，表示全连接层应该是：2 代表全连接层 (batch_size, feature)
        if len(X.shape)==2:
            # 按行求均值，即对每一列求一个均值出来。mean为1*n的行向量  
            mean = X.mean(dim=0)
            # 方差也是行向量
            var =((X-mean)**2).mean(dim=0)
        # 4 表示卷积层
        else:
            # 0为批量大小，1为输出通道，2、3为高宽。这里是沿着通道维度求均值，0->batch内不同样本，2 3 ->同一通道层的所有值求均值，获得一个1xnx1x1的4D向量。
            mean =  X.mean(dim=(0,2,3),keepdim=True)
            var = ((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
        
        X_hat = (X-mean) / torch.sqrt(var + eps)
        # 累加，将计算的均值累积到全局的均值上，更新moving_mean
        moving_mean = momentum*moving_mean+(1-momentum)*mean
        # 当前全局的方差与当前算的方差做加权平均，最后会无限逼近真实的方差。仅训练时更新，推理时不更新
        moving_var = momentum * moving_var + (1.0 - momentum)* var
    Y = gamma*X_hat +beta
    return Y,moving_mean.data,moving_var.data

def test_bn_2d():
    """测试全连接层的 BN"""
    print("=" * 60)
    print("测试1：全连接层（2D输入）")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 参数
    N, D = 32, 10  # 32个样本，10个特征
    eps = 1e-5
    momentum = 0.9
    
    # 输入数据：均值5，方差4
    X = torch.randn(N, D) * 2 + 5
    
    # 可学习参数
    gamma = torch.ones(D)
    beta = torch.zeros(D)
    
    # 全局统计量（初始化为0和1）
    moving_mean = torch.zeros(D)
    moving_var = torch.ones(D)
    
    print(f"输入形状: {X.shape}")
    print(f"输入均值: {X.mean():.4f}, 方差: {X.var():.4f}")
    print("-" * 60)
    
    # 模拟训练（多次前向传播，更新全局统计量）
    for step in range(5):
        # 注意：需要开启梯度，让 torch.is_grad_enabled() 返回 True
        # 这里用 torch.set_grad_enabled(True) 或使用 requires_grad_
        X.requires_grad_(False)  # 不计算梯度，但为了进入训练模式，用 torch.enable_grad()
        
        with torch.enable_grad():  # 进入训练模式
            Y, moving_mean_new, moving_var_new = batch_norm(
                X, gamma, beta, moving_mean, moving_var, eps, momentum
            )
            
            # 更新全局统计量
            moving_mean = moving_mean_new
            moving_var = moving_var_new
        
        print(f"Step {step+1}:")
        print(f"  输出均值: {Y.mean():.4f}, 方差: {Y.var():.4f}")
        print(f"  moving_mean[0]: {moving_mean[0]:.4f}, moving_var[0]: {moving_var[0]:.4f}")
    
    # 推理模式测试
    print("-" * 60)
    print("推理模式测试:")
    with torch.no_grad():  # 进入推理模式
        Y_infer, _, _ = batch_norm(
            X, gamma, beta, moving_mean, moving_var, eps, momentum
        )
        print(f"  推理输出均值: {Y_infer.mean():.4f}, 方差: {Y_infer.var():.4f}")

def test_bn_4d():
    """测试卷基层4D"""
    print("="*60)
    torch.manual_seed(42)

    N,C,H,W = 32,3,28,28
    eps = 1e-5
    momentum = 0.9

    X = torch.randn(N,C,H,W)*2+5
    # print(f'X:{X}')

    gamma = torch.ones(C,1,1)
    beta = torch.ones(C,1,1)

    moving_mean = torch.zeros(C,1,1)
    moving_var = torch.zeros(C,1,1)

    print(f'输入形状:{X.shape}')
    print(f'输入均值:{X.mean():.4f},方差:{X.var():.4f}')

    for step in range(5):
        with torch.enable_grad():
            Y, moving_mean_new, moving_var_new = batch_norm(
                X, gamma, beta, moving_mean, moving_var, eps, momentum
            )
            moving_mean = moving_mean_new
            moving_var = moving_var_new
        
        print(f"Step {step+1}:")
        print(f"  输出均值: {Y.mean():.4f}, 方差: {Y.var():.4f}")
        print(f"  moving_mean[0,0]: {moving_mean[0,0,0].item():.4f}, moving_var[0,0]: {moving_var[0,0,0].item():.4f}")
    
    # # 推理
    # print("-" * 60)
    # print("推理模式测试:")
    # with torch.no_grad():
    #     Y_infer, _, _ = batch_norm(
    #         X, gamma, beta, moving_mean, moving_var, eps, momentum
    #     )
    #     print(f"  推理输出均值: {Y_infer.mean():.4f}, 方差: {Y_infer.var():.4f}")






if __name__ =='__main__':
    # test_bn_2d()
    test_bn_4d()