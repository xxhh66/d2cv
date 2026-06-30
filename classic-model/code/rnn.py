import math
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# 随机种子
random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
num_steps = 8
num_hiddens = 64
lr = 1.0
num_epochs = 10

"""_summary_

输入序列 x: [t0, t1, t2, ..., t_seq_len]
              ↓    ↓    ↓          ↓
    ┌─────────────────────────────────┐
    │         RNN 层 (nn.RNN)         │
    │  h_t = tanh(W_hh·h_{t-1} +     │
    │           W_xh·x_t + b_h)      │
    └─────────────────────────────────┘
              ↓    ↓    ↓          ↓
    隐藏状态: [h0, h1, h2, ..., h_seq_len]
              ↓    ↓    ↓          ↓
    ┌─────────────────────────────────┐
    │      全连接层 (nn.Linear)       │
    │    y_t = W_fc·h_t + b_fc       │
    └─────────────────────────────────┘
              ↓    ↓    ↓          ↓
    输出:     [y0, y1, y2, ..., y_seq_len]
"""
class RNNPyTorch(nn.Module):
    """
    使用PyTorch内置RNN实现的循环神经网络
    
    这个类封装了PyTorch的nn.RNN，并添加了一个全连接输出层，
    用于将RNN的隐藏状态映射到最终的输出维度。
    """    
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        """
        初始化RNN网络
        
        Args:
            input_size (int): 输入特征的维度
            hidden_size (int): 隐藏状态的维度
            output_size (int): 输出特征的维度
            num_layers (int): RNN的层数，默认为1
        """        
        super(RNNPyTorch,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,      # 每个时间步输入的特征维度
            hidden_size=hidden_size,    # 隐藏状态的维度
            num_layers=num_layers,      # RNN层数（堆叠的层数）
            batch_first=False,          # False表示输入形状为(seq_len, batch, feature)
            nonlinearity='tanh'         # 激活函数，可选'tanh'或'relu'
        )
        # 全连接层，将RNN输出的隐藏状态映射到目标输出维度
        self.fc = nn.Linear(
            in_features=hidden_size,    # 输入是RNN的隐藏状态
            out_features=output_size    # 输出是最终的预测值
        )
    def forward(self,x,h_prev=None):
        """
        前向传播函数
        
        输入数据形状说明：
        - x: (seq_len, batch_size, input_size)
          seq_len: 序列长度（时间步数）
          batch_size: 批次大小
          input_size: 每个时间步的输入特征维度
        
        - h_prev: (num_layers, batch_size, hidden_size)
          num_layers: RNN层数
          batch_size: 批次大小
          hidden_size: 隐藏状态维度
        
        Returns:
            output: (seq_len, batch_size, output_size) 所有时间步的输出
            h_n: (num_layers, batch_size, hidden_size) 最后一个时间步的隐藏状态
        """        
        device = x.device
        # ========== 步骤1: 初始化隐藏状态 ==========
        # 如果调用者没有提供初始隐藏状态，则创建全零张量
        if h_prev is None:
            # x.size(1) 获取batch_size
            # 创建形状为 (num_layers, batch_size, hidden_size) 的零张量
            h_prev = torch.zeros(
                self.num_layers,    # 层数
                x.size(1),          # batch_size
                self.hidden_size,   # 隐藏状态维度
                device=device
            )
        else:
            if h_prev.device !=device:
                h_prev = h_prev.to(device)
        
            # 注：在实践中，也可以使用随机初始化或从上一批次的最后状态继续
        
        # ========== 步骤2: RNN核心计算 ==========
        # 调用PyTorch的内置RNN进行前向传播
        # 这一步自动完成了所有时间步的循环计算
        rnn_out, h_n = self.rnn(x, h_prev)
        """
        rnn_out: (seq_len, batch_size, hidden_size)
            - 包含所有时间步的隐藏状态
            - rnn_out[t] 表示第t个时间步所有样本的隐藏状态
        
        h_n: (num_layers, batch_size, hidden_size)
            - 只包含最后一个时间步的隐藏状态
            - 用于下一时刻的初始状态或作为序列的最终表示
        """
        
        # ========== 步骤3: 重塑数据以应用全连接层 ==========
        # 获取序列长度和批次大小
        seq_len, batch_size, _ = rnn_out.size()
        
        # 将 rnn_out 从 3D 张量重塑为 2D 张量
        # 原始形状: (seq_len, batch_size, hidden_size)
        # 重塑后:   (seq_len * batch_size, hidden_size)
        # 这样做的原因：全连接层(Linear)期望输入是2D的
        rnn_out_reshaped = rnn_out.view(seq_len * batch_size, -1)
        # view中的-1表示自动推断该维度的大小
        # 这里 -1 会被推断为 hidden_size
        
        # ========== 步骤4: 通过全连接层 ==========
        # 对重塑后的数据应用全连接层
        # 输入: (seq_len * batch_size, hidden_size)
        # 输出: (seq_len * batch_size, output_size)
        output = self.fc(rnn_out_reshaped)
        
        # ========== 步骤5: 恢复原始形状 ==========
        # 将输出从 2D 重塑回 3D
        # 从: (seq_len * batch_size, output_size)
        # 到: (seq_len, batch_size, output_size)
        output = output.view(seq_len, batch_size, -1)
        # 这里的 -1 会被推断为 output_size
        
        # ========== 返回结果 ==========
        # output: 所有时间步的输出，用于计算损失
        # h_n: 最后一个时间步的隐藏状态，可用于继续生成序列
        return output, h_n

if __name__ =="__main__":
    print("="*60)
    #　创建模型
    model = RNNPyTorch(input_size=10, hidden_size=64, output_size=5, num_layers=2)
    print(model)
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)

    summary(model=model,input_size=(50,10),batch_size=32,device=str(device))
