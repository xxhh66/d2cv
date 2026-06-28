# 多层感知机（MLP）原理详解

> 本文档全面讲解多层感知机（Multilayer Perceptron）的数学原理、网络结构、训练算法及实现细节。

---

## 目录

1. [概述](#1-概述)
2. [网络结构](#2-网络结构)
3. [前向传播](#3-前向传播)
4. [激活函数](#4-激活函数)
5. [损失函数](#5-损失函数)
6. [反向传播](#6-反向传播)
7. [参数更新与优化器](#7-参数更新与优化器)
8. [正则化技术](#8-正则化技术)
9. [完整训练流程](#9-完整训练流程)
10. [MLP 的局限性与改进](#10-mlp-的局限性与改进)
11. [PyTorch 实现示例](#11-pytorch-实现示例)
12. [关键公式汇总](#12-关键公式汇总)
13. [参考文献](#13-参考文献)

---

## 1. 概述

**多层感知机（Multilayer Perceptron, MLP）** 是一种前馈式人工神经网络，由输入层、一个或多个隐藏层和输出层构成。MLP 是深度学习的基石模型，其核心能力在于**通过层层堆叠的线性变换与非线性激活函数的复合**，可以逼近任意复杂的连续函数（通用近似定理，Universal Approximation Theorem）。

### 1.1 核心思想

MLP 的核心思想可以概括为：

> **特征逐层抽象**：每一层将上一层的输出作为输入，通过线性变换和非线性激活，逐步提取更高层次、更抽象的特征表示。

### 1.2 信息流向

MLP 的信息流向是**前向的**（从输入到输出），没有反馈连接（区别于循环神经网络 RNN）。训练时使用**反向传播算法**（Backpropagation）计算梯度并更新参数。

---

## 2. 网络结构

一个典型的 MLP 由以下三层组成：

| 层类型                     | 作用               | 神经元数量       |
| -------------------------- | ------------------ | ---------------- |
| **输入层（Input Layer）**  | 接收原始数据特征   | 等于特征维度 $d$ |
| **隐藏层（Hidden Layer）** | 进行特征变换与提取 | 超参数，可多层   |
| **输出层（Output Layer）** | 产生最终预测结果   | 等于任务输出维度 |

### 2.1 数学表示

设网络共有 $L$ 层（包含输出层，不包含输入层），第 $l$ 层的神经元数量为 $n^{(l)}$，其中：

- 输入层记为第 $0$ 层：$n^{(0)} = d$
- 输出层为第 $L$ 层：$n^{(L)} = \text{输出维度}$

第 $l$ 层的权重矩阵和偏置向量为：

$$ \mathbf{W}^{(l)} \in \mathbb{R}^{n^{(l-1)} \times n^{(l)}} $$

$$ \mathbf{b}^{(l)} \in \mathbb{R}^{n^{(l)}} $$

### 2.2 网络结构示意图

```
输入层          隐藏层          输出层
  x1 ────────► h1 ────────► y1
  x2 ────────► h2 ────────► y2
  x3 ────────► h3
  x4 ────────► h4
```

每一层的每个神经元都与下一层的所有神经元相连接，这就是**全连接（Fully Connected）** 结构。

---

## 3. 前向传播

前向传播（Forward Propagation）是指数据从输入层开始，逐层计算直到输出层的过程。

### 3.1 单个神经元的计算

对于第 $l$ 层的第 $j$ 个神经元，其计算过程为：

**第一步：加权求和**

$$ z_j^{(l)} = \sum_{i=1}^{n^{(l-1)}} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)} $$

或用向量形式表示为：

$$ \mathbf{z}^{(l)} = \mathbf{W}^{(l)T} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} $$

**第二步：激活函数变换**

$$ a_j^{(l)} = g^{(l)}(z_j^{(l)}) $$

其中 $g^{(l)}(\cdot)$ 为第 $l$ 层的激活函数，$\mathbf{a}^{(0)} = \mathbf{x}$ 为原始输入。

### 3.2 整体前向传播流程

```
输入: x = a^(0)
对于 l = 1 到 L:
    z^(l) = W^(l)T · a^(l-1) + b^(l)
    a^(l) = g^(l)(z^(l))
输出: y_hat = a^(L)
```

### 3.3 前向传播示例（单隐藏层）

假设有一个单隐藏层的 MLP：

$$ \mathbf{h} = \text{ReLU}(\mathbf{W}_1^T \mathbf{x} + \mathbf{b}_1) $$

$$ \hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_2^T \mathbf{h} + \mathbf{b}_2) $$

---

## 4. 激活函数

激活函数的核心作用是为网络引入**非线性**。如果没有非线性激活函数，多层线性变换等价于单层线性变换，网络将无法拟合非线性函数。

### 4.1 常见激活函数对比

| 激活函数       | 公式                                                                           | 导数                                                                 | 优点                     | 缺点                   |
| -------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------ | ---------------------- |
| **Sigmoid**    | $\sigma(x) = \frac{1}{1 + e^{-x}}$                                             | $\sigma(x)(1-\sigma(x))$                                             | 输出范围 (0,1)，概率解释 | 梯度消失，输出非零中心 |
| **Tanh**       | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$                                 | $1 - \tanh^2(x)$                                                     | 输出范围 (-1,1)，零中心  | 仍有梯度消失问题       |
| **ReLU**       | $\text{ReLU}(x) = \max(0, x)$                                                  | $\begin{cases}1 & x>0 \\ 0 & x \le 0 \end{cases}$                    | 计算简单，缓解梯度消失   | 神经元"死亡"问题       |
| **Leaky ReLU** | $\text{LReLU}(x) = \max(\alpha x, x)$                                          | $\begin{cases}1 & x>0 \\ \alpha & x \le 0 \end{cases}$               | 解决神经元死亡           | 需要调整 $\alpha$      |
| **ELU**        | $\text{ELU}(x) = \begin{cases} x & x>0 \\ \alpha(e^x-1) & x \le 0 \end{cases}$ | $\begin{cases}1 & x>0 \\ \text{ELU}(x)+\alpha & x \le 0 \end{cases}$ | 输出接近零中心           | 计算稍复杂             |
| **Softmax**    | $\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$                         | 雅可比矩阵                                                           | 输出概率分布             | 仅用于输出层           |

### 4.2 激活函数图像对比

```
Sigmoid:      Tanh:         ReLU:         Leaky ReLU:
  1 |          1 |            |              |
    |          /               |  /           | /
    |         /                | /            |/
    |        /                 |/             |
  0 |-------                0 |-------     0 |-------x
    |                         |              |\
    |                         |              | \
   -|                         |              |  \
```

### 4.3 激活函数选择建议

- **隐藏层**：优先使用 **ReLU** 或其变体（Leaky ReLU、ELU）
- **输出层**：
  - 二分类：Sigmoid
  - 多分类：Softmax
  - 回归：无激活函数（线性输出）或 ReLU（输出非负）

---

## 5. 损失函数

损失函数（Loss Function）衡量模型预测 $\hat{\mathbf{y}}$ 与真实标签 $\mathbf{y}$ 之间的差距。

### 5.1 回归任务

**均方误差（Mean Squared Error, MSE）：**

$$ \mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 $$

**平均绝对误差（Mean Absolute Error, MAE）：**

$$ \mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i| $$

**Huber Loss（结合 MSE 和 MAE 的优点）：**

$$ \mathcal{L}_{\text{Huber}} = \begin{cases} \frac{1}{2}(\hat{y} - y)^2 & |\hat{y} - y| \le \delta \\ \delta(|\hat{y} - y| - \frac{1}{2}\delta) & \text{otherwise} \end{cases} $$

### 5.2 分类任务

**二分类交叉熵（Binary Cross Entropy）：**

$$ \mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right] $$

**多分类交叉熵（Categorical Cross Entropy）：**

$$ \mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) $$

其中 $C$ 为类别数，$y_{i,c}$ 为 one-hot 编码的真实标签。

### 5.3 损失函数选择指南

| 任务类型     | 输出层激活   | 推荐损失函数              |
| ------------ | ------------ | ------------------------- |
| 二分类       | Sigmoid      | Binary Cross Entropy      |
| 多分类       | Softmax      | Categorical Cross Entropy |
| 多标签分类   | Sigmoid      | Binary Cross Entropy      |
| 回归（无界） | 线性         | MSE                       |
| 回归（有界） | Sigmoid/ReLU | MSE / MAE                 |

---

## 6. 反向传播

反向传播（Backpropagation）是训练 MLP 的核心算法，它通过**链式法则**（Chain Rule）计算损失函数对每个参数的梯度。

### 6.1 链式法则基础

对于复合函数 $f(g(x))$，其导数为：

$$ \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} $$

在神经网络中，损失函数 $\mathcal{L}$ 是各层参数的复合函数，梯度需要从输出层逐层传递到输入层。

### 6.2 反向传播的数学推导

设损失函数为 $\mathcal{L}$，对于第 $l$ 层，我们需要计算：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \quad \text{和} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} $$

定义**误差项** $\boldsymbol{\delta}^{(l)}$ 为损失对第 $l$ 层加权输入 $\mathbf{z}^{(l)}$ 的偏导数：

$$ \boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} $$

**输出层误差项**（以多分类交叉熵 + Softmax 为例）：

$$ \boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y} $$

**隐藏层误差项**（反向传播公式）：

$$ \boldsymbol{\delta}^{(l)} = \left( \mathbf{W}^{(l+1)} \boldsymbol{\delta}^{(l+1)} \right) \odot g'^{(l)}(\mathbf{z}^{(l)}) $$

其中 $\odot$ 表示逐元素相乘（Hadamard 积），$g'^{(l)}$ 是激活函数的导数。

**参数梯度**：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \mathbf{a}^{(l-1)} \boldsymbol{\delta}^{(l)T} $$

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)} $$

### 6.3 反向传播算法流程

```
1. 执行前向传播，计算并缓存所有 z^(l) 和 a^(l)
2. 计算输出层误差 δ^(L) = ∇_a L ⊙ g'^(L)(z^(L))
3. for l = L-1 到 1:
       δ^(l) = (W^(l+1) · δ^(l+1)) ⊙ g'^(l)(z^(l))
4. 计算所有参数梯度:
       ∂L/∂W^(l) = a^(l-1) · δ^(l)T
       ∂L/∂b^(l) = δ^(l)
5. 使用梯度下降法更新参数
```

### 6.4 梯度消失与梯度爆炸

**梯度消失（Vanishing Gradient）** ：

- 在深层网络中，梯度逐层相乘后变得极小
- 使用 Sigmoid 或 Tanh 激活函数时尤为严重
- 导致浅层参数几乎不更新，网络无法有效训练

**梯度爆炸（Exploding Gradient）** ：

- 梯度逐层相乘后变得极大
- 导致参数更新不稳定，损失值剧烈波动

**解决方案**：

| 问题     | 解决方案                                             |
| -------- | ---------------------------------------------------- |
| 梯度消失 | 使用 ReLU 类激活函数、残差连接（ResNet）             |
| 梯度爆炸 | 梯度裁剪（Gradient Clipping）、合理的权重初始化      |
| 通用     | 批归一化（Batch Normalization）、Layer Normalization |

---

## 7. 参数更新与优化器

### 7.1 梯度下降（Gradient Descent）

最基本的参数更新方式：

$$ \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta) $$

其中 $\eta$ 是**学习率（Learning Rate）**，$\theta$ 代表所有可训练参数。

### 7.2 梯度下降的三种变体

| 变体                                | 特点                    | 优点             | 缺点              |
| ----------------------------------- | ----------------------- | ---------------- | ----------------- |
| **批量梯度下降（BGD）**             | 使用全部数据计算梯度    | 梯度准确         | 大数据集慢        |
| **随机梯度下降（SGD）**             | 使用单个样本计算梯度    | 更新快           | 梯度噪声大        |
| **小批量梯度下降（Mini-batch GD）** | 使用 batch 数据计算梯度 | 平衡效率和稳定性 | 需选择 batch size |

### 7.3 常见优化器

| 优化器                | 特点                           | 公式                                                                      |
| --------------------- | ------------------------------ | ------------------------------------------------------------------------- |
| **SGD**               | 最基础，容易陷入局部最优       | $\theta \leftarrow \theta - \eta \nabla L$                                |
| **SGD + Momentum**    | 引入动量，加速收敛             | $v \leftarrow \gamma v + \eta \nabla L$<br>$\theta \leftarrow \theta - v$ |
| **Nesterov Momentum** | 前瞻性动量                     | 在 Momentum 基础上改进                                                    |
| **AdaGrad**           | 自适应学习率，稀疏特征效果好   | 累积梯度平方，调整学习率                                                  |
| **RMSprop**           | 改进 AdaGrad，使用指数移动平均 | 解决 AdaGrad 学习率单调递减问题                                           |
| **Adam**              | 结合 Momentum 和 RMSprop       | 最常用，自适应学习率 + 动量                                               |
| **AdamW**             | Adam + 权重衰减解耦            | 改进 Adam 的 L2 正则化                                                    |

### 7.4 Adam 优化器详解

Adam（Adaptive Moment Estimation）是当前最流行的优化器：

**算法步骤**：

1. 计算梯度：$g_t = \nabla_\theta L(\theta_{t-1})$
2. 更新一阶矩估计：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
3. 更新二阶矩估计：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
4. 偏差修正：$\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$
5. 更新参数：$\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

**超参数默认值**：

- $\eta = 0.001$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

### 7.5 学习率调度策略

| 策略                  | 说明                              |
| --------------------- | --------------------------------- |
| **固定学习率**        | 整个训练过程保持不变              |
| **阶梯衰减**          | 每隔固定 epoch 降低学习率         |
| **指数衰减**          | $ \eta_t = \eta_0 \cdot e^{-kt} $ |
| **余弦退火**          | 周期性调整学习率                  |
| **ReduceLROnPlateau** | 验证集性能停滞时降低学习率        |

---

## 8. 正则化技术

正则化用于防止模型过拟合，提升泛化能力。

### 8.1 L2 正则化（权重衰减）

在损失函数中添加权重平方和惩罚项：

$$ \mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{l} \| \mathbf{W}^{(l)} \|_F^2 $$

其中 $\lambda$ 是正则化强度超参数。

**效果**：鼓励权重趋近于零，使模型更简单、更平滑。

### 8.2 L1 正则化

$$ \mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \sum_{l} \| \mathbf{W}^{(l)} \|_1 $$

**效果**：产生稀疏权重，可用于特征选择。

### 8.3 Dropout

在训练过程中，以概率 $p$ 随机"丢弃"（置零）部分神经元的输出，迫使网络学习更鲁棒的特征。

$$ \mathbf{a}^{(l)} = \mathbf{a}^{(l)} \odot \mathbf{m}, \quad \mathbf{m} \sim \text{Bernoulli}(1-p) $$

测试时使用全部神经元，但输出乘以 $(1-p)$ 以保持期望值一致。

**Dropout 示意图**：

```
训练时:         测试时:
[●]──[●]       [●]──[●]
[●]──[ ]       [●]──[●]
[ ]──[●]       [●]──[●]
[●]──[●]       [●]──[●]
 丢弃一些连接    使用全部连接
```

### 8.4 早停法（Early Stopping）

在验证集性能不再提升时提前终止训练，防止过拟合。

**早停策略**：

1. 每轮训练后在验证集评估
2. 记录最佳验证集性能
3. 若连续 $n$ 轮（patience）未改善则停止训练
4. 恢复至最佳模型参数

### 8.5 批归一化（Batch Normalization）

对每一批数据的激活值进行归一化，加速收敛并起到一定正则化作用：

$$ \hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$

$$ \tilde{z} = \gamma \hat{z} + \beta $$

其中 $\gamma$ 和 $\beta$ 是可学习参数。

**优点**：

- 加速训练收敛
- 允许使用更大的学习率
- 起到一定的正则化效果
- 减轻对初始化的依赖

### 8.6 数据增强

通过对训练数据进行变换，生成更多样的训练样本：

| 数据类型 | 增强方法                             |
| -------- | ------------------------------------ |
| 图像     | 旋转、翻转、裁剪、色彩抖动、添加噪声 |
| 文本     | 同义词替换、回译、随机插入/删除      |
| 表格数据 | 添加噪声、特征扰动                   |

---

## 9. 完整训练流程

### 9.1 训练算法伪代码

```
算法：MLP 训练（梯度下降法）

输入：训练集 D = {(x_i, y_i)}，学习率 η，批量大小 B，轮数 E
输出：训练好的模型参数 Θ

1. 初始化网络参数 Θ（权重和偏置）
2. For epoch = 1 to E:
3.     将训练集随机打乱
4.     For each mini-batch {(x_i, y_i)} 大小 B:
5.         // 前向传播
6.         对 batch 中每个样本计算预测值 ŷ_i
7.         // 计算损失
8.         L = (1/B) * Σ loss(ŷ_i, y_i)
9.         // 反向传播
10.        计算 L 对每个参数的梯度
11.        // 参数更新
12.        Θ ← Θ - η * ∇Θ L
13.    // 可选：验证集评估
14.    在验证集上计算准确率/损失
15.    // 早停检查
16.    如果验证集性能不再提升，停止训练
17. 返回训练好的模型
```

### 9.2 训练流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 原始数据  │───▶│ 预处理   │───▶│ 划分数据集│                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                    (归一化/标准化)   训练/验证/测试              │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型训练阶段                              │
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│   │权重初始化│───▶│前向传播 │───▶│损失计算 │───▶│反向传播 │    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                      │          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐       │          │
│   │模型保存 │◀───│早停检查 │◀───│参数更新 │◀──────┘          │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                      │                                          │
│                      ▼                                          │
│                ┌─────────────┐                                  │
│                │ 是否收敛?   │──▶ 否 ──▶ 继续迭代              │
│                └─────────────┘                                  │
│                      │ 是                                       │
│                      ▼                                          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型评估阶段                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 加载模型  │───▶│ 测试集评估│───▶│ 性能报告  │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                    (准确率/F1/损失等)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 训练过程中的监控指标

| 指标           | 说明                   | 期望趋势   |
| -------------- | ---------------------- | ---------- |
| **训练损失**   | 模型在训练集上的损失   | 持续下降   |
| **验证损失**   | 模型在验证集上的损失   | 下降后趋稳 |
| **训练准确率** | 模型在训练集上的准确率 | 持续上升   |
| **验证准确率** | 模型在验证集上的准确率 | 上升后趋稳 |
| **学习率**     | 当前学习率值           | 可能衰减   |
| **梯度范数**   | 梯度的 L2 范数         | 保持稳定   |

### 9.4 过拟合与欠拟合判断

```
                    训练损失    验证损失    判断
                    高          高         欠拟合（模型容量不足）
                    低          高         过拟合（需要正则化）
                    低          低         良好拟合
                    高          低         数据问题或异常
```

---

## 10. MLP 的局限性与改进

### 10.1 主要局限性

| 局限性               | 说明                                   | 影响                       |
| -------------------- | -------------------------------------- | -------------------------- |
| **参数量大**         | 全连接结构导致参数量为 $O(n^2)$        | 难以处理高维数据，易过拟合 |
| **无法处理空间结构** | 对图像等数据忽略了像素间的空间位置关系 | 图像任务效果差             |
| **无法处理序列**     | 没有记忆机制，不擅长处理时序数据       | NLP 任务效果差             |
| **容易过拟合**       | 参数量大，需要大量数据和正则化         | 小数据集泛化差             |
| **对输入尺度敏感**   | 需要特征缩放（标准化/归一化）          | 预处理要求高               |
| **局部最优**         | 非凸优化问题，可能陷入局部最优         | 训练不充分                 |

### 10.2 改进方向

| 改进方向       | 对应模型/技术                | 解决的问题           |
| -------------- | ---------------------------- | -------------------- |
| 处理空间数据   | **CNN**（卷积神经网络）      | 图像、视频任务       |
| 处理序列数据   | **RNN / LSTM / GRU**         | 文本、语音、时序任务 |
| 处理图结构数据 | **GNN**（图神经网络）        | 社交网络、分子结构   |
| 处理长距离依赖 | **Transformer / 注意力机制** | NLP、多模态任务      |
| 减少参数量     | 稀疏连接、参数共享           | 提升效率             |
| 提升泛化能力   | 数据增强、强正则化           | 小数据集场景         |
| 提升训练效率   | 混合精度训练、分布式训练     | 大规模训练           |

### 10.3 MLP 的现代应用场景

尽管有上述局限性，MLP 在现代深度学习中仍有重要地位：

- ✅ 作为更复杂模型的基础组件（如 Transformer 中的 FFN）
- ✅ 处理表格数据（结构化数据）时仍然非常有效
- ✅ 特征变换和维度映射（如嵌入层）
- ✅ 小规模数据集上的基线模型
- ✅ 强化学习中的策略网络
- ✅ 元学习中的基础架构

---

## 11. PyTorch 实现示例

以下是使用 PyTorch 实现的完整 MLP 代码，用于 MNIST 手写数字分类。

### 11.1 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ==================== 1. 超参数设置 ====================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
INPUT_SIZE = 28 * 28      # MNIST 图像展平后的大小
HIDDEN_SIZES = [256, 128] # 两个隐藏层
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ==================== 2. 数据加载 ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值标准差
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ==================== 3. 定义 MLP 模型 ====================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))  # 批归一化
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # 将图像展平: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)

# ==================== 4. 初始化模型 ====================
model = MLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(DEVICE)

# 使用 He 初始化（PyTorch 默认已做，这里显式设置）
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# ==================== 5. 训练与测试函数 ====================
def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ==================== 6. 训练循环 ====================
train_losses, train_accs = [], []
test_losses, test_accs = [], []

print("\n" + "="*60)
print("开始训练".center(60))
print("="*60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch()
    test_loss, test_acc = evaluate(test_loader)
    
    # 学习率调度
    scheduler.step(test_loss)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:2d}/{EPOCHS} | '
          f'Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | '
          f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}% | '
          f'LR={current_lr:.6f}')

print("="*60)
print("训练完成".center(60))
print("="*60)

# ==================== 7. 可视化训练过程 ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 损失曲线
axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, EPOCHS+1), test_losses, 'r-o', label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(range(1, EPOCHS+1), train_accs, 'b-o', label='Train Accuracy')
axes[1].plot(range(1, EPOCHS+1), test_accs, 'r-o', label='Test Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 显示部分测试样本的预测结果
model.eval()
sample_data, sample_target = next(iter(test_loader))
sample_data, sample_target = sample_data.to(DEVICE), sample_target.to(DEVICE)
sample_output = model(sample_data)
_, sample_pred = torch.max(sample_output.data, 1)

# 显示前 10 个样本
axes[2].axis('off')
img_grid = torch.cat([sample_data[i] for i in range(10)], dim=2)
img_grid = img_grid.squeeze(0).cpu().numpy()
axes[2].imshow(img_grid, cmap='gray')
axes[2].set_title(f'Predictions: {sample_pred[:10].cpu().numpy().tolist()}\n'
                  f'Labels: {sample_target[:10].cpu().numpy().tolist()}')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.show()

# ==================== 8. 模型保存 ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accs': train_accs,
    'test_accs': test_accs,
}, 'mlp_mnist_model.pth')

print("\n模型已保存至 mlp_mnist_model.pth")
```

### 11.2 NumPy 从零实现（核心部分）

如果想更深入理解底层原理，以下是仅用 NumPy 实现的简易 MLP 核心部分：

