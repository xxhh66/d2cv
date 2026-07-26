Attention is All you Need 全文翻译
==============================

著名的提出 Transformer 的文章，来自 NIPS 2017。由本人翻译，原文：

[Attention is All you Needarxiv.org/abs/1706.03762](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)

![](https://picx.zhimg.com/v2-aa4d530401d3d8036bf62215f0db84ff_1440w.jpg)

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

封面图截自动漫 [ブレンド・S](https://link.zhihu.com/?target=https%3A//blend-s.jp/) 第 12 集。

* * *

摘要
--

主流的[序列转换](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E8%BD%AC%E6%8D%A2&zhida_source=entity)（sequence transduction）模型都是编码器（encoder）和解码器（decoder）架构，并基于复杂的循环或[卷积神经网络](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity)实现。目前性能最好的模型还加入了注意力机制将编码器和解码器连接起来。我们提出了一种新的简单网络架构——Transformer，其仅使用注意力机制，完全不需要循环和卷积单元。两个[机器翻译](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91&zhida_source=entity)任务的实验表明，我们的模型有很高的质量，同时更具[并行性](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%B9%B6%E8%A1%8C%E6%80%A7&zhida_source=entity)、需要的训练时间也显著减少。我们的模型在 [WMT 2014](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=WMT+2014&zhida_source=entity) 英译德任务中获得了 28.4 的 BLEU 分数，比现有的最佳结果（包括[集成模型](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E9%9B%86%E6%88%90%E6%A8%A1%E5%9E%8B&zhida_source=entity)）提高了 2 BLEU 以上。对于 WMT 2014 英译法任务中，我们将模型在 8 个 GPU 上训练了 3.5 天，随之取得了单模型的最新 sota 结果， BLEU 分数为 41.0；比之当前文献中的最好的模型，我们的训练成本仅是他们的一小部分。我们将 Transformer 成功应用于大量及有限训练数据的英语[成分句法分析](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%88%90%E5%88%86%E5%8F%A5%E6%B3%95%E5%88%86%E6%9E%90&zhida_source=entity)（constituency parsing）工作，这表明 Transformer 可以很好地推广到其它任务。

1\. 介绍
------

[循环神经网络](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity)（Recurrent neural networks, RNN），特别是[长短期记忆网络](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9C&zhida_source=entity)（long short-term memory, LSTM） \[13\] 和门控循环（gated recurrent） \[7\] 神经网络；在序列建模和转换问题上，例如语言模型（language modeling）和机器翻译（machine translation），它们已建立了不可动摇的 sota 方法 \[35,2,5\]。此后，人们做出了许多努力，不断突破[循环语言模型](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%BE%AA%E7%8E%AF%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)和编码器-[解码器](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=3&q=%E8%A7%A3%E7%A0%81%E5%99%A8&zhida_source=entity)架构的界限 \[38,24,15\]。

[循环模型](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%BE%AA%E7%8E%AF%E6%A8%A1%E5%9E%8B&zhida_source=entity)通常会沿着输入和输出序列的符号位置进行计算。通过在计算时将位置与时间步（steps）对齐，其生成一系列隐藏状态 $h_t$ ，作为前一个隐藏状态 $h_{t−1}$ 与位置 $t$ 二者的函数。这些模型天生要求顺序操作，这阻碍了训练样本中的[并行化](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%B9%B6%E8%A1%8C%E5%8C%96&zhida_source=entity)，但在较长的序列上进行并行化十分重要，因为有限的内存限制了样本之间的[批处理](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%89%B9%E5%A4%84%E7%90%86&zhida_source=entity)。最近的工作通过[分解技巧](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%88%86%E8%A7%A3%E6%8A%80%E5%B7%A7&zhida_source=entity)（factorization tricks） \[21\] 和条件计算（conditional computation） \[32\] 显著提高了计算效率，同时后者还提高了的模型性能。然而，需要[顺序计算](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E9%A1%BA%E5%BA%8F%E8%AE%A1%E7%AE%97&zhida_source=entity)这一基本限制仍然存在。

在各种任务中，对于各种引人注目序列模型和[转换模型](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B&zhida_source=entity)，注意力机制（Attention）已然成为它们的一个重要组成部分，其允许对依赖关系进行建模，而不用考虑它们在输入或输出序列中的距离 \[2,19\]。然而，除了少数情况外 \[27\]，这种注意力机制都是与[循环网络](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%BE%AA%E7%8E%AF%E7%BD%91%E7%BB%9C&zhida_source=entity)结合使用的。

在这项工作中，我们提出了 Transformer，这是一种避免使用循环的模型架构，完全依赖注意力机制来描述输入和输出之间的全局依赖关系。Transformer 显著提高了[并行度](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%B9%B6%E8%A1%8C%E5%BA%A6&zhida_source=entity)，仅在 8 个 P100 GPU 上进行 12 小时的训练后，即可以在翻译质量方面达到新的 sota。

2\. 背景
------

“减少顺序计算”这一目标也构成了扩展神经 GPU（Extended Neural GPU） \[16\]、ByteNet \[18\] 和 ConvS2S \[9\] 的基础，所有这些都使用卷积神经网络作为基本模块，并行地计算所有输入和输出位置的隐藏表示（hidden representations）。在这些模型中，将来自两个任意输入或输出位置的信号关联起来所需的操作数，随位置之间的距离而增加， ConvS2S 为[线性增长](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E5%A2%9E%E9%95%BF&zhida_source=entity)， ByteNet 则是[对数](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%AF%B9%E6%95%B0&zhida_source=entity)增长。这使得学习远距离的依赖关系变得更加困难 \[12\]。在 Transformer 中，这被减少至常数次操作，但这也导致平均注意力加权位置信息而使有效分辨率降低，我们用多头注意力（Multi-Head Attention）来抵消这种影响，见 3.2 节。

自注意力（[Self-attention](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=Self-attention&zhida_source=entity)，有时称为内注意力 intra-attention）是一种将单一序列不同位置相关联的注意力机制，可以计算序列的表示形式。自注意力已成功应用于各种任务，包括阅读理解、抽象概括、文本蕴涵（textual entailment）和任务无关的句子表示学习等 \[4,27,28,22\]。

[端到端记忆网络](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%AB%AF%E5%88%B0%E7%AB%AF%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9C&zhida_source=entity)（End-to-end memory networks）基于循环注意力（recurrent attention）机制，而非[序列对齐循环](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E5%AF%B9%E9%BD%90%E5%BE%AA%E7%8E%AF&zhida_source=entity)（sequence-aligned recurrence），人们已证明其在简单语言问答和语言模型任务上有良好的表现 \[34\]。

然而，据我们所知，Transformer 是第一个完全依赖自注意力来计算其输入输出表示的转换模型，没有使用序列对齐 RNN 或卷积。在接下来的章节中，我们将描述 Transformer，说明[自注意力机制](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&zhida_source=entity)的动因，并讨论其相对于 \[17, 18\] 和 \[9\] 等模型的优势。

3\. 模型架构
--------

大多数有竞争力的[神经序列](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%A5%9E%E7%BB%8F%E5%BA%8F%E5%88%97&zhida_source=entity)转换模型都采用编码器-解码器结构 \[5,2,35\]。编码器将符号表示的输入序列 $(x_1, ..., x_n)$ 映射到连续的序列 $\mathbf{z} = (z_1, ..., z_n)$ 。给定 $\mathbf{z}$ ，解码器随之生成一个符号输出序列 $(y_1,...,y_m)$ ，一次生成一个元素。每一步中，模型都是[自回归](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%87%AA%E5%9B%9E%E5%BD%92&zhida_source=entity)（auto-regressive）\[10\] 的，生成下一个元素时，将先前生成的符号用作附加输入。

Transformer 遵循这个整体架构，对编码器和解码器使用多层堆叠的自注意力层，以及逐点（point-wise）的[全连接层](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82&zhida_source=entity)，分别如图 1 的左右两部分所示。

![](https://picx.zhimg.com/v2-a5c0f318eb3afd6c0d566040b0251569_1440w.jpg)

图 1：Transformer 模型架构

### 3.1 编码器和[解码器栈](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%A7%A3%E7%A0%81%E5%99%A8%E6%A0%88&zhida_source=entity)

**编码器**：编码器由 $N=6$ 个相同层组成的栈构成。每一层有两个子层，其一是[多头自注意力](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%A4%9A%E5%A4%B4%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B&zhida_source=entity)（multi-head self-attention）机制，其二是简单的位置全连接前馈网络。我们的两个子层都采用[残差连接](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5&zhida_source=entity)（residual connection）\[11\]，随之进行[层归一化](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%B1%82%E5%BD%92%E4%B8%80%E5%8C%96&zhida_source=entity)（layer normalization）\[1\]。换言之，每个子层的输出为 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$ ，其中 $\mathrm{Sublayer}(x)$ 是子层本身实现的函数。为方便残差连接，模型中所有子层及嵌入（embedding）层都生成 $d_{\text{model}}=512$ 维的输出。

**解码器**：解码器也由 $N=6$ 个相同层组成的栈构成。除了编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器栈的输出执行多头注意力。与编码器类似，我们对每个子层采用残差连接，随之进行层归一化。我们还修改了解码器栈中的[自注意力子层](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%AD%90%E5%B1%82&zhida_source=entity)，以防止以当前位置信息中被添加进后续的位置信息。这种掩码（mask）与偏移一个位置输出嵌入的相结合，保证位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

### 3.2 注意力

注意力函数的作用是：将查询（query）和一组键值对（key-value pairs）映射到输出，其中 query、keys、values 和输出都是[向量](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%90%91%E9%87%8F&zhida_source=entity)。输出是 values 的[加权和](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%8A%A0%E6%9D%83%E5%92%8C&zhida_source=entity)，其中分配给每个 value 的权重是由 query 与相应 key 的兼容函数（compatibility function）计算。

![](https://pic3.zhimg.com/v2-f961880eacca12ea3f0638baa4453fa2_1440w.jpg)

图 2：（左）缩放点积注意力；（右）多头注意力由多个并行运行的注意力层组成。

**3.2.1 缩放点积注意力（Scaled Dot-Product Attention）**

我们的特别注意力机制称作“缩放点积注意力”（图 2 左）。输入由 $d_k$ 维的 queries 和 keys 以及 $d_v$ 维的 values 组成。我们使用计算 query 和所有 keys 的点积，随之除以 $\sqrt{d_k}$ ，再应用 softmax 函数来获取 values 的权重。

实际应用中，我们将一组 queries 转换成一个[矩阵](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5&zhida_source=entity) $Q$ ，同时应用[注意力函数](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=2&q=%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%87%BD%E6%95%B0&zhida_source=entity)。keys 和 values 也同样被转换成矩阵 $K$ 和 $V$ 。按照如下方式计算输出矩阵：

$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\\$两种最常用的注意力函数是加性注意力（additive attention）\[2\] 和点积（乘法）注意力。点积注意力与我们的[算法](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%AE%97%E6%B3%95&zhida_source=entity)相同，只是没有 $\frac{1}{\sqrt{d_k}}$ 的缩放因子。加性注意力使用有单个隐藏层的前馈网络来计算兼容函数。虽然二者在理论复杂性上相似，但在实践中点积注意力更快、更节省空间，因其可以使用高度优化的[矩阵乘法](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95&zhida_source=entity)代码来实现。

对于较小的 $d_k$ 值，两种机制的表现相似，但对于较大的 $d_k$ 值，[加性注意力](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=3&q=%E5%8A%A0%E6%80%A7%E6%B3%A8%E6%84%8F%E5%8A%9B&zhida_source=entity)优于点积注意力，且无需进行缩放\[3\]。我们认为，对于较大的 $d_k$ 值，点积的[数量级](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%95%B0%E9%87%8F%E7%BA%A7&zhida_source=entity)会变大，从而会将 softmax 函数推入[梯度](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%A2%AF%E5%BA%A6&zhida_source=entity)极小的区域 【注：为说明点积变大的原因，假设 $q$ 和 $k$ 的分量是[独立随机变量](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%8B%AC%E7%AB%8B%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F&zhida_source=entity)，均值为 $0$ ，[方差](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%96%B9%E5%B7%AE&zhida_source=entity)为 $1$ 。那么它们点积 $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$ 的均值为 $0$ ，方差为 $d_k$ 】。为了抵消这种影响，我们将点积缩放 $\frac{1}{\sqrt{d_k}}$ 倍。

**3.2.2 多头注意力（Multi-Head Attention）**

我们发现，与其使用 $d_{\text{model}}$ 维的 keys、values 和 queries 执行单个注意力函数，使用学习到的不同[线性映射](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E6%98%A0%E5%B0%84&zhida_source=entity)分 $h$ 次将 queries、keys 和 values [线性投影](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E6%8A%95%E5%BD%B1&zhida_source=entity)到 $d_k$ 、 $d_k$ 和 $d_v$ 维则更有裨益。随后，在 queries、keys 和 values 的每个投影上，我们并行地执行注意力函数，产生 $d_v$ 维输出值，其被连接起来再次进行投影，产生最终值，如图 2 所示。

多头注意力使得模型同时关注来自不同位置的、不同表示子空间的信息。对于单一注意力头，均值运算反而会抑制之。

$\begin{align*} \mathrm{MultiHead}(Q, K, V) &= \mathrm{Concat}(\mathrm{head}_1, ..., \mathrm{head}_h)W^O\\ \text{where}~\mathrm{head}_i &= \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)\\ \end{align*}\\$其中投影操作为参数矩阵 $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$ 、 $W^K_i \in \mathbb{R}^{d_{\text {model}} \times d_k}$ 、 $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 和 $W^O \in \mathbb{R}^{ hd_v \times d_{\text{model}}}$ 。

在这项工作中，我们采用 $h=8$ 个并行注意力层或头（head）。每个头都采用 $d_k=d_v=d_{\text{model}}/h=64$ 。由于每个头的维度减少，总[计算成本](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%AE%A1%E7%AE%97%E6%88%90%E6%9C%AC&zhida_source=entity)与全维度的单头注意力相似。

**3.2.3 注意力在我们模型中的应用**

多头注意力在 Transformer 中有三种不同的使用方式：

*   在“编码器-解码器注意力（encoder-decoder attention）”层中，queries 来自先前的解码器层，而 keys 和 values 来自编码器的输出。这使得解码器中的每个位置都可关联到输入序列中的所有位置。这是在模仿序列到序列（sequence-to-sequence, seq2seq）模型中典型的编码器-解码器注意机制，例如 \[38,2,9\]。
*   编码器包含了自注意力层。在自注意力层中，所有 keys、values 和 queries 都来自同一位置，在本例中是编码器中前一层的输出。编码器中的每个位置可以关注到编码器上一层中的所有位置。
*   类似地，解码器中也包含自注意力层，这使得解码器中的每个位置都关注到解码器之前的所有位置（并包括当前位置）。为了保持解码器的自回归特性，需要防止解码器中的信息向左流动。在缩放点积注意力的内部，我们通过屏蔽（设置为−∞）softmax 输入中所有非法连接对应的值，从而实现了这一点。见图 2。

### 3.3 逐位置的[前馈神经网络](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity)

除了注意力子层之外，我们编码器与解码器中的每个层中都包含一个全连接前馈网络，该网络单独且相同地应用于每个位置。其由两个线性变换组成，中间有一个 ReLU 激活。

$ \mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2\\$虽然不同位置的线性变换是相同的，但它们在层与层之间采用不同的参数。另一种描述方式是两个核大小为 1 的卷积。输入和输出的维度为 $d_{\text{model}}=512$ ，内层的维度为 $d_{ff}=2048$ 。

### 3.4 [词嵌入](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%AF%8D%E5%B5%8C%E5%85%A5&zhida_source=entity)和 softmax

与其他序列转换模型类似，我们使用学习到的嵌入将输入和输出 tokens 转换为维度 $d_{\text{model}}$ 的向量。我们还使用常用的线性变换与 softmax 函数，将解码器输出转换为预测下一个 token 的概率。在我们的模型中，两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，类似于 \[30\]。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$ 。

### 3.5 位置编码

由于我们的模型不包含循环和卷积，所以为了使模型能够利用序列的顺序信息，我们必须注入一些有关序列中 tokens 的相对或绝对位置的信息。为此，我们将“位置编码（positional encodings）”添加到编码器和解码器栈底部的输入嵌入中。位置编码与嵌入具有相同的 $d_{\text{model}}$ 维度，因此可以将两者相加。位置编码有多种选择，既可以学习得到，也可以将其固定 \[9\]。

在本工作中，我们使用不同频率的[正弦和余弦](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%AD%A3%E5%BC%A6%E5%92%8C%E4%BD%99%E5%BC%A6&zhida_source=entity)函数：

$\begin{align*} PE_{(pos,2i)} &= \sin(pos / 10000^{2i/d_{\text{model}}}) \\ PE_{(pos,2i+1)} &= \cos(pos / 10000^{2i/d_{\text{model}}}) \end{align*}\\$其中 $pos$ 是位置， $i$ 是维度。换言之，位置编码的每个维度都对应于一个正弦曲线。波长呈从 $2\pi$ 到 $10000 \cdot 2\pi$ 的[几何级数](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%87%A0%E4%BD%95%E7%BA%A7%E6%95%B0&zhida_source=entity)。之所以选择此函数，是因为我们假设它可以让模型很容易地关注相对位置进行学习，因为对于任何固定偏移 $k$ ， $PE_{pos+k}$ 可以表示为 $PE_{pos }$ 的线性函数。

我们还尝试使用可学习的位置嵌入 \[9\]，发现这两种方法结果几乎相同（参见表 3 第 (E) 行）。我们选择正弦函数，因其可以令模型推断出的序列长度比训练期间遇到的序列更长。

4\. 为什么使用自注意力
-------------

在本节中，我们将从各个方面将自注意力层与循环层和卷积层进行比较，这些层都通常用于将用符号表示的一个可变长度序列 $(x_1, ..., x_n)$ 映射到另一个等长序列 $(z_1 , ..., z_n)$ ，其中 $x_i, z_i \in \mathbb{R}^d$ ，比如用于典型的序列转换编码器或解码器中的隐藏层。主要有三个方面促使我们使用自注意力。

其一，是关于每层的总计算复杂度。其二是可以并行化的计算量，以所需的最小顺序操作数来衡量。

其三，是网络中长距离依赖之间的路径长度。长距离依赖（long-range dependencies）的学习是许多序列转换任务中的一个关键挑战。有一个关键因素会对这种依赖性的学习能力产生影响：前向和后向信号在网络中必须经过的路径的长度。输入和输出序列中的任意位置组合之间的路径越短，学习长距离依赖关系就越容易 \[12\]。因此，我们还比较了由不同层类型组成的网络中、任意两个输入和输出位置之间的最大路径长度。

![](https://pic4.zhimg.com/v2-a1b50217f2ef04abe3ee442ea79ece59_1440w.jpg)

表 1：不同层类型的最大路径长度、每层复杂度和最小顺序操作数。n 是序列长度，d 是表示维度，k 为卷积核大小，r 是受限自注意力中邻域的大小。

如表 1 中所示，自注意力层将所有位置与常数个顺序执行操作相连，而循环层需要 $O(n)$ 次顺序操作。就计算复杂度而言，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快，使用 sota 的机器翻译模型表示句子（sentence representations）时，这是常见情况，例如 word-piece \[38\] 和 byte-pair \[31\] 表示。

对于涉及很长序列的任务，为了提高计算性能，可以对自注意力进行限制，仅考虑输入序列中以相应输出位置为中心的大小为 $r$ 的邻域。这会将最大路径长度增加到 $O(n/r)$ 。我们计划在未来的工作中对该方法进一步研究。

核宽度为 $k < n$ 的单个卷积层不会连接所有输入和输出位置对。要实现这点，需要在卷积核连续（contiguous kernels）的情况下堆叠 $O(n/k)$ 个卷积层，或者在空洞卷积（dilated convolutions）\[18\] 的情况下需要 $O(log_k(n))$ ，这增加了网络中任意两个位置之间的最长路径的长度。卷积层通常比循环层开销贵 $k$ 倍。然而，可分离卷积（Separable convolutions）\[6\] 大大降低了复杂性，可至 $O(k \cdot n \cdot d + n \cdot d^2)$ 。然而，即使 $k=n$ ，可分离卷积的复杂性也等于自注意力层和逐点前馈层的组合，即我们模型采用的方法。

一个附加好处是，自注意力可以产生更多可解释的模型。我们可以对模型中的注意力分布进行检查，相关展示和讨论例子见附录。单个注意力头不仅可以清楚地学习并执行不同的任务，而且多个注意力头似乎表现出与句子的句法和[语义结构](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%AF%AD%E4%B9%89%E7%BB%93%E6%9E%84&zhida_source=entity)相关的行为。

5\. 训练
------

本节介绍我们模型的[训练方法](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95&zhida_source=entity)。

### 5.1 训练数据和批处理

我们使用标准 WMT 2014 英语-德语数据集进行训练，该数据集包含约 450 万对句子。句子编码采用 byte-pair 编码（byte-pair encoding）\[3\] ，源语句和目标语句共享约 37000 个 tokens 的词汇表。对于英语-法语翻译，我们使用了更大的 WMT 2014 英语-法语数据集，其由 3600 万个句子组成，并将 tokens 拆分为 32000 个 word-piece 词汇表 \[38\]。序列长度大体相近的句子分入同一批。每个训练批次（batch）包含一组句子对，其中包含大约 25000 个源 tokens 和 25000 个目标 tokens。

### 5.2 硬件和时间调度

我们在一台配备 8 个 NVIDIA P100 GPU 的机器上训练模型。我们使用论文中描述的超参数作为基础模型（base model），每个训练步骤大约需要 0.4 秒。我们对基础模型进行了总计 100,000 步即 12 小时的训练。对于大模型（big model）（见表 3 的最下列），单步时间为 1.0 秒。大模型进行了 300,000 步（3.5 天）的训练。

### 5.3 优化器

我们使用 [Adam 优化器](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=Adam+%E4%BC%98%E5%8C%96%E5%99%A8&zhida_source=entity) \[20\]，参数为 $\beta_1=0.9$ 、 $\beta_2=0.98$ 、 $\epsilon=10^{-9}$ 。在训练过程中，我们根据下述公式改变学习率：

$lrate = d_{\text{model}}^{-0.5} \cdot \min({step\_num}^{-0.5}, {step\_num} \cdot {warmup\_steps}^{-1.5})\\$这对应于第一次 $warmup\_steps$ 训练步骤中线性地增加学习速率，随之将其与步骤数的[平方根](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%B9%B3%E6%96%B9%E6%A0%B9&zhida_source=entity)成比例地减小。我们使用 $warmup\_steps=4000$ 。

### 5.4 正则化

训练期间，我们使用了三种类型的正则化（Regularization）：

**残差丢弃（Residual Dropout）**：我们将 dropout \[33\] 应用于每个子层的输出，随即将其加到子层输入，并进行归一化。此外，在编码器和解码器栈中，我们将 dropout 应用于嵌入和位置编码的加和。对于基础模型，比例为 $P_{drop}=0.1$ 。

**标签平滑（Label Smoothing）**：在训练期间，我们采用了值为 $\epsilon_{ls}=0.1$ \[36\]的标签平滑。这会影响模型的困惑度（perplexity），因为模型会变得更加不确定，但会提高准确性和 BLEU 分数。

6\. 结果
------

### 6.1 机器翻译

在 WMT 2014 英德翻译任务中，大 Transformer 模型（表 2 中的 Transformer (big)）比之前报道的最佳模型（包括集成）的性能高出超过 2.0 的 BLEU 分数 ，达到了新的 sota 的 BLEU 分数 28.4。该模型的配置列在 Table 3 的最后一行中。模型在 8 个 P100 GPU 上进行训练了 3.5 天。甚至我们的基础模型也超越了之前发布的所有模型和集成模型（ensembles），而训练成本只是这些竞争模型的一小部分。

在 WMT 2014 英法翻译任务中，我们的大模型获得了 41.0 的 BLEU 分数，优于之前发布的所有单个模型，且训练成本不到先前 sota 模型的 1/4 。针对英译法训练的（大） Transformer 模型使用 dropout 比例为 $P_{drop}=0.1$ ，而非 0.3。

对于基础模型，我们使用的单个模型来自最后 5 个 checkpoints 的均值，这些 checkpoints 每十分钟保存一次。对于大模型，我们对最后 20 个 checkpoints 进行了平均。我们使用了束搜索（beam search），束宽（beam size）为 4，长度惩罚 $α = 0.6$ \[38\]。这些超参数是在开发集（development set）上进行实验后选择的。推理期间的最大输出长度设为输入长度+50，但尽可能提前终止 \[38\]。

表 2 对我们的结果进行了总结，并就的翻译质量和训练成本将我们的模型与文献中其他模型架构进行了比较。我们将训练时间、使用的 GPU 数量以及每个 GPU 持续单精度浮点能力的估计相乘，用来估计用于训练模型的[浮点运算](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E6%B5%AE%E7%82%B9%E8%BF%90%E7%AE%97&zhida_source=entity)数量【注：对于 K80、K40、M40 和 P100 ，我们使用的 TFLOPS 值为分别 2.8、3.7、6.0 和 9.5 。】

![](https://pic3.zhimg.com/v2-be706d68d1ded9e638b6eacf0bb4077e_1440w.jpg)

表 2：在英译德和英译法 newstest2014 测试中，Transformer 比之前最先进的模型取得了更好的 BLEU 分数，而训练成本只是其一小部分。

### 6.2 模型变体

为了估计 Transformer 不同组件的重要性，我们以不同的方式对基本模型进行了修改，并观测了开发集 newstest2013 上英译德性能的变化。我们使用了上一节中描述的束搜索，但没有平均 checkpoints。这些结果见表 3。

在表 3 的 (A) 行中，我们改变了注意力头的数量以及注意力 keys 和 values 维度，但保持计算量不变，如第 3.2.2 节所述。单头注意力比最佳设置差 0.9 BLEU，但头数过多质量也会下降。

在表 3 的 (B) 行中，我们观察到：减少注意力 keys 的 $d_k$ 大小会影响模型质量。这表明确定兼容性并不容易，并且比点积更复杂的兼容函数可能是有益的。我们在 (C) 和 (D) 行中进一步观察到，一如预期，模型越大越好，并且 dropout 对于避免[过拟合](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%BF%87%E6%8B%9F%E5%90%88&zhida_source=entity)非常有帮助。在 (E) 行中，我们用可学习的位置嵌入替换正弦位置编码 \[9\]，并观察到其结果与基本模型几乎相同。

![](https://pic3.zhimg.com/v2-946679195afc8a8e5456348a560b489c_1440w.jpg)

表 3：Transformer 架构的变体。未列出的值与基础模型的值相同。所有指标均来自英译德开发集 newstest2013。根据我们的 byte-pair 编码，列出的困惑度是每个单词的困惑度，不应与每个单词的困惑度进行比较。

### 6.3 英语成分句法分析

为了评估 Transformer 是否可以泛化到其他任务，我们对英语成分句法分析（Constituency Parsing）任务进行了实验。该任务有特殊挑战：输出受很强的结构约束，并且明显长于输入。此外，RNN 序列到序列模型尚未能够在[小数据](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%B0%8F%E6%95%B0%E6%8D%AE&zhida_source=entity)情况下获得最 sota 的结果 \[37\]。

我们在 Penn Treebank 数据集 \[25\] 的《[华尔街日报](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%8D%8E%E5%B0%94%E8%A1%97%E6%97%A5%E6%8A%A5&zhida_source=entity)（Wall Street Journal, WSJ）》部分训练了一个 $d_{model} = 1024$ 的 4 层 Transformer，大约 40K 训练句子。我们还在半监督环境中对其进行了训练，使用了更大的高置信度（high-confidence）以及 BerkleyParser [语料库](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%AF%AD%E6%96%99%E5%BA%93&zhida_source=entity)，其中包含大约 1700 万个句子 \[37\]。我们在仅用 WSJ 的设置下使用了 16K 个 tokens 的词汇表，在半监督设置下使用了 32K 个 tokens 的词汇表。

我们只在开发集的Section 22 上进行了少量的实验来选择 dropout、注意力和残差（第 5.4 节）、学习率和束宽，所有其他参数与英译德的基础模型保持相同。在推理过程中，我们将最大输出长度增加到输入长度+300。对于仅 WSJ 和半监督设置，我们使用的束宽为 21， $α = 0.3$ 。

表 4 中的结果表明，尽管缺乏针对特定任务的调整，但我们的模型表现非常好，其结果比之前报告的所有模型都要好，仅有 循环神经网络语法（Recurrent Neural Network Grammar）\[8\] 除外。

与 RNN 序列到序列模型 \[37\] 相比，即使仅在 WSJ 40K 句子训练集上进行训练，Transformer 的性能也优于 BerkeleyParser \[29\]。

7\. 结论
------

这项工作中，我们提出了 Transformer，这是首个完全基于注意力的序列转换模型，用多头自注意力取代了编码器-解码器架构中最常用的循环层。

对于翻译任务，Transformer 的训练速度明显快于基于循环层或卷积层的架构。在 WMT 2014 英移德和 WMT 2014 英译法任务中，我们都达到了新的 sota 水平。在前一项任务中，我们最好的模型甚至优于所有先前报告的集成模型。

我们对基于注意力的模型之未来感到格外兴奋，并计划着手将其应用于其他任务。我们计划将 Transformer 扩展到文本以外的输入和输出模式的问题，并研究局部的、受限的注意力机制，以有效地处理图像、音频和视频等大型输入和输出。让生成具有更少的顺序性，是我们的另一个研究目标。

我们用于训练和[评估模型](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E8%AF%84%E4%BC%B0%E6%A8%A1%E5%9E%8B&zhida_source=entity)的代码见 [https://github.com/tensorflow/tensor2tensor](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/tensor2tensor)。

**致谢** 我们感谢 Nal Kalchbrenner 和 Stephan Gouws 令我们获益良多的评论、更正和启发。

参考文献略

附录：注意力[可视化](https://zhida.zhihu.com/search?content_id=239687332&content_type=Article&match_order=1&q=%E5%8F%AF%E8%A7%86%E5%8C%96&zhida_source=entity)
-----------------------------------------------------------------------------------------------------------------------------------------------------

![](https://pic4.zhimg.com/v2-c71f79f83ffa98f8e2a920bd5770f507_1440w.jpg)

图 3：第 5 层（共 6 层）中，编码器自注意力中的长距离依赖关系的注意力机制示例。许多注意力头关注动词“making”的长距离依赖，完成短语“making...more difficult”。这里仅显示“制作”一词的注意力。不同的颜色代表不同的头。

![](https://pic1.zhimg.com/v2-8681ccb8e2d9782656114a8b8cae8236_1440w.jpg)

图 4：第 5 层（共 6 层）中两个注意力头显然参与了回指解析（anaphora resolution）。右：head 5 的完整注意力。左：head 5 和 6 的注意力与单词“its”隔离。请注意，该单词的注意力非常强。

![](https://pic2.zhimg.com/v2-7ca3533054f7986979e91fbf460bc6a1_1440w.jpg)

图 5：许多注意力头表现出的行为似乎与句子的结构有关。我们在上面给出了两个这样的例子，来自第 5 层（共 6 层）编码器自注意力的两个不同的头。这些头显然学会了执行不同的任务。