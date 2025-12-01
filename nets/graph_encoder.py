import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F


# SkipConnection 类：实现残差连接（skip connection），将输入与模块输出相加
class SkipConnection(nn.Module):

    # 初始化方法
    # 参数：
    #   module: 要包裹的模块（如 MultiHeadAttention 或前馈网络）
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module  # 保存传入的模块

    # 前向传播
    # 参数：
    #   input: 输入张量
    # 返回：
    #   输入与模块输出的和，实现残差连接
    def forward(self, input):
        return input + self.module(input)  # 输入 + 模块输出


# MultiHeadAttention 类：实现多头自注意力机制（Multi-Head Self-Attention）
class MultiHeadAttention(nn.Module):
    # 初始化方法
    # 参数：
    #   n_heads: 注意力头数
    #   input_dim: 输入维度
    #   embed_dim: 输出嵌入维度
    #   val_dim: 值（value）的维度（默认 embed_dim / n_heads）
    #   key_dim: 键（key）的维度（默认等于 val_dim）
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        # 设置值维度（若未指定，则为 embed_dim / n_heads）
        if val_dim is None:
            val_dim = embed_dim // n_heads
        # 设置键维度（若未指定，则等于 val_dim）
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads  # 注意力头数
        self.input_dim = input_dim  # 输入维度
        self.embed_dim = embed_dim  # 输出嵌入维度
        self.val_dim = val_dim  # 值维度
        self.key_dim = key_dim  # 键维度

        # 缩放因子，用于缩放点积注意力（参考 "Attention is All You Need"）
        self.norm_factor = 1 / math.sqrt(key_dim)  # 1 / sqrt(key_dim)

        # 查询（query）、键（key）、值（value）的权重矩阵
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))  # 查询权重
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))  # 键权重
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))  # 值权重

        # 输出投影权重
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        # 初始化参数
        self.init_parameters()

    # 初始化权重参数
    # 使用均匀分布初始化权重，范围为 [-stdv, stdv]，stdv = 1 / sqrt(维度)
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # 前向传播
    # 参数：
    #   q: 查询张量，形状 (batch_size, n_query, input_dim)
    #   h: 数据张量，形状 (batch_size, graph_size, input_dim)，若为 None 则使用 q（自注意力）
    #   mask: 掩码张量，形状 (batch_size, n_query, graph_size)，1 表示不可关注
    # 返回：
    #   输出张量，形状 (batch_size, n_query, embed_dim)
    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        # 如果 h 为 None，执行自注意力（h = q）
        if h is None:
            h = q  # compute self-attention

        # h 的形状为 (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)  # 查询数量
        assert q.size(0) == batch_size  # 确保批次大小一致
        assert q.size(2) == input_dim  # 确保输入维度一致
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        # 将 h 和 q 展平为 (batch_size * graph_size, input_dim) 和 (batch_size * n_query, input_dim)
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # 输出形状
        shp = (self.n_heads, batch_size, graph_size, -1)  # 键和值的形状
        shp_q = (self.n_heads, batch_size, n_query, -1)  # 查询的形状

        # 计算查询 Q = q * W_query，形状 (n_heads, batch_size, n_query, key_dim)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # 计算键 K = h * W_key，形状 (n_heads, batch_size, graph_size, key_dim)
        K = torch.matmul(hflat, self.W_key).view(shp)
        # 计算值 V = h * W_val，形状 (n_heads, batch_size, graph_size, val_dim)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # 计算兼容性（compatibility）= Q * K^T / sqrt(key_dim)
        # 形状 (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # 应用掩码，屏蔽不可关注的节点
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf  # 将不可关注的位置设为负无穷

        # 计算注意力权重，使用 softmax
        attn = torch.softmax(compatibility, dim=-1)

        # 处理掩码导致的 NaN（若节点无邻居，softmax 返回 NaN）
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0  # 将不可关注位置的权重设为 0
            attn = attnc

        # 计算注意力输出 heads = attn * V，形状 (n_heads, batch_size, n_query, val_dim)
        heads = torch.matmul(attn, V)

        # 投影多头输出到最终嵌入
        # heads 展平为 (batch_size * n_query, n_heads * val_dim)
        # 投影后形状为 (batch_size, n_query, embed_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out  # 返回注意力输出


# Normalization 类：实现归一化（批归一化或实例归一化）
class Normalization(nn.Module):

    # 初始化方法
    # 参数：
    #   embed_dim: 嵌入维度
    #   normalization: 归一化类型（'batch' 或 'instance'）
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        # 根据归一化类型选择归一化类
        normalizer_class = {
            'batch': nn.BatchNorm1d,  # 批归一化
            'instance': nn.InstanceNorm1d  # 实例归一化
        }.get(normalization, None)

        # 创建归一化层，启用仿射变换
        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    # 初始化归一化参数（未使用）
    # 使用均匀分布初始化权重，范围为 [-stdv, stdv]
    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # 前向传播
    # 参数：
    #   input: 输入张量，形状 (batch_size, graph_size, embed_dim)
    # 返回：
    #   归一化后的张量，形状不变
    def forward(self, input):

        # 批归一化：展平为 (batch_size * graph_size, embed_dim) 进行归一化
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        # 实例归一化：转置为 (batch_size, embed_dim, graph_size) 进行归一化
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input  # 无归一化，直接返回输入


# MultiHeadAttentionLayer 类：实现单层多头注意力，包含注意力、归一化和前馈网络
class MultiHeadAttentionLayer(nn.Sequential):

    # 初始化方法
    # 参数：
    #   n_heads: 注意力头数
    #   embed_dim: 嵌入维度
    #   feed_forward_hidden: 前馈网络隐藏层维度（默认 512）
    #   normalization: 归一化类型（'batch' 或 'instance'）
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch'
    ):
        # 使用 nn.Sequential 按顺序组合子模块
        super(MultiHeadAttentionLayer, self).__init__(
            # 第一个残差连接：多头注意力
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            # 第一个归一化层
            Normalization(embed_dim, normalization),
            # 第二个残差连接：前馈网络
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),  # 线性层
                    nn.ReLU(),  # ReLU 激活
                    nn.Linear(feed_forward_hidden, embed_dim)  # 线性层
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            # 第二个归一化层
            Normalization(embed_dim, normalization)
        )


# GraphAttentionEncoder 类：实现图注意力编码器，用于将输入节点特征编码为嵌入
class GraphAttentionEncoder(nn.Module):
    # 初始化方法
    # 参数：
    #   n_heads: 注意力头数
    #   embed_dim: 嵌入维度
    #   n_layers: 编码器层数
    #   node_dim: 输入节点特征维度（若为 None，则输入已是嵌入）
    #   normalization: 归一化类型（'batch' 或 'instance'）
    #   feed_forward_hidden: 前馈网络隐藏层维度
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # 输入特征到嵌入的线性层（若 node_dim 非 None）
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        # 多层注意力层，包含 n_layers 个 MultiHeadAttentionLayer
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    # 前向传播
    # 参数：
    #   x: 输入张量，形状 (batch_size, graph_size, node_dim) 或 (batch_size, graph_size, embed_dim)
    #   mask: 掩码（当前不支持）
    # 返回：
    #   元组：
    #     - 节点嵌入，形状 (batch_size, graph_size, embed_dim)
    #     - 图嵌入（节点嵌入的均值），形状 (batch_size, embed_dim)
    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"  # 当前不支持掩码

        # 若有初始嵌入层，将输入特征映射到嵌入空间
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        # 通过多层注意力编码
        h = self.layers(h)

        # 返回节点嵌入和图嵌入
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )