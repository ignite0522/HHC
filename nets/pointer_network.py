import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np


# === 编码器类 ===
class Encoder(nn.Module):
    """将输入序列（图）映射到隐藏向量，使用 LSTM 编码。
    - input_dim: 输入维度
    - hidden_dim: 隐藏状态维度
    """

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)  # LSTM 层
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)  # 可训练的初始隐藏状态

    def forward(self, x, hidden):
        """前向传播，编码输入序列。
        - x: 输入序列 [sourceL, batch_size, input_dim]
        - hidden: 初始隐藏状态 (hx, cx)
        - 返回: 输出序列和最终隐藏状态
        """
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """初始化可训练的隐藏状态。
        - hidden_dim: 隐藏维度
        - 返回: 初始隐藏状态 (hx, cx)
        """
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)
        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx


# === 注意力模块 ===
class Attention(nn.Module):
    """通用的注意力模块，用于解码器。
    - dim: 隐藏维度
    - use_tanh: 是否使用 tanh 激活
    - C: tanh 探索参数
    """

    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)  # 查询投影
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)  # 参考投影
        self.C = C  # tanh 裁剪系数
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(dim))  # 注意力权重
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """计算注意力分数。
        - query: 解码器当前隐藏状态 [batch_size, dim]
        - ref: 编码器输出 [sourceL, batch_size, hidden_dim]
        - 返回: 投影后的参考向量和注意力 logits
        """
        ref = ref.permute(1, 2, 0)  # [batch_size, hidden_dim, sourceL]
        q = self.project_query(query).unsqueeze(2)  # [batch_size, dim, 1]
        e = self.project_ref(ref)  # [batch_size, hidden_dim, sourceL]
        expanded_q = q.repeat(1, 1, e.size(2))  # [batch_size, dim, sourceL]
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)  # [batch_size, 1, dim]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)  # [batch_size, sourceL]
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


# === 解码器类 ===
class Decoder(nn.Module):
    """基于 LSTM 和注意力的解码器，生成动作序列。
    - embedding_dim: 嵌入维度
    - hidden_dim: 隐藏维度
    - tanh_exploration: tanh 裁剪系数
    - use_tanh: 是否使用 tanh
    - n_glimpses: 注意力瞥视次数
    - mask_glimpses/logits: 是否掩码瞥视/logits
    """

    def __init__(self, embedding_dim, hidden_dim, tanh_exploration, use_tanh, n_glimpses=1, mask_glimpses=True,
                 mask_logits=True):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = None  # 解码类型（greedy/sampling）

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)  # LSTM 单元
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)  # 指针注意力
        self.glimpse = Attention(hidden_dim, use_tanh=False)  # 瞥视注意力
        self.sm = nn.Softmax(dim=1)  # Softmax 层

    def update_mask(self, mask, selected):
        """更新掩码，标记已选择的节点。
        - mask: 当前掩码 [batch_size, sourceL]
        - selected: 选择的节点索引 [batch_size]
        - 返回: 更新后的掩码
        """
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):
        """单步解码，计算 logits 和概率。
        - x: 当前输入 [batch_size, embedding_dim]
        - h_in: 当前隐藏状态 (hy, cy)
        - prev_mask: 前一步掩码
        - prev_idxs: 前一步选择索引
        - step: 当前步数
        - context: 编码器输出 [sourceL, batch_size, hidden_dim]
        - 返回: 下一隐藏状态、对数概率、概率、更新后的掩码
        """
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)
        log_p = torch.log_softmax(logits, dim=1)  # 计算对数概率
        probs = log_p.exp()  # 计算概率
        if not self.mask_logits:
            probs[logit_mask] = 0.  # 屏蔽不可行节点
        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):
        """计算注意力 logits。
        - 参数同 recurrence
        - 返回: logits 和下一隐藏状态
        """
        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses
        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)  # LSTM 前向
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):  # 多轮瞥视
            ref, logits = self.glimpse(g_l, context)
            if mask_glimpses:
                logits[logit_mask] = -np.inf  # 屏蔽瞥视
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)  # 更新瞥视状态
        _, logits = self.pointer(g_l, context)  # 计算指针 logits
        if mask_logits:
            logits[logit_mask] = -np.inf  # 屏蔽 logits
        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, eval_tours=None):
        """解码器前向传播，生成路径序列。
        - decoder_input: 初始输入 [batch_size, embedding_dim]
        - embedded_inputs: 嵌入输入 [sourceL, batch_size, embedding_dim]
        - hidden: 初始隐藏状态 (hy, cy)
        - context: 编码器输出 [sourceL, batch_size, hidden_dim]
        - eval_tours: 可选的评估路径
        - 返回: (对数概率, 路径索引), 最终隐藏状态
        """
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = Variable(
            embedded_inputs.data.new().byte().new(embedded_inputs.size(1), embedded_inputs.size(0)).zero_(),
            requires_grad=False
        )  # 初始化掩码

        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            idxs = self.decode(probs, mask) if eval_tours is None else eval_tours[:, i]  # 选择下一节点
            idxs = idxs.detach()  # 断开梯度
            decoder_input = torch.gather(embedded_inputs, 0,
                                         idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size,
                                                                                         *embedded_inputs.size()[
                                                                                          2:])).squeeze(0)  # 获取下一输入
            outputs.append(log_p)
            selections.append(idxs)
        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden

    def decode(self, probs, mask):
        """根据概率选择下一节点。
        - probs: 概率分布 [batch_size, sourceL]
        - mask: 掩码 [batch_size, sourceL]
        - 返回: 选择的节点索引 [batch_size]
        """
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)  # 贪婪选择
            assert not mask.gather(1, idxs.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)  # 采样选择
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"
        return idxs


# === Critic 网络 ===
class CriticNetworkLSTM(nn.Module):
    """用于 REINFORCE 的基线网络，预测状态价值。
    - embedding_dim: 嵌入维度
    - hidden_dim: 隐藏维度
    - n_process_block_iters: 处理块迭代次数
    - tanh_exploration: tanh 裁剪系数
    - use_tanh: 是否使用 tanh
    """

    def __init__(self, embedding_dim, hidden_dim, n_process_block_iters, tanh_exploration, use_tanh):
        super(CriticNetworkLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters
        self.encoder = Encoder(embedding_dim, hidden_dim)  # 编码器
        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)  # 注意力处理块
        self.sm = nn.Softmax(dim=1)  # Softmax 层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )  # 输出单一价值

    def forward(self, inputs):
        """前向传播，预测状态价值。
        - inputs: 嵌入输入 [embedding_dim, batch_size, sourceL]
        - 返回: 预测价值 [batch_size, 1]
        """
        inputs = inputs.transpose(0, 1).contiguous()
        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))  # 编码
        process_block_state = enc_h_t[-1]  # 获取最终隐藏状态
        for i in range(self.n_process_block_iters):  # 注意力处理
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        out = self.decoder(process_block_state)  # 输出价值
        return out


# === 指针网络 ===
class PointerNetwork(nn.Module):
    """指针网络，解决 TSP 问题。
    - embedding_dim: 嵌入维度
    - hidden_dim: 隐藏维度
    - problem: 问题定义（仅支持 TSP）
    - tanh_clipping: tanh 裁剪系数
    - mask_inner/logits: 是否掩码瞥视/logits
    """

    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=None, tanh_clipping=10., mask_inner=True,
                 mask_logits=True, normalization=None, **kwargs):
        super(PointerNetwork, self).__init__()
        self.problem = problem
        assert problem.NAME == "tsp", "Pointer Network only supported for TSP"
        self.input_dim = 2  # TSP 输入维度 (x, y)

        self.encoder = Encoder(embedding_dim, hidden_dim)  # 编码器
        self.decoder = Decoder(embedding_dim, hidden_dim, tanh_clipping, tanh_clipping > 0, n_glimpses=1,
                               mask_glimpses=mask_inner, mask_logits=mask_logits)  # 解码器
        std = 1. / math.sqrt(embedding_dim)
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))  # 可训练的初始解码输入
        self.decoder_in_0.data.uniform_(-std, std)
        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embedding_dim))  # 嵌入矩阵
        self.embedding.data.uniform_(-std, std)

    def set_decode_type(self, decode_type):
        """设置解码类型（greedy/sampling）"""
        self.decoder.decode_type = decode_type

    def forward(self, inputs, eval_tours=None, return_pi=False):
        """前向传播，生成路径和对数似然。
        - inputs: 输入节点 [batch_size, graph_size, input_dim]
        - eval_tours: 可选的评估路径
        - return_pi: 是否返回路径
        - 返回: 成本、对数似然、[路径]
        """
        batch_size, graph_size, input_dim = inputs.size()
        embedded_inputs = torch.mm(inputs.transpose(0, 1).contiguous().view(-1, input_dim), self.embedding).view(
            graph_size, batch_size, -1)  # 嵌入输入
        _log_p, pi = self._inner(embedded_inputs, eval_tours)  # 解码生成路径
        cost, mask = self.problem.get_costs(inputs, pi)  # 计算路径成本
        ll = self._calc_log_likelihood(_log_p, pi, mask)  # 计算对数似然
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):
        """计算路径的对数似然。
        - _log_p: 对数概率 [batch_size, steps, graph_size]
        - a: 路径 [batch_size, steps]
        - mask: 掩码
        - 返回: 对数似然 [batch_size]
        """
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            log_p[mask] = 0  # 屏蔽无关动作
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        return log_p.sum(1)

    def _inner(self, inputs, eval_tours=None):
        """内部解码过程。
        - inputs: 嵌入输入 [sourceL, batch_size, embedding_dim]
        - eval_tours: 可选的评估路径
        - 返回: 对数概率和路径索引
        """
        encoder_hx = encoder_cx = Variable(
            torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, out=inputs.data.new()), requires_grad=False)
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))  # 编码
        dec_init_state = (enc_h_t[-1], enc_c_t[-1])  # 初始解码状态
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)  # 初始解码输入
        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input, inputs, dec_init_state, enc_h,
                                                                 eval_tours)  # 解码
        return pointer_probs, input_idxs