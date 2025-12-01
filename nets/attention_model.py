import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

import pickle

# === 辅助函数：设置解码类型 ===
def set_decode_type(model, decode_type):
    """设置模型的解码类型（'greedy' 或 'sampling'）。
    - model: AttentionModel 实例或 DataParallel 包裹的模型
    - decode_type: 解码类型
    """
    if isinstance(model, DataParallel):
        model = model.module  # 解包 DataParallel
    model.set_decode_type(decode_type)

# === 固定上下文类 ===
class AttentionModelFixed(NamedTuple):
    """
    存储解码过程中固定的上下文信息，避免重复计算，提高效率。
    - node_embeddings: 节点嵌入 [batch_size, graph_size+1, embedding_dim]
    - context_node_projected: 投影后的全局上下文 [batch_size, 1, embedding_dim]
    - fleet_embedding: 车队嵌入（仅 AGH 使用）[batch_size, 1, embedding_dim]
    - glimpse_key: 注意力键 [n_heads, batch_size, graph_size+1, key_size]
    - glimpse_val: 注意力值 [n_heads, batch_size, graph_size+1, val_size]
    - logit_key: logits 键 [batch_size, graph_size+1, embedding_dim]
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    fleet_embedding: torch.Tensor  # only use for AGH problem
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        """支持对所有字段同时索引（如切片），返回新的 AttentionModelFixed 实例"""
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            fleet_embedding=self.fleet_embedding[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

# === 主模型类 ===
class AttentionModel(nn.Module):
    """
    基于注意力机制的神经网络模型，解决 AGH（机场地面处理）等 VRP 问题。
    - 编码器：将输入特征（需求、时间窗口等）转为节点嵌入。
    - 解码器：通过注意力机制选择节点，构建满足约束的路径。
    - 训练：使用强化学习（REINFORCE）优化策略。
    """
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=3,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 wo_time=False,
                 rnn_time=False):
        super(AttentionModel, self).__init__()

        # === 模型参数 ===
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.hidden_dim = hidden_dim  # 隐藏层维度（未直接使用）
        self.n_encode_layers = n_encode_layers  # 编码器层数
        self.decode_type = None  # 解码类型（'greedy' 或 'sampling'）
        self.temp = 1.0  # 温度参数，控制 softmax 分布
        self.allow_partial = problem.NAME == 'sdvrp'  # 是否允许部分配送
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'  # 是否为 VRP 问题
        self.is_orienteering = problem.NAME == 'op'  # 是否为 Orienteering 问题
        self.is_pctsp = problem.NAME == 'pctsp'  # 是否为 PCTSP 问题
        self.is_agh = problem.NAME == 'agh'  # 是否为 AGH 问题
        self.wo_time = wo_time  # 是否忽略时间窗口
        self.rnn_time = rnn_time  # 是否使用 RNN 嵌入时间窗口

        print("调用AttentionModel")

        # === AGH 特定配置 ===
        if self.is_agh:
            with open('problems/agh/fleet_info.pkl', 'rb') as f:
                # 加载车队信息：优先级、操作时长、后续操作时间
                # {'order': [1, 2, 4, 8, 3, 5, 7, 9, 6, 10],  # 车队求解顺序
                # 'precedence': {1: 0, 2: 1, ...},  # 车队优先级
                # 'duration': {1: [0.0, 0.0, 0.0], ...},  # 操作时长
                # 'next_duration': {4: [0.0, 0.0, 0.0], ...}}  # 后续操作最短时间
                self.fleet_info = pickle.load(f)
            with open('problems/agh/distance.pkl', 'rb') as f:
                # 加载距离矩阵：节点间距离，这里面包含了车库
                # {(0, 0): 0, (0, 1): 439.06, ... (0, 91): xxx, ...}
                self.distance = pickle.load(f)
            self.distance = torch.tensor(list(self.distance.values()))  # 转换为张量

        self.tanh_clipping = tanh_clipping  # tanh 裁剪参数，限制 logits 范围

        self.mask_inner = mask_inner  # 是否在注意力兼容性计算中应用掩码
        self.mask_logits = mask_logits  # 是否在 logits 计算中应用掩码

        self.problem = problem  # 问题对象，定义约束和成本计算
        self.n_heads = n_heads  # 多头注意力头数
        self.checkpoint_encoder = checkpoint_encoder  # 是否对编码器使用检查点
        self.shrink_size = shrink_size  # 批次收缩阈值

        # === 上下文维度 ===
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # 步骤上下文：上一个节点嵌入 + 剩余容量/长度/奖励
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # 节点特征：x, y, 预期奖励, 惩罚
            else:
                node_dim = 3  # 节点特征：x, y, 需求/奖励

            # 车库节点嵌入
            self.init_embed_depot = nn.Linear(2, embedding_dim)

            if self.is_vrp and self.allow_partial:  # 支持部分配送
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        elif self.is_agh:
            # AGH：6 个车队的嵌入
            self.fleets_embedding = nn.Embedding(6, embedding_dim)     #######################
            # 步骤上下文：当前节点嵌入 + 当前空闲时间
            step_context_dim = embedding_dim + 1
            if self.wo_time or self.rnn_time:
                node_dim = 1  # 仅需求
            else:
                node_dim = 2  #  时间窗口左右边界
            # 91 个登机口 + 1 个车库的嵌入
            self.loc_embedding = nn.Embedding(92, embedding_dim)    #####################
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # 上下文：首末节点嵌入
            node_dim = 2  # 节点特征：x, y

            # 首动作的占位符
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # 随机初始化

        # === 嵌入层 ===
        self.init_embed = nn.Linear(node_dim, embedding_dim)  # 特征嵌入

        # === 编码器 ===
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # === 时间窗口嵌入（AGH 特有） ===
        if self.rnn_time:
            self.time_embed = nn.Linear(2, embedding_dim)  # 时间窗口嵌入
            self.rnn = nn.LSTMCell(embedding_dim, embedding_dim)  # LSTM 单元
            self.pre_tw = None  # 初始化隐藏状态

        # === 投影层 ===
        # 节点嵌入投影到键、值和 logits
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)  # 多头输出投影

    def set_decode_type(self, decode_type, temp=None):
        """设置解码类型和温度参数。
        - decode_type: 'greedy' 或 'sampling'
        - temp: softmax 温度（可选）
        """
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        前向传播：生成路径、成本和对数似然。
        - input: 字典，包含以下字段：
            'loc': 登机口索引 [batch_size, graph_size]
            'distance': 节点间距离 [batch_size, len(distance)]
            'duration': 操作时长 [batch_size, graph_size]
            'tw_right',
            'tw_left': 时间窗口边界 [batch_size, graph_size+1]
            'fleet': 车队索引 [batch_size, 1]
        - return_pi: 是否返回路径序列（因 DataParallel 可能不兼容）
        - 输出:
            - AGH: (cost, ll, serve_time, [pi]) 或 (cost, ll, serve_time)
            - 其他: (cost, ll, [pi])
        """
        if self.checkpoint_encoder and self.training:
            # 使用检查点减少内存占用
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            # 正常编码：生成节点嵌入
            embeddings, _ = self.embedder(self._init_embed(input))  # [batch_size, graph_size+1, embedding_dim]，embeddings包含位置和时间信息

        # 解码：生成对数概率、路径和服务时间
        _log_p, pi, serve_time = self._inner(input, embeddings)# pi是路径（50个节点）
        # print(f"路径：{pi[:5]}")
        # 计算成本和掩码
        cost, mask = self.problem.get_costs(input, pi)

        # 计算对数似然
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if self.is_agh:
            if return_pi:
                return cost, ll, serve_time, pi
            else:
                return cost, ll, serve_time

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def beam_search(self, *args, **kwargs):
        """调用问题特定的束搜索方法"""
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input, fleet=None):
        """预计算固定上下文，用于束搜索等场景。
        - input: 输入数据
        - fleet: 车队索引（AGH 特有）
        - 输出: CachedLookup 包装的 AttentionModelFixed
        """
        embeddings, _ = self.embedder(self._init_embed(input))
        return CachedLookup(self._precompute(embeddings, fleet))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        """
        提出束搜索的扩展候选。
        - beam: 当前束
        - fixed: 固定上下文
        - expand_size: 扩展数量（top-k）
        - normalize: 是否标准化对数概率
        - max_calc_batch_size: 最大批次大小
        - 输出: (父节点索引, 动作, 分数)
        """
        # 计算 top-k 概率和索引
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # 计算扩展分数
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # 展平动作和分数
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # 过滤不可行扩展

        # 计算父节点索引
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # 过滤不可行扩展
        feas_ind_2d = torch.nonzero(flat_feas)
        if len(feas_ind_2d) == 0:
            return None, None, None
        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        """
        计算路径的对数似然。
        - _log_p: 对数概率 [batch_size, steps, graph_size+1]
        - a: 路径 [batch_size, steps]
        - mask: 掩码
        - 输出: 对数似然 [batch_size]
        """
        # 提取对应动作的对数概率
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # 屏蔽无关动作
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # 计算总对数似然
        return log_p.sum(1)

    def _init_embed(self, input):
        """
        生成初始节点嵌入。
        - input: 输入数据字典
        - 输出: 节点嵌入 [batch_size, graph_size+1, embedding_dim]
        """
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # VRP/Orienteering/PCTSP：嵌入车库和节点特征
            if self.is_vrp:
                features = ('demand',)
            elif self.is_orienteering:
                features = ('prize',)
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        elif self.is_agh:
            loc_input = torch.cat((torch.zeros_like(input['loc'][:, :1]), input['loc']), dim=1)
            init_embed = self.loc_embedding(loc_input)
            mask = (input['tw_right'][:, 1:] > 0).float()  # 有效节点为1，忽略 tw_right <= 0 的节点
            mask = torch.cat((torch.ones(input['tw_right'].size(0), 1, device=mask.device), mask), dim=1)  # 包含仓库
            init_embed = init_embed * mask.unsqueeze(-1)  # 屏蔽无效节点的位置嵌入，逐元素相乘：有效节点保持原嵌入，无效节点嵌入置零

            if self.wo_time:
                fea_embed = torch.zeros_like(init_embed[:, 1:, :])
            elif self.rnn_time:
                batch_size, graph_size, embed_size = init_embed[:, 1:, :].size()
                if self.pre_tw is None:
                    self.pre_tw = (torch.zeros(batch_size * graph_size, embed_size).float().to(init_embed.device),
                                   torch.zeros(batch_size * graph_size, embed_size).float().to(init_embed.device))
                time_embed = self.time_embed(torch.cat((input['tw_left'][:, 1:, None] / 1440,
                                                        input['tw_right'][:, 1:, None] / 1440), -1).float())
                h_1, c_1 = self.rnn(time_embed.view(-1, embed_size), self.pre_tw)
                h_1 = h_1.view(batch_size, graph_size, embed_size) * mask[:, 1:].unsqueeze(-1)
                fea_embed = h_1
                h_1 = h_1.view(-1, embed_size)
                self.pre_tw = (h_1, c_1)
            else:
                fea_embed = self.init_embed(torch.cat((input['tw_left'][:, 1:, None] / 1440,
                                                       input['tw_right'][:, 1:, None] / 1440),
                                                      -1).float()) * mask[:, 1:].unsqueeze(-1)
            init_embed[:, 1:, :] = init_embed[:, 1:, :] + fea_embed # 位置节点和时间节点嵌入相加
            return init_embed
        # 这里移除了demand
        return self.init_embed(input)


    def _inner(self, input, embeddings):
        """解码过程：逐步选择节点，构建路径"""
        outputs, sequences = [], []
        state = self.problem.make_state(input) # 初始化state
        fixed = self._precompute(embeddings, input['fleet'])
        batch_size = state.ids.size(0)
        # print("调用inner")

        # print(f"[Inner] 初始化完成，团队: {input['fleet'][0].item()}, 批次大小: {batch_size}")

        i = 0

        while not (self.shrink_size is None and state.all_finished()):
            # [Inner] 处理批次收缩
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            # [Inner] 计算对数概率和掩码
            log_p, mask = self._get_log_p(fixed, state)
            # print(f"mask: {mask},mask.shape: {mask.shape}")

            # [Inner] 选择下一个节点
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            # [Inner] 更新状态
            state = state.update(selected)
            # print(f"tour: {state.tour[:4]},\n ")

            # [Inner] 恢复批次大小
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # [Inner] 收集输出
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1
        

        
        # print(f"[Inner] 解码循环结束，总步骤: {i}")
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state.serve_time

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        批量采样：生成多样化路径。
        - input: 输入数据
        - batch_rep, iter_rep: 重复次数
        - 输出: 路径和成本
        """
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], pi),
            (input, self.embedder(self._init_embed(input))[0]),
            batch_rep, iter_rep
        )

    # def _select_node(self, probs, mask):
    #     """
    #     选择下一个节点。
    #     - probs: 概率分布 [batch_size, graph_size+1]
    #     - mask: 掩码 [batch_size, graph_size+1]
    #     - 输出: 选择的节点索引 [batch_size]
    #     """
    #     assert (probs == probs).all(), "Probs should not contain any nans"
    #
    #     if self.decode_type == "greedy":
    #         _, selected = probs.max(1)  # 贪婪选择
    #         assert not mask.gather(1, selected.unsqueeze(
    #             -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
    #     elif self.decode_type == "sampling":
    #         selected = probs.multinomial(1).squeeze(1)  # 采样选择
    #         while mask.gather(1, selected.unsqueeze(-1)).data.any():
    #             print('Sampled bad values, resampling!')
    #             selected = probs.multinomial(1).squeeze(1)
    #     else:
    #         assert False, "Unknown decode type"
    #
    #     return selected

    # 改进
    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            masked_probs = probs.masked_fill(mask, -float('inf'))
            _, selected = masked_probs.max(1)

        elif self.decode_type == "sampling":
            # 方法：直接置零，让采样概率重新分布
            sampling_probs = probs.clone()
            sampling_probs[mask] = 0.0

            # 检查是否有有效概率
            if (sampling_probs.sum(dim=1) > 0).all():
                selected = sampling_probs.multinomial(1).squeeze(1)
            else:
                # 后备方案：使用原始的重采样逻辑
                selected = probs.multinomial(1).squeeze(1)
                while mask.gather(1, selected.unsqueeze(-1)).data.any():
                    selected = probs.multinomial(1).squeeze(1)

        return selected

    def _precompute(self, embeddings, fleet, num_steps=1):
        """
        预计算固定上下文。
        - embeddings: 节点嵌入
        - fleet: 车队索引（AGH 特有）
        - num_steps: 解码步数（默认 1）
        - 输出: AttentionModelFixed 实例
        """
        # 计算全局上下文
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # AGH：获取车队嵌入
        if self.is_agh:
            fleet_embedding = self.fleets_embedding(fleet)
        else:
            fleet_embedding = torch.zeros_like(fixed_context)

        # 投影节点嵌入到键、值和 logits
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # 转换为多头格式
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, fleet_embedding, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        """
        计算 top-k 节点选择概率。
        - fixed: 固定上下文
        - state: 当前状态
        - k: 返回 top-k（默认全部）
        - normalize: 是否标准化
        - 输出: (top-k 概率, 索引)
        """
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        """
        计算节点选择对数概率。
        - fixed: 固定上下文
        - state: 当前状态
        - normalize: 是否标准化为对数概率
        - 输出: (对数概率 [batch_size, 1, graph_size+1], 掩码)
        """
        # 计算查询
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state).float())
        # 这里包括固定上文，当前节点嵌入 + 当前空闲时间

        if self.is_agh:
            query = query + fixed.fleet_embedding

        # 获取键和值
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # 获取掩码
        mask = state.get_mask()
        # print(f"mask: {mask[:4].int()}, shape: {mask.shape}")

        # 计算 logits
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        生成每步上下文。
        - embeddings: 节点嵌入 [batch_size, graph_size+1, embedding_dim]
        - state: 当前状态
        - from_depot: 是否从车库开始（VRP 特有）
        - 输出: 上下文 [batch_size, num_steps, context_dim]
        """
        current_node = state.get_current_node()  # [batch_size, 1]
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # VRP：当前节点嵌入 + 剩余容量
            if from_depot:
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1))
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous().view(batch_size,
                                                           num_steps, 1).expand(batch_size, num_steps,
                                                                                embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1))
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_pctsp:
            # Orienteering/PCTSP：当前节点嵌入 + 剩余长度/奖励
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous().view(batch_size,
                                                       num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        elif self.is_agh:
            # AGH：当前节点嵌入  + 当前空闲时间
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    # 这里删掉 self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None],
                    state.cur_free_time[:, :, None] / 1440
                ),
                -1
            )
        else:  # TSP
            if num_steps == 1:
                if state.i.item() == 0:
                    # 首步：使用占位符
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    # 单步：首末节点嵌入
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2,
                                                                                       embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # 多步：首步占位符 + 当前/前节点嵌入
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        计算注意力 logits。
        - query: 查询向量 [batch_size, num_steps, embedding_dim]
        - glimpse_K, glimpse_V, logit_K: 键、值和 logits 键
        - mask: 掩码 [batch_size, graph_size+1]
        - 输出: (logits [batch_size, num_steps, graph_size], glimpse)
        """
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # 重塑查询为多头格式
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # 计算多头自注意力
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # 计算注意力权重
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # 投影多头输出
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # 计算 logits
        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # 应用 tanh 裁剪和掩码
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        """
        获取注意力键和值。
        - fixed: 固定上下文
        - state: 当前状态
        - 输出: (glimpse_key, glimpse_val, logit_key)
        """
        if self.is_vrp and self.allow_partial:
            # 支持部分配送：添加需求信息
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )
        # 默认：直接使用固定上下文
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        """
        将向量重塑为多头格式。
        - v: 输入向量 [batch_size, num_steps, graph_size, embed_dim]
        - num_steps: 解码步数（可选）
        - 输出: [n_heads, batch_size, num_steps, graph_size, head_dim]
        """
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)
        )