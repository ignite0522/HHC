# AGH_机场调度

## 前言
**这是整个项目的实验记录**

论文：https://arxiv.org/abs/2303.02442

代码：https://github.com/RoyalSkye/AGH

代码整体过了一遍后，给我的感受是，这个项目的代码量比之前我看到的都要大，之中有很多处理细节的地方，比如条件约束、掩码机制等等，这次呢，就从条件约束和掩码处理这里走一下整个流程

## 整体流程

### 训练初始化与数据准备

run.py的train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts)是整个代码的起始，是一整个轮次的开始

来看看train_epoch()传了哪些参数，optimizer和lr_scheduler就不说了，看看model， val_dataset（验证集数据）和baseline

我们今天直奔AttentionModel和CriticNetworkLSTM



AttentionModel这个模型大致的结构和Transformer差不多，由解码器和编码器组成，这个在之后用到的时候再讲

CriticNetworkLSTM主要的工作是生成基线，至于基线是做什么的，也留在后面讲



好了，继续跟进train_epoch()，把这个函数的代码完整贴出来

```py
def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    """训练一个 epoch，更新模型参数并验证。
    - model: AttentionModel 实例
    - optimizer: 优化器
    - baseline: 基线对象
    - lr_scheduler: 学习率调度器
    - epoch: 当前 epoch
    - val_dataset: 验证数据集
    - problem: 问题定义
    - tb_logger: TensorBoard 日志记录器
    - opts: 配置选项
    """
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)  # 计算全局步骤
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)  # 记录学习率

    # 生成训练数据
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    model.train()  # 切换到训练模式
    set_decode_type(model, "sampling")  # 使用采样解码

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        if model.is_agh:
            train_batch_agh(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        else:
            train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # 保存模型检查点
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # 验证模型
    avg_reward = validate(model, val_dataset, opts)

    # 记录验证结果
    with open(os.path.join(opts.save_dir, 'validate_log.txt'), 'a') as f:
        f.write('Validating Epoch {}, Validation avg_cost: {}\n'.format(epoch, avg_reward))
        f.write('\n')

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)  # 记录验证奖励

    baseline.epoch_callback(model, epoch)  # 基线回调

    lr_scheduler.step()  # 更新学习率

```

在训练之前看看他的数据长什么样

```py
training_dataset = baseline.wrap_dataset(
        problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
```

problem.make_dataset生成原始数据，包含：

​	loc：登机口索引，形状 [num_samples, graph_size]

​	demand：需求，形状 [num_samples, num_fleets, graph_size]

​	arrival：时间窗口左边界，形状 [num_samples, graph_size]，**每个登机口的最早服务时间**

​	departure：时间窗口右边界，形状 [num_samples, graph_size]，**每个登机口的最晚服务时间**

​	type：登机口类型，形状 [num_samples, graph_size]

baseline.wrap_dataset函数根据输入的数据生成每个车队对应的b_val(这就是提到的基线，他的作用等会儿再讲)，之后讲data和b_val打包返回

DataLoader 在这里的作用是：从 training_dataset 中分批加载数据，每次返回一个批次 (batch, bl_val)



继续来到了

```py
for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
    if model.is_agh:
        train_batch_agh(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
    else:
        train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
    step += 1
```

这里我们只关注处理agh问题的代码

在这个轮次中，继续使用for循环，针对每个批次进行训练train_batch_agh

之后针对每个批次训练出的模型，使用validate(model, val_dataset, opts)进行验证(**验证过程不更新模型参数（编码器和解码器），而是运行模型在验证集上的推理，评估其在未见过数据上的表现**)



之后使用baseline.epoch_callback(model, epoch)

baseline.epoch_callback(model, epoch) 是在每个训练 epoch 结束时调用，旨在更新基线策略或状态，以提供更准确的基线值（bl_val）给 REINFORCE 算法，减少梯度方差

这个基线的话，大有学问，暂且在这里不讲

思路继续跟回来，我们接着看train_batch_agh

这里还是老规矩，把代码先贴出来

```py
# === 训练 AGH 批次 ===
def train_batch_agh(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    """训练 AGH 批次，使用 REINFORCE 优化多车队路径。
    - model: AttentionModel 实例
    - optimizer: 优化器
    - baseline: 基线对象
    - epoch, batch_id, step: 训练进度
    - batch: 批次数据
    - tb_logger: TensorBoard 日志记录器
    - opts: 配置选项
    """
    x, bl_val = baseline.unwrap_batch(batch)  # 解包数据和基线值
    assert bl_val is not None
    x = move_to(x, opts.device)  # 移动到设备
    bl_val = move_to(bl_val, opts.device)

    set_decode_type(model, "sampling")  # 使用采样解码
    # 初始化时间窗口
    bat_tw_left = x['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1) # 全局时间窗口左边界
    bat_tw_right = x['departure'] # 全局时间窗口右边界
    fleet_cost_together, log_likelihood_together, fleet_cost_list, log_likelihood_list = None, None, [], []

    for f in model.fleet_info['order']:  # 按车队优先级
        # 构造车队输入
        next_duration = torch.tensor(model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                                    device=x['type'].device).repeat(x['loc'].size(0), 1)
        tw_right = bat_tw_right - torch.gather(next_duration, 1, x['type'])
        tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)# 添加车库时间窗

        tw_left = bat_tw_left[model.fleet_info['precedence'][f]]# 单个车队时间窗左边界=全局时间窗口左边界
        tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)# 添加车队时间窗
        duration = torch.tensor(model.fleet_info['duration'][f], device=x['type'].device).repeat(x['loc'].size(0), 1)
        fleet_bat = {'loc': x['loc'], 'demand': x['demand'][:, f - 1, :],
                     'distance': model.distance.expand(x['loc'].size(0), len(model.distance)),
                     'duration': torch.gather(duration, 1, x['type']),
                     'tw_right': tw_right, 'tw_left': tw_left,
                     'fleet': torch.full((x['loc'].size(0), 1), f - 1)}

        if model.rnn_time:
            model.pre_tw = None  # 重置 RNN 隐藏状态

        # 前向传播
        fleet_cost, log_likelihood, serve_time = model(move_to(fleet_bat, opts.device))# 返回的是车队距离成本、对数似然、车队服务开始时间

        # 收集成本和对数似然
        fleet_cost_list.append(fleet_cost)
        log_likelihood_list.append(log_likelihood)
        if fleet_cost_together is None:
            fleet_cost_together, log_likelihood_together = fleet_cost, log_likelihood
        else:
            fleet_cost_together = fleet_cost_together + fleet_cost
            log_likelihood_together = log_likelihood_together + log_likelihood

        # 更新时间窗口
        bat_tw_left[model.fleet_info['precedence'][f] + 1, :, :] = torch.max(
            bat_tw_left[model.fleet_info['precedence'][f] + 1, :, :], serve_time[:, 1:])

    # 计算 REINFORCE 损失
    loss = ((fleet_cost_list[0] - bl_val[:, 0]) * log_likelihood_list[0]).mean()
    for i in range(1, len(fleet_cost_list)):
        loss += ((fleet_cost_list[i] - bl_val[:, i]) * log_likelihood_list[i]).mean()
    loss = loss / len(fleet_cost_list)  # 平均损失

    optimizer.zero_grad()
    loss.backward()  # 反向传播
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)  # 裁剪梯度
    optimizer.step()  # 更新参数

    # 记录日志
    if step % int(opts.log_step) == 0:
        log_values(fleet_cost_together, grad_norms, epoch, batch_id, step, log_likelihood_together, loss, 0, tb_logger, opts)

```

这里就慢慢开始变得烧脑了起来(我花了好大功夫理清楚逻辑)

```x, bl_val = baseline.unwrap_batch(batch)```解包，得到原始数据和bl_val



在这里使用REINFORCE 算法优化 AttentionModel，以最小化多车队路径的总成本

优化多车队（num_fleets）的路径规划，满足登机口的时间窗口（tw_left, tw_right）、容量（demand）和车队优先级约束，在这三个当中首先最不好理解的就是，这里时间窗口的处理（左时间窗口，右时间窗口）



讲好这个时间窗口，首先要理解原始数据给了些什么东西

x['arrival']: 登机口的到达时间（时间窗口左边界），形状 [batch_size, graph_size]，表示每个样本的每个登机口的最早服务时间

x['departure']: 登机口的离开时间（时间窗口右边界），形状 [batch_size, graph_size]，表示最晚服务时间



接着往下看

```py
bat_tw_left = x['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1) 
```

model.fleet_info['next_duration']，跟进model.fleet_info

```py
with open('problems/agh/fleet_info.pkl', 'rb') as f:
    # 加载车队信息：优先级、操作时长、后续操作时间
    # {'order': [1, 2, 4, 8, 3, 5, 7, 9, 6, 10],  # 车队求解顺序
    # 'precedence': {1: 0, 2: 1, ...},  # 车队优先级
    # 'duration': {1: [0.0, 0.0, 0.0], ...},  # 表示当前车队在每种登机口执行服务的操作时长（服务时间）
    # 'next_duration': {4: [0.0, 0.0, 0.0], ...}}  # 表示后续优先级车队在每种登机口执行服务的最短操作时间
    self.fleet_info = pickle.load(f)
```

可以看到加载了车队顺序，车队优先级，操作服务时长和后续操作最短时间

**len(model.fleet_info['next_duration'])**：长度通常等于 优先级数量，

这里的作用是为每一个优先级都创建一个tw_left左时间窗口(这里我之前是很疑惑的，为什么要添加一个总优先级数量的维度，这里造成我疑惑的原因还是没有很好理解到原始数据的含义，原始数据中的arrival和departure仅仅是每个登机口的信息而不是对于优先级而言的)

```bat_tw_right = x['departure'] ```右边界是固定



接着往下看

```py
for f in model.fleet_info['order']:  # 按车队优先级
    # 构造车队输入
    next_duration = torch.tensor(model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                                device=x['type'].device).repeat(x['loc'].size(0), 1)
    tw_right = bat_tw_right - torch.gather(next_duration, 1, x['type'])
    tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)# 添加车库时间窗

    tw_left = bat_tw_left[model.fleet_info['precedence'][f]]# 单个车队时间窗左边界=全局时间窗口左边界
    tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)# 添加车队时间窗
    duration = torch.tensor(model.fleet_info['duration'][f], device=x['type'].device).repeat(x['loc'].size(0), 1)
    fleet_bat = {'loc': x['loc'], 'demand': x['demand'][:, f - 1, :],
                 'distance': model.distance.expand(x['loc'].size(0), len(model.distance)),
                 'duration': torch.gather(duration, 1, x['type']),
                 'tw_right': tw_right, 'tw_left': tw_left,
                 'fleet': torch.full((x['loc'].size(0), 1), f - 1)}
```



```py
next_duration = torch.tensor(model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                            device=x['type'].device).repeat(x['loc'].size(0), 1)
```

这个是获取当前车队优先级对应的后续服务时长

紧接着

```tw_right = bat_tw_right - torch.gather(next_duration, 1, x['type'])```

这个是每个车队对应的右时间窗口，等于全局右边界减去后续服务时长，确保服务时间提前完成以留给低优先级车队足够时间完成服务

```tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)```

这里添加了一个1441，代表一天24小时，车库右边界设为 1441，保证路径可以从任意位置回到车库

```tw_right = tensor([[585, 675, 765], [615, 730, 830]])```==>```tw_right = tensor([[1441, 585, 675, 765], [1441, 615, 730, 830]])```



现在来看看右时间窗口

```
tw_left = bat_tw_left[model.fleet_info['precedence'][f]]# 单个车队时间窗左边界=全局时间窗口左边界
tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)# 添加车队时间窗
```



```bat_tw_left[model.fleet_info['precedence'][f]]```: 选择当前优先级的左边界,示例：

```
bat_tw_left = tensor([[[300, 400, 500], [350, 450, 550]],  # 优先级 0
                      [[300, 400, 500], [350, 450, 550]],  # 优先级 1
                      [[0, 0, 0], [0, 0, 0]]])  # 优先级 2
tw_left = tensor([[300, 400, 500], [350, 450, 550]])  # 优先级 0
```

之后再在前面添加车库左边界（0 分钟），这里0表示允许路径从车库开始



```duration = torch.tensor(model.fleet_info['duration'][f], device=x['type'].device).repeat(x['loc'].size(0), 1)```提取服务时长



最后将整个处理好的数据打包

```py
fleet_bat = {'loc': x['loc'], 'demand': x['demand'][:, f - 1, :],
                     'distance': model.distance.expand(x['loc'].size(0), len(model.distance)),
                     'duration': torch.gather(duration, 1, x['type']),
                     'tw_right': tw_right,
                     'tw_left': tw_left,
                     'fleet': torch.full((x['loc'].size(0), 1), f - 1)}
```

### 模型和训练

#### 编码器

接下来数据处理好了，就该拿去训练了

```py
fleet_cost, log_likelihood, serve_time = model(move_to(fleet_bat, opts.device))
```

跟着我们一起去看看model怎么实现的吧

现在看到attentionmodel.py文件

直接看forward

```py
def forward(self, input, return_pi=False):
    """
    前向传播：生成路径、成本和对数似然。
    - input: 字典，包含以下字段：
        'loc': 登机口索引 [batch_size, graph_size]
        'demand': 需求 [batch_size, graph_size]
        'distance': 节点间距离 [batch_size, len(distance)]
        'duration': 操作时长 [batch_size, graph_size]
        'tw_right', 'tw_left': 时间窗口边界 [batch_size, graph_size+1]
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
        embeddings, _ = self.embedder(self._init_embed(input))  # [batch_size, graph_size+1, embedding_dim]

    # 解码：生成对数概率、路径和服务时间
    _log_p, pi, serve_time = self._inner(input, embeddings)

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
```

这个模型其实就是由编码器和解码器组成

#### 解码器

先来看看编码器吧

``` embeddings, _ = self.embedder(self._init_embed(input))```这里以来就将输入的数据输入到编码器中，生成节点嵌入

那现在就来看看这个编码器GraphAttentionEncoder是怎么实现的吧

```py
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
```

这段代码主要就是线形层=>多层注意力层layers

```py
self.layers = nn.Sequential(*(
    MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
    for _ in range(n_layers)
))
```

那就接着来看看MultiHeadAttentionLayer :

```py
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
```

主要就是使用Sequential来组合各个子模块两个SkipConnection残差连接层和两个Normalization归一化层

这里提到的残差连接层是之前不曾知道的知识，那就来学习一下吧

先来看看SkipConnection是怎么定义的

```py
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
```

实现残差连接：y=x+F(x)，其中：

- x : 输入张量（input）
- F(x) : self.module 的输出（例如 MultiHeadAttention 或前馈网络）
- y : 输出张量（input + self.module(input)）

其实残差连接就是输入加上输出之后一起返回

那他的作用是什么呢？

1.稳定深层编码器训练：我们这里的层数很多，会导致深层网络容易因梯度消失而难以训练，使用残差连接的话可以确保浅层参数（例如 init_embed 或早期层的 W_query, W_key）得到有效更新

2.输入包含多种特征（loc, demand, duration, tw_left, tw_right），使用残差连接可以捕捉更原始特征之间的依赖性

等等......



其中SkipConnection层分别是多头注意力层和前馈网络层

先来看看多头注意力层MultiHeadAttention

```py
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
```

这段代码就是在实现多头注意力机制，虽然很重要(要理清多头注意力机制怎么实现的还是要花点功夫)，因为今天的重点在讲掩码和整个流程，这里掩码机制(这段代码中有)并没有在编码器中实际应用到，所以这里就不细细讲来了

前馈网络层没什么好讲的，这里是两个全连接层，目的是为了捕捉更复杂的模式



到此为止编码器就理完了

现在来看看解码器吧

```_log_p, pi, serve_time = self._inner(input, embeddings)	```

看看inner是怎么实现解码器功能的

```py
def _inner(self, input, embeddings):
    """
    解码过程：逐步选择节点，构建路径。
    - input: 输入数据
    - embeddings: 节点嵌入 [batch_size, graph_size+1, embedding_dim]
    - 输出: (对数概率 [batch_size, steps, graph_size+1], 路径 [batch_size, steps], 服务时间)
    """
    outputs = []
    sequences = []

    # 初始化状态
    state = self.problem.make_state(input)

    # 预计算固定上下文
    if self.is_agh:
        fixed = self._precompute(embeddings, input['fleet'])
    else:
        fixed = self._precompute(embeddings, None)

    batch_size = state.ids.size(0)

    # 解码循环
    i = 0
    while not (self.shrink_size is None and state.all_finished()):
        if self.shrink_size is not None:
            # 检查是否可以收缩批次
            unfinished = torch.nonzero(state.get_finished() == 0)
            if len(unfinished) == 0:
                break
            unfinished = unfinished[:, 0]
            if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                state = state[unfinished]
                fixed = fixed[unfinished]

        # 计算节点选择概率和掩码
        log_p, mask = self._get_log_p(fixed, state)

        # 选择下一个节点
        selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

        # 更新状态
        state = state.update(selected)

        # 恢复原始批次大小（若收缩）
        if self.shrink_size is not None and state.ids.size(0) < batch_size:
            log_p_, selected_ = log_p, selected
            log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
            selected = selected_.new_zeros(batch_size)
            log_p[state.ids[:, 0]] = log_p_
            selected[state.ids[:, 0]] = selected_

        # 收集结果
        outputs.append(log_p[:, 0, :])
        sequences.append(selected)

        i += 1

    # 整理输出
    if self.is_agh:
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state.serve_time
    else:
        return torch.stack(outputs, 1), torch.stack(sequences, 1), None
```

这里的```fixed = self._precompute(embeddings, input['fleet'])```十分重要，刚开始还看漏了

```py
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
```


```py
graph_embed = embeddings.mean(1)
fixed_context = self.project_fixed_context(graph_embed)[:, None, :]# 对提取到的全局特征做一个全连接层处理
```

mean(1) 对所有节点取平均，生成一个全局表示，捕捉图的整体特征

例如：

- 空间特征：所有登机口的平均位置（loc）
- 时间特征：时间窗口（tw_left, tw_right）的平均分布
- 需求特征：总需求（demand）的平均值

```fleet_embedding = self.fleets_embedding(fleet)```就是获取对车队的嵌入

```py
# 投影节点嵌入到键、值和 logits
glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
    self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

# 转换为多头格式
fixed_attention_node_data = (
    self._make_heads(glimpse_key_fixed, num_steps),
    self._make_heads(glimpse_val_fixed, num_steps),
    logit_key_fixed.contiguous()
)
```

目的是为了从嵌入层中获取注意力所需的键、值和 logits



还有个主要的功能主要就是使用while实现（着重要说的就是这里使用了all_finished，确保每个节点都被访问到）

**解码循环**：

- 计算节点选择概率（log_p）和掩码（mask）
- 根据概率和掩码选择下一个节点（selected）
- 更新状态（state），包括路径和服务时间

那就一步步往下看吧

```log_p, mask = self._get_log_p(fixed, state)```

```py
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
    if self.is_agh:
        query = query + fixed.fleet_embedding

    # 获取键和值
    glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

    # 获取掩码
    mask = state.get_mask()

    # 计算 logits
    log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

    if normalize:
        log_p = torch.log_softmax(log_p / self.temp, dim=-1)

    assert not torch.isnan(log_p).any()

    return log_p, mask
```

一步步看

```py
query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state).float())
if self.is_agh:
    query = query + fixed.fleet_embedding
```

根据全局上下文，动态步骤上下文和车队嵌入生成查询向量



```py
glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
```

从 fixed 和 state 提取注意力机制的键和值

#### 掩码mask机制

```py
mask = state.get_mask()
```

获取掩码

这就是今天的主角了

```py
def get_mask(self):
    """
    生成掩码，标记不可行节点（0=可行，1=不可行）
    - 形状: [batch_size, 1, graph_size+1]
    - 考虑已访问节点、剩余容量和时间窗口约束
    - 禁止连续两次访问车库，除非所有登机口已访问
    """
    # 处理已访问节点掩码
    if self.visited_.dtype == torch.uint8:
        visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, graph_size]，排除车库
    else:
        visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))  # [batch_size, 1, graph_size]，转换为布尔掩码

    # 检查容量约束
    exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)  # [batch_size, 1, graph_size]，需求超过容量

    # 计算距离索引
    pre_coord = self.coords[self.ids, self.prev_a]  # [batch_size, 1, 2]，上一个节点坐标
    all_coord = self.coords[:, 1:]  # [batch_size, graph_size, 2]，所有登机口坐标
    distance_index = self.NODE_SIZE * pre_coord.expand(self.coords.size(0), self.coords.size(1)-1) + all_coord  # [batch_size, graph_size]，距离矩阵索引

    # 检查时间窗口约束
    exceeds_tw = ((torch.max(
        self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 完成上一个服务后的时间
        self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
        self.tw_left[:, 1:])  # 确保不早于时间窗口左边界
        + self.duration[:, 1:]) > self.tw_right[:, 1:])[:, None, :]  # [batch_size, 1, graph_size]，服务时间超过右边界

    # 不可行节点 = 已访问 或 容量超限 或 时间窗口不可行
    mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | exceeds_tw  # [batch_size, 1, graph_size]

    # 车库掩码：如果刚访问车库且仍有未访问登机口，禁止再次访问车库
    mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]
    return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size+1]，合并车库和登机口掩码
```

这么多代码，实现的就是**约束**：

- **已访问节点**：屏蔽已访问的登机口（visited_loc）
- **容量约束**：屏蔽需求超过车辆剩余容量的节点（exceeds_cap）
- **时间窗口约束**：屏蔽服务时间超出时间窗口右边界的节点（exceeds_tw）
- **车库访问规则**：禁止连续两次访问车库（mask_depot），除非所有登机口已访问



1.已访问节点

```py
 visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1)) 
```

屏蔽已访问的登机口（visited_loc = True），防止重复访问



2.容量约束

```py
exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY) 
```

当前节点需求demand加上已经使用的容量used_capacity，将两者之和与车辆总容量做比较，若大于，则设为True



3.时间窗口约束

```py
 # 计算距离索引
        pre_coord = self.coords[self.ids, self.prev_a]  # [batch_size, 1, 2]，上一个节点坐标
        all_coord = self.coords[:, 1:]  # [batch_size, graph_size, 2]，所有登机口坐标
        distance_index = self.NODE_SIZE * pre_coord.expand(self.coords.size(0), self.coords.size(1)-1) + all_coord  # [batch_size, graph_size]，距离矩阵索引
# 这步是为了计算当前点到所有点的距离索引

 # 检查时间窗口约束
exceeds_tw = ((torch.max(
            self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 完成上一个服务后的时间
            self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
            self.tw_left[:, 1:])  # 确保不早于时间窗口左边界
            + self.duration[:, 1:]) > self.tw_right[:, 1:])[:, None, :]  # [batch_size, 1, graph_size]，服务时间超过右边界
```

这步使用max，在完成上一个服务后的时间+旅行时间（即两点之间行驶的时间）和 左时间窗口做比较

如果前者大于后者，证明该车队不能在规定时间内完成该任务，设为True



4.车库访问规则

```py
mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | exceeds_tw  # [batch_size, 1, graph_size]

# 车库掩码：如果刚访问车库且仍有未访问登机口，禁止再次访问车库
mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]
```

这一步是为了限制连续重复访问车库



最后使用cat将所有约束条件的掩码合并返回



接下来就是计算概率数值了

```py
log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
```

接着看_one_to_many_logits

```py
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

    # 计算兼容性
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
```

这段的功能是使用前面已经准备好的Q，K，V来计算最终的分数

其中值得注意的是

```py
compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
if self.mask_inner:
    assert self.mask_logits, "Cannot mask inner without masking logits"
    compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
    
if self.tanh_clipping > 0:
    logits = torch.tanh(logits) * self.tanh_clipping   #  限制 logits 范围（例如 [-10, 10]），增强稳定性
if self.mask_logits:
    logits[mask] = -math.inf
```

这里将我们之前辛辛苦苦做好的mask实际应用到了注意力计算上(标记为True的赋值为负无穷，经过softmax后即为0)



最后使用```log_p = torch.log_softmax(log_p / self.temp, dim=-1)```得到概率分布



接着回到_inner函数

_get_log_p过了就是```selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])```了

```py
def _select_node(self, probs, mask):
    """
    选择下一个节点。
    - probs: 概率分布 [batch_size, graph_size+1]
    - mask: 掩码 [batch_size, graph_size+1]
    - 输出: 选择的节点索引 [batch_size]
    """
    assert (probs == probs).all(), "Probs should not contain any nans"

    if self.decode_type == "greedy":
        _, selected = probs.max(1)  # 贪婪选择
        assert not mask.gather(1, selected.unsqueeze(
            -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
    elif self.decode_type == "sampling":
        selected = probs.multinomial(1).squeeze(1)  # 采样选择
        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            print('Sampled bad values, resampling!')
            selected = probs.multinomial(1).squeeze(1)
    else:
        assert False, "Unknown decode type"

    return selected
```

通常在训练时使用随机采样（decode_type="sampling"），以促进探索，而在测试时使用贪婪策略（decode_type="greedy"），以获得最优解

在训练时，这里主要功能就是返回随机采样的选择的节点



接着就是```state = state.update(selected)```更新状态

```py
def update(self, selected):
    # 更新状态，基于选择的节点（selected）更新路径和状态
    assert self.i.size(0) == 1, "Can only update if state represents single step"  # 确保当前状态只表示单步

    # 更新状态
    selected = selected[:, None]  # [batch_size, 1]，扩展维度以匹配状态
    prev_a = selected  # [batch_size, 1]，更新上一个节点为当前选择
    n_loc = self.demand.size(-1)  # 登机口数（不包括车库）

    # 计算路径长度
    cur_coord = self.coords[self.ids, selected]  # [batch_size, 1, 2]，当前节点的坐标
    distance_index = self.NODE_SIZE * self.coords[self.ids, self.prev_a] + cur_coord  # [batch_size, 1]，距离矩阵索引
    lengths = self.lengths + self.distance.gather(1, distance_index)  # [batch_size, 1]，累加路径距离

    # 更新已使用容量
    selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]  # [batch_size, 1]，当前节点的需求（车库需求为 0）
    used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()  # [batch_size, 1]，非车库时累加需求，车库时重置为 0

    # 计算当前可用时间（cur_free_time）
    cur_free_time = (torch.max(
        self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,  # 到达时间 = 上一个完成时间 + 旅行时间
        self.tw_left[self.ids, selected])  # 确保不早于时间窗口左边界
                     + self.duration[self.ids, selected]) * (prev_a != 0).float() - 60 * (prev_a == 0).float()  # [batch_size, 1]，非车库时加服务时长，车库时设为 -60

    # 更新已访问掩码
    if self.visited_.dtype == torch.uint8:
        # 对于 uint8 掩码，直接使用 scatter 设置当前节点为已访问
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)  # [batch_size, 1, graph_size+1]
    else:
        # 对于 int64 掩码，使用 mask_long_scatter 设置（忽略车库，prev_a - 1）
        visited_ = mask_long_scatter(self.visited_, prev_a - 1)

    # 更新服务时间
    serve_time = self.serve_time.scatter_(1, selected, cur_free_time.float())  # [batch_size, graph_size+1]，记录当前节点的服务时间

    # 更新路径序列
    tour = torch.cat((self.tour, selected), dim=1)  # [batch_size, steps+1]，添加当前节点到路径

    # 返回更新后的 StateAGH 实例
    return self._replace(
        prev_a=prev_a,  # 更新上一个节点
        used_capacity=used_capacity,  # 更新已使用容量
        visited_=visited_,  # 更新已访问掩码
        lengths=lengths,  # 更新路径长度
        cur_coord=cur_coord,  # 更新当前坐标
        i=self.i + 1,  # 步骤计数器加 1
        cur_free_time=cur_free_time,  # 更新当前可用时间
        tour=tour,  # 更新路径序列
        serve_time=serve_time  # 更新服务时间
    )
```

根据return返回的值可以看到update更新了什么



最后_inner函数将整个批次的对数概率log_p，节点索引selected和服务时长serve_time返回



现在是计算成本和掩码

```py
cost, mask = self.problem.get_costs(input, pi)
```

接下来就看看get_costs,说实话，整个项目的工程量太大了，函数一个接着套一个，流程理到这里已经过半了，已经疲惫了，实在是佩服整个项目的创作者们

```py
def get_costs(dataset, pi):
    """
    计算路径成本并验证路径有效性。
    - dataset: 输入数据，包含 loc, demand, distance, tw_left, tw_right, duration 等。
    - pi: 路径（[batch_size, seq_length]），表示每个样本的节点访问顺序。
    - 返回：总距离成本 ([batch_size]) 和掩码（当前为 None）。
    """
    batch_size, graph_size = dataset['demand'].size()  # 获取批次大小和登机口数量（不含车库）

    # 验证路径有效性：确保 pi 包含 0 到 n-1 的所有节点
    sorted_pi = pi.data.sort(1)[0]  # 对路径按节点索引排序
    assert (
        # 检查排序后的路径后半部分是否为 1...n
        torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
        sorted_pi[:, -graph_size:]
    ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"  # 前半部分应全为 0（车库）

    # 处理需求：访问车库重置容量，添加虚拟需求 -VEHICLE_CAPACITY
    demand_with_depot = torch.cat(
        (
            torch.full_like(dataset['demand'][:, :1], -AGH.VEHICLE_CAPACITY),  # 车库需求为 -1.0
            dataset['demand']  # 登机口需求
        ),
        1
    )
    d = demand_with_depot.gather(1, pi)  # 按路径 pi 顺序获取需求

    # 验证容量约束
    used_cap = torch.zeros_like(dataset['demand'][:, 0])  # 初始化已用容量为 0
    for i in range(pi.size(1)):
        used_cap += d[:, i]  # 累加需求，访问车库时重置（负值）
        used_cap[used_cap < 0] = 0  # 容量不能为负
        assert (used_cap <= AGH.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

    # 获取路径的节点索引（包括车库）
    loc = torch.cat((torch.zeros_like(dataset['loc'][:, :1]), dataset['loc']), dim=1)  # 车库索引为 0
    loc = loc.gather(1, pi)  # 按路径 pi 顺序获取节点

    # 计算距离索引：NODE_SIZE * from_node + to_node
    distance_index = AGH.NODE_SIZE * torch.cat((torch.zeros_like(loc[:, :1]), loc), dim=1) + \
                     torch.cat((loc, torch.zeros_like(loc[:, :1])), dim=1)
    batch_distance = dataset['distance']  # 距离矩阵 [batch_size, NODE_SIZE*NODE_SIZE]

    # 验证时间窗口约束
    ids = torch.arange(batch_size, dtype=torch.int64, device=pi.device)[:, None]  # 批次索引
    time_distance = batch_distance / AGH.SPEED  # 距离转换为时间（分钟）
    time_distance = time_distance.gather(1, distance_index)  # 按路径获取时间
    cur_time = torch.full_like(pi[:, 0:1], -60)  # 初始时间为 -60（假设提前到达）
    duration = torch.cat((torch.zeros_like(dataset['duration'][:, :1], device=dataset['duration'].device),
                         dataset['duration']), dim=1)  # 车库服务时长为 0
    for i in range(pi.size(1)):
        # 更新当前时间：max(到达时间, tw_left) + 服务时长
        cur_time = (torch.max(cur_time + time_distance[:, i:i+1], dataset['tw_left'][ids, pi[:, i:i+1]])
                    + duration[ids, pi[:, i:i+1]]) * (pi[:, i:i+1] != 0).float() - \
                   60 * (pi[:, i:i+1] == 0).float()  # 访问车库重置时间
        assert (cur_time <= dataset['tw_right'][ids, pi[:, i:i+1]] + 1e-5).all(), "Time window violation"

    # 计算总距离成本
    return batch_distance.gather(1, distance_index).sum(1), None
```

这段代码说是在计算cost成本，实际上主要在验证当前生成的路径的有效性

确保路径（pi）包含所有登机口（1 到 graph_size）且车库（0）出现合理次数

验证容量约束：路径中的需求累积不超过 VEHICLE_CAPACITY

验证时间窗口约束：服务时间在 tw_left 和 tw_right 之间



计算cost

```py
batch_distance = dataset['distance'] 
return batch_distance.gather(1, distance_index).sum(1), None
```



接着是计算对数似然```ll = self._calc_log_likelihood(_log_p, pi, mask)```

def _calc_log_likelihood(self, _log_p, a, mask) 方法的核心功能是将每条路径的对数概率求和，得到每条路径的总对数似然



终于attention_model模块流程理完了

回到train.py



使用

```
fleet_cost_list.append(fleet_cost)
log_likelihood_list.append(log_likelihood)
```

收集每个车队的成本（fleet_cost）和对数似然（log_likelihood），以便后续计算 REINFORCE 损失



```py
bat_tw_left[model.fleet_info['precedence'][f] + 1, :, :] = torch.max(
    bat_tw_left[model.fleet_info['precedence'][f] + 1, :, :], serve_time[:, 1:])
```

每个车队计算完过后，记得更新bat_tw_left全局时间窗口

## REINFORCE

最后就是计算REINFORCE 损失了

```py
loss = ((fleet_cost_list[0] - bl_val[:, 0]) * log_likelihood_list[0]).mean()
for i in range(1, len(fleet_cost_list)):
    loss += ((fleet_cost_list[i] - bl_val[:, i]) * log_likelihood_list[i]).mean()
loss = loss / len(fleet_cost_list)  # 平均损失
```

REINFORCE公式

![image-20250510160141463](https://s2.loli.net/2025/05/10/kTpgAVFzRJZjwO2.png)

其中

![image-20250510160227199](https://s2.loli.net/2025/05/10/OUJ1rEexaQDj5i4.png)



好了，现在可以聊聊基线的作用了

基线（baseline）在 REINFORCE 算法中的主要作用是减小策略梯度的方差，从而使训练过程更稳定、更高效

在本项目中有三种基线可供选择：

**RolloutBaseline**：运行基线模型（baseline.model）生成

**ExponentialBaseline**：指数移动平均

**CriticBaseline**：Critic 网络预测

举例说明：

```py
# 简单例子：基线的作用（AGH 问题，单车队，批次大小=2）

# 路径数据
costs = [1000, 200]  # 路径1: 1000（高成本）, 路径2: 200（低成本）
log_likelihoods = [-5, -6]  # 路径1: -5, 路径2: -6
baseline = [500, 500]  # 基线: 平均成本 500

# 无基线
loss_no_baseline = 0
for cost, ll in zip(costs, log_likelihoods):
    loss_no_baseline += (-cost) * ll  # 回报 = -cost
loss_no_baseline = -loss_no_baseline / len(costs)
print("无基线损失:", loss_no_baseline)  # 输出: -3100.0
# 路径1贡献: (-1000) * (-5) = 5000
# 路径2贡献: (-200) * (-6) = 1200
# 损失: -(5000 + 1200) / 2 = -3100

# 有基线
loss_with_baseline = 0
for cost, ll, bl in zip(costs, log_likelihoods, baseline):
    advantage = bl - cost  # 优势 = 基线 - 成本
    loss_with_baseline += advantage * ll
loss_with_baseline = -loss_with_baseline / len(costs)
print("有基线损失:", loss_with_baseline)  # 输出: -350.0
# 路径1贡献: (500 - 1000) * (-5) = 2500
# 路径2贡献: (500 - 200) * (-6) = -1800
# 损失: -(2500 - 1800) / 2 = -350

# 基线效果:
# - 减小梯度方差：优势[-500, 300]比回报[-1000, -200]波动小
# - 稳定训练：损失从 -3100 降到 -350
# - 平衡优化：高成本和低成本路径贡献更均衡
```

![image-20250510161219836](https://s2.loli.net/2025/05/10/nGxc5UkuPfj6RsH.png)

刚开始学的时候，看了很多概念，还不如上面的例子来得简单易懂



## 总结

就梳理到这里吧，说实话，之前看了两个类似的项目，但和这个项目比起来，都是小巫见大巫，本项目的代码体量确实很大，花了很多时间和精力来理清整个函数和每个函数的具体实现，着重理解了在代码实现过程中的各种约束和掩码机制





# HCC_Home health care

## 前言

在想本项目处理的问题可以做适当的泛化吗，应用到其他的领域

和指导老师探讨了一下，发现在医生调度这个方向并没有使用learning的方法来实现的，那我们这个项目可以用在处理这个上面不呢？

要知道可不可以用上去，那就要把本项目的原始数据是怎么生成的搞清楚

## 原始数据

那就来看看problem_agh.py文件

```py
def make_instance(args):
    """
    将输入数据转换为张量格式，生成 AGH 实例。
    - args: 包含 loc, arrival, departure, type_, demand 等。
    - 返回：字典，包含张量化的数据。
    """
    loc, arrival, departure, type_, demand, *args = args
    return {
        'loc': torch.tensor(loc, dtype=torch.long),  # 登机口索引
        'arrival': torch.tensor(arrival, dtype=torch.float),  # 到达时间
        'departure': torch.tensor(departure, dtype=torch.float),  # 离开时间
        'type': torch.tensor(type_, dtype=torch.long),  # 节点类型
        'demand': torch.tensor(demand, dtype=torch.float)  # 需求
    }


class AGHDataset(Dataset):
    """
    AGH 数据集类，用于生成或加载 AGH 数据。
    - 支持随机生成或从 .pkl 文件加载。
    - 数据包含 loc, arrival, departure, type, demand。
    """
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, fleet_size=10):
        """
        初始化数据集。
        - filename: .pkl 文件路径（可选）。
        - size: 每个样本的登机口数量（默认 50）。
        - num_samples: 样本数量（默认 1000000）。
        - offset: 数据偏移（用于切片）。
        - distribution: 未使用。
        - fleet_size: 车队数量（默认 10）。
        """
        super(AGHDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            # 从文件加载数据
            assert os.path.splitext(filename)[1] == '.pkl', "File must be .pkl"
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            # 随机生成数据
            CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50., 200: 60., 300: 70.}  # 容量缩放因子
            n_gate, n_hour, n_min, prob = 91, 24, 60, np.load('problems/agh/arrival_prob.npy')  # 登机口数、小时、分钟、到达概率
            loc = 1 + np.random.choice(n_gate, size=(num_samples, size))  # 随机生成登机口索引 (1-91)
            arrival = 60 * np.random.choice(n_hour, size=(num_samples, size),
                                            p=prob) + np.random.randint(0, n_min, size=(num_samples, size))  # 到达时间
            stay = torch.tensor([30, 34, 33]).repeat(num_samples, 1)  # 停留时间（根据类型）
            type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)  # 节点类型 (0-2)
            departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)  # 离开时间 = 到达 + 停留
            demand = np.random.randint(1, 10, size=(num_samples, fleet_size, size)) / CAPACITIES[size]  # 需求（归一化）
            data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), demand.tolist()))

        # 转换为张量实例
        self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
        self.size = len(self.data)
```

主要代码是这一块

```py
# 随机生成数据
    CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50., 200: 60., 300: 70.}  # 容量缩放因子
    n_gate, n_hour, n_min, prob = 91, 24, 60, np.load('problems/agh/arrival_prob.npy')  # 登机口数、小时、分钟、到达概率
    loc = 1 + np.random.choice(n_gate, size=(num_samples, size))  # 随机生成登机口索引 (1-91)
    arrival = 60 * np.random.choice(n_hour, size=(num_samples, size),
                                    p=prob) + np.random.randint(0, n_min, size=(num_samples, size)) # 到达时间
    stay = torch.tensor([30, 34, 33]).repeat(num_samples, 1)  # 停留时间（根据类型）
    type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long) # 节点类型 (0-2)
    departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)  # 离开时间 = 到达 + 停留
    demand = np.random.randint(1, 10, size=(num_samples, fleet_size, size)) / CAPACITIES[size] # 需求（归一化）
    data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), demand.tolist()))

# 转换为张量实例
self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
self.size = len(self.data)
```

没什么技巧，把这些都一一记录一遍吧，最重要的是和医生调度问题做一个比较

1.```CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50., 200: 60., 300: 70.} ```

这是在给归一化提供分母，确保需求量满足总车辆容量（AGH.VEHICLE_CAPACITY = 1.0）



2.```n_gate, n_hour, n_min, prob = 91, 24, 60, np.load('problems/agh/arrival_prob.npy') ```记录了登记口数，规定了一天24小时，一小时60分钟，prob飞机能到达的概率

这里的prob记录了24个小时中，飞机可能到达的概率，可以代表什么时候是高峰期，什么时候不忙

```
[0.03287671 0.03561644 0.02465753 0.00273973 0.00273973 0.01643836
 0.01643836 0.06849315 0.05479452 0.08219178 0.05479452 0.02191781
 0.04657534 0.05479452 0.03835616 0.06027397 0.03835616 0.04109589
 0.04657534 0.05753425 0.07945205 0.03287671 0.04383562 0.04657534]
```

**这里的prob类比到医生调度，相当于可以表示医生到家检查的需求在一天当中什么时候高，什么时候低**



3.```loc = 1 + np.random.choice(n_gate, size=(num_samples, size))```

使用random.choice在n_gate中随机选择，生成形状为(num_samples, size)的样本

**这里loc在这里代表登机口，类比的话，可以看成是患者家的位置，但有疑问的就是，登机口在一个机场里是固定了的（一般不会增加登机口或者减少），但患者的家的位置可能会有变动，可能会有新的患者家的位置加进来，这在训练好的模型中怎么办呢**

**这里我们遇到的就是如果有新的患者加入进来，现有模型可能对新位置泛化不足，且马上加入新位置再重新训练计算成本高，这里提几个解决方法哈**

**a.扩大患者位置数据，比如说尽可能的考虑到这个地方区域存在哪些位置，一次性将这些位置考虑进去，提前把所有患者位置想好把模型给训练出来**

**b.医院第一次接收该患者时，不用此方法来进行医生的调度，当有一次过后就可以记录该患者的位置，再更新位置数据去训练模型，之后又需要提供医生上门的服务时，就可以使用该方法来规划医生的调度问题了**



4.

```
arrival = 60 * np.random.choice(n_hour, size=(num_samples, size),p=prob) + np.random.randint(0, n_min, size=(num_samples, size))
```

这里是在生成每个航班的起始时间

**这个可以类比成每个患者开始就诊的时间**



5.```stay = torch.tensor([30, 34, 33]).repeat(num_samples, 1) ```

repeat(num_samples, 1)是指在第一纬度，复制num_samples个[30, 34, 33]，这三个是指不同的停留时间

**可以类比成针对每个患者诊治的时间**



6.```type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long) ```

随机生成0，1，2，样本形状为(num_samples, size)

这里的4和5，可以理解为飞机有三种类型0，1，2，每种类型有不同的停留时间[30, 34, 33]

**这里具体的类别可以仔细想想考虑一下，比如说可以把患者的类型分为轻症、重症、急诊，这里为了更好的医治患者应留足诊治服务时间**



7.```departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)```

将起始时间和停留时间相加，得到离开的时间

这里要解释一下torch.gather，就是根据刚刚随机生成的索引type_从stay中选取对应的停留时间

**这个类比成诊治完后离开的时间**



8.```demand = np.random.randint(1, 10, size=(num_samples, fleet_size, size)) / CAPACITIES[size]```这里要详细讲讲demand了

使用random.randint随机生成从1到10的数字除以CAPACITIES归一化

在机场调度问题中，这里的demand因车队任务的不同，具有差异化，所以demand实质性代表的东西也不同，比如说

车队 1：行李运输，demand 反映行李量

车队 8：加油，demand 反映燃料量

**这里类比成医生调度问题呢，现在大致能想到**

**急诊团队（fleet=0）：demand 表示急诊资源（如抢救药物、输液量）**

**内科团队（fleet=1）：demand 表示常规资源（如诊断试剂、口服药）**

**外科团队（fleet=2）：demand 表示手术资源（如缝合材料、麻醉剂）**

**但有个很大的问题就是，在处理机场调度问题时，每个飞机所有的车队都会进行服务，但这个医生调度问题上，并不是每个患者都需要这些团队，如果要动态考虑每个患者的需求(需要哪些车队，这个就变得复杂起来了)**

**现在最好的解决方案就是把这个几个车队不看成急诊、内科、外科，看成其他的(保证所有车队都要服务患者)**

**可以把车队看成**

**车队 0：初步评估（Triage）：检查患者病情，分配资源，类似 AGH 的“调度协调”**

**车队 1：主要治疗（Treatment）：提供核心医疗服务（如药物、急救），类似 AGH 的“主要操作”**

**车队 2：后续护理（Follow-up Care）：记录、康复指导或转诊，类似 AGH 的“收尾任务”**

**不符合实际，明明一个车队就可以完成的事为什么要拆成几个呢**

这里是个大问题，看怎么解决



还有的数据就是针对车队的信息了

```py
# fleet_info {'order': [1, 2, 4, 8, 3, 5, 7, 9, 6, 10],
# 'precedence': {1: 0, 2: 1, 4: 1, 8: 1, 3: 2, 5: 2, 7: 2, 9: 2, 6: 3, 10: 4},
# 'duration': {1: [0.0, 0.0, 0.0], 2: [6.0, 7.0, 6.0], 4: [7.0, 7.0, 8.0], 8: [2.0, 1.0, 3.0], 3: [8.0, 9.0, 8.0], 5: [4.0, 3.0, 2.0], 7: [6.0, 6.0, 6.0], 9: [4.0, 5.0, 5.0], 6: [8.0, 9.0, 8.0], 10: [2.0, 2.0, 1.0]},
# 'next_duration': {4: [0.0, 0.0, 0.0], 3: [2.0, 2.0, 1.0], 2: [10.0, 11.0, 9.0], 1: [18.0, 20.0, 17.0], 0: [25.0, 27.0, 25.0]}}
```

order:车队编号

precedence：车队优先级

duration：服务时长

next_duration:车队 f 完成任务后，低优先级车队开始任务前所需的额外等待时间



## 外出会诊场景分析

患者类型：

- 轻症：感冒、轻微疼痛（需简单检查和药物）。
- 急诊：急性症状（如高烧、呼吸困难，需快速干预）。
- 重症：慢性病急性发作或重症（如心脏病、严重感染，需复杂治疗）



**next_duration**：

**准备时间**：记录病例、准备设备、消毒（2-15 分钟）



1. 车队 0：初步评估（Triage and Initial Assessment）
   - **作用**：到患者家中评估病情（测量体温、血压、问诊病史），确定轻症、急诊或重症，制定初步治疗计划。
   - **现实性**：家庭会诊的第一步，类似急诊室分诊，但使用便携设备（如脉搏血氧仪）。
   - **AGH 类比**：调度协调（确定航班操作优先级）。
   - **优先级**：0（最高）。
   - **示例**：全科医生或护士到场，评估患者是否需紧急治疗或转院。
2. 车队 1：紧急干预（Emergency Intervention）
   - **作用**：为急诊或重症患者提供即时干预（如氧气支持、急救药物注射），稳定病情。
   - **现实性**：外出会诊中，急诊患者需快速处理，防止病情恶化。
   - **AGH 类比**：紧急操作（如飞机故障维修）。
   - **优先级**：0（与评估同级，快速响应）。
   - **示例**：急救医生携带急救箱，为呼吸困难患者提供氧气。
3. 车队 2：常规治疗（Routine Treatment）
   - **作用**：为轻症或稳定后的患者提供常规治疗（如开具处方药、换药、注射抗生素）。
   - **现实性**：家庭会诊中，轻症患者占多数，需标准治疗；急诊/重症患者也需后续治疗。
   - **AGH 类比**：主要操作（如加油）。
   - **优先级**：1（依赖评估/干预）。
   - **示例**：家庭医生为感冒患者开药，或为慢性病患者调整药物。
4. 车队 3：康复治疗（Rehabilitation Therapy）
   - **作用**：为患者提供物理治疗、功能恢复指导（如术后康复、慢性病管理），或心理支持。
   - **现实性**：外出会诊常包括慢性病或术后患者的康复服务，需专业治疗师。
   - **AGH 类比**：附加操作（如设备维护）。
   - **优先级**：1（与常规治疗同级）。
   - **示例**：康复治疗师指导中风患者进行家庭物理治疗。
5. 车队 4：后续护理与随访（Follow-up Care and Monitoring）
   - **作用**：记录病例、提供用药指导、安排转诊或后续随访，监测患者恢复情况。
   - **现实性**：会诊结束时，护士或医生需确保患者有清晰的康复计划。
   - **AGH 类比**：收尾任务（如登机、清理）。
   - **优先级**：2（最低）。
   - **示例**：护士指导糖尿病患者用药，并安排下次随访。
   - 

接着在demand设计这里，可以看到前面提到的车队的功能，在实际医疗中，demand可以类比为某些医疗物资，并且某些服务是不需要的，这里的demand

要视情况而定设为0

这是之前车队的

```
fleet_info: {'order': [1, 2, 4, 8, 3, 5, 7, 9, 6, 10], 'precedence': {1: 0, 2: 1, 4: 1, 8: 1, 3: 2, 5: 2, 7: 2, 9: 2, 6: 3, 10: 4}, 'duration': {1: [0.0, 0.0, 0.0], 2: [6.0, 7.0, 6.0], 4: [7.0, 7.0, 8.0], 8: [2.0, 1.0, 3.0], 3: [8.0, 9.0, 8.0], 5: [4.0, 3.0, 2.0], 7: [6.0, 6.0, 6.0], 9: [4.0, 5.0, 5.0], 6: [8.0, 9.0, 8.0], 10: [2.0, 2.0, 1.0]}, 'next_duration': {4: [0.0, 0.0, 0.0], 3: [2.0, 2.0, 1.0], 2: [10.0, 11.0, 9.0], 1: [18.0, 20.0, 17.0], 0: [25.0, 27.0, 25.0]}}
```



## 模型设计和训练流程

又带着这些修改后的数据(泛化到医生调度问题)走完了整个流程，感觉如果医生调度问题基本按照机场调度问题来适当修改的话，之后训练的代码在整体上能做



在更改数据过后，发现模型卡住了

初步推测是rollout 函数的问题，为 rollout 函数添加详细的 print 语句和检查，覆盖每个关键步骤（数据加载、时间窗口初始化、车队输入构造、模型推理、时间窗口更新）

```py
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

def rollout(model, dataset, opts):
    """在贪婪模式下评估模型，计算每批次成本。
    - model: AttentionModel 实例
    - dataset: 数据集
    - opts: 配置选项
    - 返回: 成本张量
    """
    # 步骤 0：函数开始，打印基本配置
    print("开始 rollout 函数")
    print(f"opts.eval_batch_size: {opts.eval_batch_size}")
    print(f"opts.device: {opts.device}")
    print(f"model.is_agh: {model.is_agh}")
    print(f"fleet_info: {model.fleet_info}")

    set_decode_type(model, "greedy")  # 设置为贪婪解码
    model.eval()  # 切换到评估模式
    print("已设置为贪婪解码并进入评估模式")

    if model.is_agh:
        cost = []
        # 步骤 1：数据集加载
        print("开始遍历数据集")
        for i, bat in enumerate(tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)):
            print(f"\n处理批次 {i+1}/{len(DataLoader(dataset))}")
            # 打印批次数据
            print("bat 键:", list(bat.keys()))
            print("bat['loc'].shape:", bat['loc'].shape)
            print("bat['demand'].shape:", bat['demand'].shape)
            print("bat['arrival'].shape:", bat['arrival'].shape)
            print("bat['departure'].shape:", bat['departure'].shape)
            print("bat['type'].shape:", bat['type'].shape)
            # 检查输入有效性
            try:
                assert torch.all(bat['departure'] >= bat['arrival']), f"批次 {i+1}: departure < arrival"
                assert not torch.any(torch.isnan(bat['arrival'])), f"批次 {i+1}: arrival 包含 NaN"
                assert not torch.any(torch.isnan(bat['departure'])), f"批次 {i+1}: departure 包含 NaN"
                assert torch.all((bat['type'] >= 0) & (bat['type'] < 3)), f"批次 {i+1}: type 越界"
                print("批次输入验证通过")
            except AssertionError as e:
                print(f"批次输入错误: {e}")
                raise

            bat_cost = []
            # 步骤 2：初始化时间窗口
            print("初始化时间窗口")
            bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
            bat_tw_right = bat['departure']
            print("bat_tw_left.shape:", bat_tw_left.shape)
            print("bat_tw_right.shape:", bat_tw_right.shape)
            print("bat_tw_left 示例:", bat_tw_left[0, :2, :5])
            print("bat_tw_right 示例:", bat_tw_right[:2, :5])
            # 检查时间窗口初始化
            try:
                assert not torch.any(torch.isnan(bat_tw_left)), "bat_tw_left 包含 NaN"
                assert not torch.any(torch.isnan(bat_tw_right)), "bat_tw_right 包含 NaN"
                print("时间窗口初始化验证通过")
            except AssertionError as e:
                print(f"时间窗口初始化错误: {e}")
                raise

            for f in model.fleet_info['order']:
                print(f"\n处理车队 {f}")
                # 步骤 3：构造车队输入
                try:
                    # 修复 next_duration 索引（避免 KeyError）
                    next_duration = torch.tensor(
                        model.fleet_info['next_duration'][f],  # 使用车队索引
                        device=bat['type'].device).repeat(bat['loc'].size(0), 1)
                    print("next_duration.shape:", next_duration.shape)
                    print("next_duration 示例:", next_duration[:2])
                except KeyError:
                    print(f"next_duration 索引错误: fleet {f}, precedence {model.fleet_info['precedence'][f]}")
                    raise

                # 计算右时间窗口
                tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
                print("tw_right（减去 next_duration 后）示例:", tw_right[:2, :5])
                try:
                    assert not torch.any(torch.isnan(tw_right)), f"车队 {f}: tw_right 包含 NaN"
                    assert torch.all(tw_right >= 0), f"车队 {f}: tw_right 包含负值"
                except AssertionError as e:
                    print(f"tw_right 计算错误: {e}")
                    print("bat_tw_right 示例:", bat_tw_right[:2, :5])
                    print("next_duration 示例:", next_duration[:2])
                    print("bat['type'] 示例:", bat['type'][:2])
                    raise

                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
                print("tw_right（添加虚拟节点后）.shape:", tw_right.shape)
                print("tw_right 示例:", tw_right[:2, :5])

                # 计算左时间窗口
                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
                print("tw_left.shape:", tw_left.shape)
                print("tw_left 示例:", tw_left[:2, :5])
                try:
                    assert not torch.any(torch.isnan(tw_left)), f"车队 {f}: tw_left 包含 NaN"
                except AssertionError as e:
                    print(f"tw_left 计算错误: {e}")
                    raise

                # 构造服务时长
                duration = torch.tensor(
                    model.fleet_info['duration'][f],
                    device=bat['type'].device).repeat(bat['loc'].size(0), 1)
                duration = torch.gather(duration, 1, bat['type'])
                print("duration.shape:", duration.shape)
                print("duration 示例:", duration[:2])

                # 构造车队输入
                fleet_bat = {
                    'loc': bat['loc'],
                    'demand': bat['demand'][:, f - 1, :],
                    'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
                    'duration': duration,
                    'tw_right': tw_right,
                    'tw_left': tw_left,
                    'fleet': torch.full((bat['loc'].size(0), 1), f - 1)
                }
                print("fleet_bat 键和形状:", {k: v.shape for k, v in fleet_bat.items()})
                try:
                    assert not any(torch.any(torch.isnan(v)) for v in fleet_bat.values()), f"车队 {f}: fleet_bat 包含 NaN"
                except AssertionError as e:
                    print(f"fleet_bat 构造错误: {e}")
                    raise

                if model.rnn_time:
                    model.pre_tw = None
                    print("已重置 RNN 隐藏状态")

                # 步骤 4：模型推理
                print(f"开始模型推理（车队 {f}）")
                with torch.no_grad():
                    fleet_cost, _, serve_time = model(move_to(fleet_bat, opts.device))
                print("fleet_cost.shape:", fleet_cost.shape)
                print("fleet_cost 示例:", fleet_cost[:5])
                print("serve_time.shape:", serve_time.shape)
                print("serve_time 示例:", serve_time[:2, :5])
                try:
                    assert not torch.any(torch.isnan(fleet_cost)), f"车队 {f}: fleet_cost 包含 NaN"
                    assert not torch.any(torch.isnan(serve_time)), f"车队 {f}: serve_time 包含 NaN"
                    assert torch.all(serve_time >= 0), f"车队 {f}: serve_time 包含负值"
                    print("模型推理验证通过")
                except AssertionError as e:
                    print(f"模型推理错误: {e}")
                    raise

                bat_cost.append(fleet_cost.data.cpu().view(-1, 1))
                print("已收集 fleet_cost 到 bat_cost")

                # 步骤 5：更新时间窗口
                print(f"更新时间窗口（车队 {f}）")
                bat_tw_left[model.fleet_info['precedence'][f] + 1] = torch.max(
                    bat_tw_left[model.fleet_info['precedence'][f] + 1], serve_time[:, 1:])
                print("更新后的 bat_tw_left 示例:", bat_tw_left[model.fleet_info['precedence'][f] + 1, :2, :5])
                try:
                    assert not torch.any(torch.isnan(bat_tw_left)), f"车队 {f}: 更新后的 bat_tw_left 包含 NaN"
                except AssertionError as e:
                    print(f"时间窗口更新错误: {e}")
                    raise

            # 步骤 6：合并批次成本
            bat_cost = torch.cat(bat_cost, 1)
            print("bat_cost.shape:", bat_cost.shape)
            print("bat_cost 示例:", bat_cost[:2])
            cost.append(bat_cost)
            print("已收集 bat_cost 到 cost")

        # 步骤 7：返回最终成本
        final_cost = torch.cat(cost, 0)
        print("最终成本形状:", final_cost.shape)
        print("最终成本示例:", final_cost[:5])
        print("Rollout 完成")
        return final_cost

    def eval_model_bat(batch):
        """评估非 AGH 批次成本"""
        print("处理非 AGH 批次")
        with torch.no_grad():
            cost_, _ = model(move_to(batch, opts.device))
        print("非 AGH 成本形状:", cost_.shape)
        return cost_.data.cpu()

    # 非 AGH 分支
    print("处理非 AGH 数据集")
    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)
```

发现是

![image-20250520113353051](https://s2.loli.net/2025/06/19/uF7bhQxADylrzjB.png)

server_time中有负值，证明我的数据构造得有问题



```
array([[1113,  973,  542,  747,  883,  718,  945, 1008,  619, 1279,  823,
         742, 1060,  917,  746,  841, 1066, 1050,  876,  508,  562,  743,
         969,  473,  550, 1207,  959,  696,  913,  865,  405,  958,  448,
        1030, 1220,  970,  715,  843, 1330,  587,  370,  516,  644,  969,
         609,  706,  687,  924, 1355, 1002]
         
tensor([[1233, 1093,  722,  827, 1003,  898, 1065, 1088,  799, 1399,  903,  862,
         1140, 1037,  826,  921, 1246, 1230,  996,  588,  742,  923, 1149,  653,
          730, 1387, 1139,  816, 1033,  985,  585, 1038,  628, 1150, 1340, 1090,
          835,  923, 1410,  667,  550,  636,  764, 1089,  689,  786,  867, 1004,
         1435, 1082]
```



## 纠错

发现车队一直返回仓库，一直在想是不是时间窗口的问题，现在发现是我使用size=10太小了，之前限制了可选节点数量，可能导致无解（只有车库可行）





## 数据尝试

```

fleet_info: {'order': [1, 2, 3], 'precedence': {1: 0, 2: 1, 3: 2}, 
'duration': {1: [5.0, 8.0, 4.0], 2: [6.0, 7.0, 6.0], 3: [7.0, 7.0, 8.0]}, 
'next_duration': {0: [18.0, 20.0, 17.0], 1: [10.0, 11.0, 9.0], 2: [0.0, 0.0, 0.0]}}
```



```
30, 34, 33

fleet_info: {'order': [1, 2, 4, 8, 3, 5, 7, 9, 6, 10], 'precedence': {1: 0, 2: 1, 4: 1, 8: 1, 3: 2, 5: 2, 7: 2, 9: 2, 6: 3, 10: 4}, 
'duration': {1: [0.0, 0.0, 0.0], 2: [6.0, 7.0, 6.0], 4: [7.0, 7.0, 8.0], 8: [2.0, 1.0, 3.0], 3: [8.0, 9.0, 8.0], 5: [4.0, 3.0, 2.0], 7: [6.0, 6.0, 6.0], 9: [4.0, 5.0, 5.0], 6: [8.0, 9.0, 8.0], 10: [2.0, 2.0, 1.0]}, 
'next_duration': {4: [0.0, 0.0, 0.0], 3: [2.0, 2.0, 1.0], 2: [10.0, 11.0, 9.0], 1: [18.0, 20.0, 17.0], 0: [25.0, 27.0, 25.0]}}
```





将数据改为

```
1: [15.0, 18.0, 14.0],  # 团队1
2: [16.0, 17.0, 16.0],  # 团队2
3: [17.0, 17.0, 18.0] 

stay = torch.tensor([100, 104,103]).repeat(num_samples, 1) 
```

测试发现，一直有个节点访问不了（4节点）

发现是因为前一个节点20完成后的时间为1248，但4节点的时间窗口为420-520，这里无法访问4节点，所以卡住了

但随即否决了自己，因为，这里会选择0，选择0过后server_time变为0，解决这个问题

但这样就不是一条时间线上了



```py
bat_tw_left0: tensor([[ 652.,  466.,  459., 1387.,  451.,   31., 1204., 1250., 1021., 1362.,  844.,  557.,  472., 1098.,  577.,   50.,  353., 1436.,  139.,  492.]], device='cuda:0')
bat_tw_left1: tensor([[ 652.,  466.,  459., 1387.,  451.,   31., 1204., 1250., 1021., 1362.,  844.,  557.,  472., 1098.,  577.,   50.,  353., 1436.,  139.,  492.]], device='cuda:0')
serve_time tensor([[ 662.,  476.,  469., 1397.,  461.,   41., 1214., 1260., 1031., 1372.,  854.,  567.,  482., 1108.,  587.,   60.,  363., 1446.,  149.,  502.]], device='cuda:0')
bat_tw_left1: tensor([[ 662.,  476.,  469., 1397.,  461.,   41., 1214., 1260., 1031., 1372.,  854.,  567.,  482., 1108.,  587.,   60.,  363., 1446.,  149.,  502.]], device='cuda:0')




bat_tw_left1: tensor([[ 662.,  476.,  469., 1397.,  461.,   41., 1214., 1260., 1031., 1372.,  854.,  567.,  482., 1108.,  587.,   60.,  363., 1446.,  149.,  502.]], device='cuda:0')
bat_tw_left2: tensor([[ 652.,  466.,  459., 1387.,  451.,   31., 1204., 1250., 1021., 1362.,  844.,  557.,  472., 1098.,  577.,   50.,  353., 1436.,  139.,  492.]], device='cuda:0')
serve_time tensor([[ 672.0000,  486.0000,  484.2074, 1407.0000,  471.0000,   51.0000, 1224.0000, 1270.0000, 1041.0000, 1382.0000,  864.0000,  577.0000,  492.0000, 1118.0000,  597.0000,   70.0000,  373.0000, 1456.0000,  159.0000,  512.0000]], device='cuda:0')
bat_tw_left2: tensor([[ 672.0000,  486.0000,  484.2074, 1407.0000,  471.0000,   51.0000, 1224.0000, 1270.0000, 1041.0000, 1382.0000,  864.0000,  577.0000,  492.0000, 1118.0000,  597.0000,   70.0000,  373.0000, 1456.0000,  159.0000,  512.0000]], device='cuda:0')



bat_tw_left2: tensor([[ 672.0000,  486.0000,  484.2074, 1407.0000,  471.0000,   51.0000, 1224.0000, 1270.0000, 1041.0000, 1382.0000,  864.0000,  577.0000,  492.0000, 1118.0000,  597.0000,   70.0000,  373.0000, 1456.0000,  159.0000,  512.0000]], device='cuda:0')
bat_tw_left3: tensor([[ 652.,  466.,  459., 1387.,  451.,   31., 1204., 1250., 1021., 1362.,  844.,  557.,  472., 1098.,  577.,   50.,  353., 1436.,  139.,  492.]], device='cuda:0')
serve_time tensor([[ 682.0000,  496.0000,  494.2074, 1417.0000,  481.0000,   61.0000, 1234.0000, 1280.0000, 1051.0000, 1392.0000,  874.0000,  587.0000,  502.0000, 1128.0000,  607.0000,   80.0000,  383.0000, 1466.0000,  169.0000,  522.0000]], device='cuda:0')
bat_tw_left3: tensor([[ 682.0000,  496.0000,  494.2074, 1417.0000,  481.0000,   61.0000, 1234.0000, 1280.0000, 1051.0000, 1392.0000,  874.0000,  587.0000,  502.0000, 1128.0000,  607.0000,   80.0000,  383.0000, 1466.0000,  169.0000,  522.0000]], device='cuda:0')

```





```py
2025-05-27 21:51:00,817 - INFO - bat_t_right2: tensor([[1441.,  712.,  544.,  611., 1149.,  912.,  133., 1190., 1202., 1290., 1012.,  852.,  888.,  541., 1065.,  573.,  846.,  129.,  856., 1330.,  215.]])
2025-05-27 21:51:00,818 - INFO - bat_tw_left2: tensor([[ 652.,  484.,  551., 1089.,  852.,   73., 1130., 1142., 1230.,  952.,  792.,  828.,  481., 1005.,  513.,  786.,   69.,  796., 1270.,  155.]], device='cuda:0')
2025-05-27 21:51:00,818 - INFO - bat_tw_left2: tensor([[ 652.,  484.,  551., 1089.,  852.,   73., 1130., 1142., 1230.,  952.,  792.,  828.,  481., 1005.,  513.,  786.,   69.,  796., 1270.,  155.]], device='cuda:0')
2025-05-27 21:51:00,819 - INFO - serve_time tensor([[ 662.0000,  498.9112,  561.0000, 1099.0000,  862.0000,   83.0000, 1140.0000, 1152.0000, 1285.3965,  962.0000,  811.2418,  872.4122,  491.0000, 1015.0000,  523.0000,  796.0000,   97.6543,  818.6038, 1280.0000,  165.0000]], device='cuda:0')
2025-05-27 21:51:00,819 - INFO - serve_time tensor([[ 662.0000,  498.9112,  561.0000, 1099.0000,  862.0000,   83.0000, 1140.0000, 1152.0000, 1285.3965,  962.0000,  811.2418,  872.4122,  491.0000, 1015.0000,  523.0000,  796.0000,   97.6543,  818.6038, 1280.0000,  165.0000]], device='cuda:0')
2025-05-27 21:51:00,820 - INFO - bat_tw_left2更新后: tensor([[ 662.0000,  498.9112,  561.0000, 1099.0000,  862.0000,   83.0000, 1140.0000, 1152.0000, 1285.3965,  962.0000,  811.2418,  872.4122,  491.0000, 1015.0000,  523.0000,  796.0000,   97.6543,  818.6038, 1280.0000,  165.0000]], device='cuda:0')
```







## 尝试后能正确运行的数据

```py
[90, 94, 93]

fleet_info= {'order': [1, 2, 4,  3, 5],
             'precedence': {1: 0, 2: 1, 4: 1,  3: 2, 5: 3 },
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 4: [25.0, 25.0, 26.0], 3: [8.0, 9.0, 8.0], 5: [4.0, 3.0, 2.0]},
             'next_duration': {3: [0.0, 0.0, 0.0],  2: [10.0, 11.0, 9.0], 1: [18.0, 20.0, 17.0], 0: [55.0, 57.0, 55.0]}}
```



```py
110, 114, 113

fleet_info= {'order': [1, 2, 4,  3],
             'precedence': {1: 0, 2: 1, 4: 1,  3: 2},
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 4: [35.0, 35.0, 36.0], 3: [8.0, 9.0, 8.0]},
             'next_duration': {2: [0.0, 0.0, 0.0],   1: [35.0, 32.0, 31.0], 0: [75.0, 77.0, 75.0]}}


fleet_info= {'order': [1, 2, 4,  3],
             'precedence': {1: 0, 2: 1, 4: 1,  3: 2},
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 4: [35.0, 35.0, 36.0], 3: [30.0, 31.0, 30.0]},
             'next_duration': {2: [0.0, 0.0, 0.0],   1: [35.0, 32.0, 31.0], 0: [75.0, 77.0, 75.0]}}
```

## 最终解决

终于！！！！

搞定了！太激动了

我找到我最疑惑的答案了

我最开始说

```
stay 30, 34, 33
fleet_info: {'order': [1, 2, 4, 8, 3, 5, 7, 9, 6, 10], 'precedence': {1: 0, 2: 1, 4: 1, 8: 1, 3: 2, 5: 2, 7: 2, 9: 2, 6: 3, 10: 4}, 'duration': {1: [0.0, 0.0, 0.0], 2: [6.0, 7.0, 6.0], 4: [7.0, 7.0, 8.0], 8: [2.0, 1.0, 3.0], 3: [8.0, 9.0, 8.0], 5: [4.0, 3.0, 2.0], 7: [6.0, 6.0, 6.0], 9: [4.0, 5.0, 5.0], 6: [8.0, 9.0, 8.0], 10: [2.0, 2.0, 1.0]}, 'next_duration': {4: [0.0, 0.0, 0.0], 3: [2.0, 2.0, 1.0], 2: [10.0, 11.0, 9.0], 1: [18.0, 20.0, 17.0], 0: [25.0, 27.0, 25.0]}}
```

10个车队duration加起来的时间远大于stay，我在想是不是代码哪里错了不严谨，害，还是太菜了

不过质疑权威的精神值得夸奖，不然也不会有之后的一次次尝试

说说一开始我哪里弄错了吧

这10个车队是有优先级的，我最开始就在想这个优先级到底有什么作用，一开始完全忽略了他

```py
tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
```

可以看到我们的左时间窗口是以优先级来定的，有些车队，比如2，4，8三个车队具有同一优先级，**意思就是这三个车队是可以同时进行各自的服务的**，这点非常重要，就这么小一个点，害我辛辛苦苦调试了一个半星期，不得不承认原项目作者在这里的巧思，**使用优先级这个概念巧妙的将一些车队可以同时工作服务表现了出来**



我们现在来回顾，梳理一下吧

还是先把数据给出来

```py
stay  110, 114, 113
fleet_info= {'order': [1, 2, 4,  3],
             'precedence': {1: 0, 2: 1, 4: 1,  3: 2},
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 4: [35.0, 35.0, 36.0], 3: [35.0, 31.0, 30.0]},
             'next_duration': {2: [0.0, 0.0, 0.0],   1: [38.0, 32.0, 31.0], 0: [75.0, 77.0, 75.0]}}

#时间窗30-72      66-110
```



我门从两个角度去理清代码的运行

**1.针对单个车队，跑完全部节点**

这里就是使用inner实现的，在这个inner里面，我们不看代码就结合实际也知道要先算对数概率和掩码，然后根据对数概率和掩码来选择下一个节点，选择完下一个节点过后更新状态

代码也是上述这个逻辑，那就跟着这个逻辑来吧

这里结合代码来讲要清楚点，不然看着云里雾里的

```py
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
```

这个函数就干两个事，以根据已知数据得到Q，K，V，接着使用**state的get_mask**去得到掩码mask（这里大概讲一下就是主要根据是否已经访问，容量，时间窗口来判断是否给予掩码），在这个get_mask中我着重讲一下检查时间窗口的实现

```py
exceeds_tw = ((torch.max(
            self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 此车队完成上一个服务后的时间
            self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
            self.tw_left[:, 1:])  # 确保不早于时间窗口左边界
            + self.duration[:, 1:]) > self.tw_right[:, 1:])[:, None, :]
```

这里着重注意self.cur_free_time，这里的self.cur_free_time是在整个类中共用的，这个类中存在update函数，也就是说当前的self.cur_free_time是上一次完成update过后的时间节点，代表该车队上次完成服务过后的时间节点

车队完成上一次服务过后会向下一个节点出发，这两个节点之间是存在距离的，这部分距离会花费时间```self.distance.gather(1, distance_index) / self.SPEED```（以此表示），之后在**(cur_free_time+旅行时间之和)**，**时间左窗口**之间使用max选择最大值，最后加上服务时间duration来和右时间窗口做比较来确定是否给予掩码



继续流程，拿着Q，K，V和掩码mask来计算对数概率



接着就是拿着对数概率和掩码mask来选择先一个节点，返回selected



紧接着，调用**state的update**，注意这里我把**state的get_mask**和**state的update**标黑了，这两个函数都是在state中的，我要强调的是两个函数共用一个self.cur_free_time

update的大概作用就是计算路径长度，更新容量，计算当前可用时间节点(self.cur_free_time)，更新已经访问节点(赋予其掩码)

这里还是主要针对self.cur_free_time来讲，因为一开始我很大的问惑就在于时间窗口

```py
cur_free_time = (torch.max(
            self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,  # 到达时间 = 上一个完成时间 + 旅行时间
            self.tw_left[self.ids, selected])  # 确保不小于时间窗口左边界
                         + self.duration[self.ids, selected]) * (prev_a != 0).float() - 60 * (prev_a == 0).float()  # [batch_size, 1]，非车库时加服务时长，车库时设为-60
```

这里的作用就是在**(上一次完成服务的时间节点+旅行时间)**与**左时间窗口**之间使用max选择最大值，再加上服务时间duration得到当前的cur_free_time

我单独讲get_mask和upate中的cur_free_time想强调的是，如果一开始我们从仓库出发的话，cur_free_time的值为-60，所以怎么选最大值都是左时间窗口tw_left，但之后针对单个车队来说，从仓库出发到A节点，从A节点到B节点时就要考虑**(上一次完成服务的时间节点+旅行时间)**与**左时间窗口**的大小关系了，这也是判断时间窗口掩码的重要依据(判断点)

还要注意的就是如果下一个节点时仓库的话，这里的cur_free_time又变为了-60

我现在的疑问是，这样给cur_free_time赋值为-60的话，会打破时间线，比如本来在A节点服务的时间窗口是800-910，随着返回仓库再从仓库出发去B节点(时间窗口400-510)，exceeds_tw的逻辑中，因为cur_free_time=-60了，在考虑了仓库到B节点旅行时间情况下，max取之会取到B节点的左时间窗口，但有问题的就是这个左时间窗口的值为400，小于了刚刚访问A节点的800，这样的话就把时间线打破了(不符合常理逻辑)





2.针对整个车队，只针对第一个访问的实际节点来探讨

再来看看数据

```py
stay  110, 114, 113
fleet_info= {'order': [1, 2, 4, 3],
             'precedence': {1: 0, 2: 1, 4: 1, 3: 2},
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 4: [35.0, 35.0, 36.0], 3: [35.0, 31.0, 30.0]},
             'next_duration': {2: [0.0, 0.0, 0.0],   1: [38.0, 32.0, 31.0], 0: [75.0, 77.0, 75.0]}}

#时间窗30-72      66-110
```

要注意的是全局右时间窗口要减去对应车队的next_duration，这个next_duration非常重要，最开始没有理解到，卡了一个星期，**next_duration的作用是为了给后续所有车队留足足够的时间来服务飞机**，比如这里第1个车队的next_duration为75，那么1车队可以服务的时间窗口为0-35(35为110-75得来)

一开始，因为**从仓库出发**，所以可以认为1车队在飞机左时间窗口(这里假设一开始左时间窗口为0)还没开始之前就已经到达了，开始服务，服务30分钟过后，1车队服务完，该节点的左时间窗口从0更新成了30，下一个车队，2，4车队(2，4车队的优先级是一样的，代表这两个车队同时服务该节点)的服务时间窗口为30-72，满足2，3车队的duration的值36和35，2车队服务完过后，该节点的左时间窗口从30变为66，下一个车队，3车队的服务时间窗口为66-110，足够3车队(duration=35)在这段时间内服务了

**这里的这些数据也是根据上述原则的话就不会出错**





# 实际应用

对比了之前的论文，仅列举本篇论文（The home health care routing and scheduling ）的不同点吧，也可以说是考虑修改点

## 不同点

1.本篇HHC使用的的是单一服务和双重服务,双重服务里面又细分是同时进行还是按优先级进行

在AGH中，可以类比为本篇HHC中的多重服务，使用优先级来区分了同时进行和按先后顺序进行(同一优先级，同时进行，不同优先级，按先后顺序进行)

2.还有就是时间窗这里，本篇HHC这里的左时间窗处理和AGH中的处理是一样的，但右时间窗口不一样，HHC允许服务结束时间大于右时间窗口，我认为AGH的处理比这个更好，事先就用掩码机制限制了服务结束时间大于右时间窗口的车队不能进行服务

3.还有的最大的不同点就是，本篇HHC中工作人员是规定了一个工作人员可以提供3种服务，而我们的AGH中每个车队只提供一种服务，而且HHC中并没有容量约束

4.还有一些零星要注意的点，在section5.1中，患者被服务的时间是随机放置在10小时的日常规划周期内，在AGH中是考虑到24小时内，并且根据实际情况考虑的分布(而不是随机分布)

## 相同点

相同的地方就很多了，比如：

1.本篇HHC和AGH在问题路线上都是从指定的唯一一个起点出发，最后返回起点

2.都具有访问节点约束和时间窗口约束

3.都是多个工作者提供多种服务

4.要服务的节点都具有一定大的规模(20-300个1节点)



## 修改

在```for f in model.fleet_info['order']:```下就把对应车队能选择的节点筛选出来

意思是

```py
 for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            bat_cost = []
            # 初始化时间窗口
            bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
            bat_tw_right = bat['departure']
            # print(f"order:{model.fleet_info['order']} ")
            for f in model.fleet_info['order']:  # 按车队优先级顺序
```

在第一个for循环(划分数据批次)之前就将对应车队能选择的节点筛选出来

首先我做的是，生成了need变量

```py
service_to_select = torch.tensor([1, 2, 3, 4, 5])
need = service_to_select[torch.randint(0, 5, size=(num_samples, size))]
```



在rolout中

```py
def rollout(model, dataset, opts):
    """在贪婪模式下评估模型，计算每批次成本。
    - model: AttentionModel 实例
    - dataset: 数据集
    - opts: 配置选项
    - 返回: 成本张量
    """
    set_decode_type(model, "greedy")  # 设置为贪婪解码
    model.eval()  # 切换到评估模式

    if model.is_agh:
        cost = []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            bat_cost = []
            # 初始化时间窗口
            bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
            bat_tw_right = bat['departure']
            # 获取 need
            need = bat['need'].to(opts.device)  # 假设 need 在 bat 中，形状为 (batch_size, agh_size)

            for f in model.fleet_info['order']:  # 按车队优先级顺序
                # 构造车队输入
                next_duration = torch.tensor(
                    model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                    device=bat['type'].device).repeat(bat['loc'].size(0), 1)
                tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
                duration = torch.tensor(model.fleet_info['duration'][f],
                                       device=bat['type'].device).repeat(bat['loc'].size(0), 1)

                # 根据 need 过滤患者位置
                if f == 2 or f == 3:  # 车队2或3，需要考虑 need == 5
                    mask = (need == f) | (need == 5)
                else:  # 车队1或4，只访问 need == f
                    mask = (need == f)
                loc_filtered = bat['loc'] * mask.unsqueeze(-1)  # 仅保留需要访问的患者位置

                fleet_bat = {
                    'loc': loc_filtered,  # 使用过滤后的位置
                    'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
                    'duration': torch.gather(duration, 1, bat['type']),
                    'tw_right': tw_right,
                    'tw_left': tw_left,
                    'fleet': torch.full((bat['loc'].size(0), 1), f - 1)
                }
                if model.rnn_time:
                    model.pre_tw = None  # 重置 RNN 隐藏状态

                # 评估车队成本
                with torch.no_grad():
                    fleet_cost, _, serve_time = model(move_to(fleet_bat, opts.device))
                bat_cost.append(fleet_cost.data.cpu().view(-1, 1))

                # 更新时间窗口
                bat_tw_left[model.fleet_info['precedence'][f] + 1] = torch.max(
                    bat_tw_left[model.fleet_info['precedence'][f] + 1], serve_time[:, 1:])
            bat_cost = torch.cat(bat_cost, 1)  # [batch_size, 10]
            cost.append(bat_cost)
        return torch.cat(cost, 0)  # [dataset, 10]

    def eval_model_bat(batch):
        """评估非 AGH 批次成本"""
        with torch.no_grad():
            cost_, _ = model(move_to(batch, opts.device))
        return cost_.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)
```

核心代码为

```py

# 根据 need 过滤患者位置
if f == 2 or f == 3:  # 车队2或3，需要考虑 need == 5
    mask = (need == f) | (need == 5)
else:  # 车队1或4，只访问 need == f
    mask = (need == f)
loc_filtered = bat['loc'] * mask.unsqueeze(-1)  # 仅保留需要访问的患者位置

fleet_bat = {
    'loc': loc_filtered,  # 使用过滤后的位置
    'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
    'duration': torch.gather(duration, 1, bat['type']),
    'tw_right': tw_right,
    'tw_left': tw_left,
    'fleet': torch.full((bat['loc'].size(0), 1), f - 1)
}
```

这里巧妙的应用了掩码

通过将不需要访问的患者位置置为 [0, 0]，过滤掉不需要访问的患者，**同时保留张量形状，供模型处理**

但遇到了问题，这些[0,0]坐标可能会影响模型对仓库[0,0]的判断

提出三个解决方案

**方案1：使用无效坐标（如 [-1, -1]）**

方案2：使用掩码传递给模型

方案3：调整时间窗口

我更认同方案3，这是我没想到的，我为什么认为更好呢，因为直接将tw_right = 0，这样在之后时间窗口掩码这里就处理了这些不需要的节点，代码更改量不大，想法新颖

```py
def rollout(model, dataset, opts):
    """在贪婪模式下评估模型，计算每批次成本。
    - model: AttentionModel 实例
    - dataset: 数据集
    - opts: 配置选项
    - 返回: 成本张量
    """
    set_decode_type(model, "greedy")  # 设置为贪婪解码
    model.eval()  # 切换到评估模式

    if model.is_agh:
        cost = []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            bat_cost = []
            # 初始化时间窗口
            bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
            bat_tw_right = bat['departure']
            # 获取 need
            need = bat['need'].to(opts.device)  # 形状为 (batch_size, agh_size)

            for f in model.fleet_info['order']:  # 按车队优先级顺序
                # 构造车队输入
                next_duration = torch.tensor(
                    model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                    device=bat['type'].device).repeat(bat['loc'].size(0), 1)
                tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)
                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)
                duration = torch.tensor(model.fleet_info['duration'][f],
                                       device=bat['type'].device).repeat(bat['loc'].size(0), 1)

                # 根据 need 生成 mask
                if f == 2 or f == 3:
                    mask = (need == f) | (need == 5)
                else:
                    mask = (need == f)
                
                # 过滤时间窗口
                tw_right_filtered = tw_right * mask.float()  # 不需要访问的节点 tw_right 置为 0
                tw_left_filtered = tw_left * mask.float()    # 不需要访问的节点 tw_left 置为 0

                fleet_bat = {
                    'loc': bat['loc'],  # 保留原始位置
                    'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
                    'duration': torch.gather(duration, 1, bat['type']),
                    'tw_right': tw_right_filtered,
                    'tw_left': tw_left_filtered,
                    'fleet': torch.full((bat['loc'].size(0), 1), f - 1)
                }
                if model.rnn_time:
                    model.pre_tw = None  # 重置 RNN 隐藏状态

                # 评估车队成本
                with torch.no_grad():
                    fleet_cost, _, serve_time = model(move_to(fleet_bat, opts.device))
                bat_cost.append(fleet_cost.data.cpu().view(-1, 1))

                # 更新时间窗口
                bat_tw_left'][model.fleet_info['precedence'][f] + 1] = torch.max(
                    bat_tw_left[model.fleet_info['precedence'][f] + 1], serve_time[:, 1:])
            bat_cost = torch.cat(bat_cost, 1)  # [batch_size, 10]
            cost.append(bat_cost)
        return torch.cat(cost, 0)  # [dataset, 10]

    def eval_model_bat(batch):
        """评估非 AGH 批次成本"""
        with torch.no_grad():
            cost_, _ = model(move_to(batch, opts.device))
        return cost_.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)
```

在_init_embed中直接

```py
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
            mask = (input['tw_right'][:, 1:] > 0).float()  # 忽略 tw_right <= 0 的节点
            mask = torch.cat((torch.ones(input['tw_right'].size(0), 1, device=mask.device), mask), dim=1)  # 包含仓库
            init_embed = init_embed * mask.unsqueeze(-1)  # 屏蔽无效节点的位置嵌入
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
            init_embed[:, 1:, :] = init_embed[:, 1:, :] + fea_embed
            return init_embed
        # 这里移除了demand
        return self.init_embed(input)
```



移除demand和容量过后，紧接着带来了很多问题，因为模型的初始嵌入和一些前期的数据处理都和demand挂钩，改了很久，其中遇到最头疼的就是，当我把所有的demand该删的删改改的改后，发现有一些约束是在更深层次手demand的影响，比如

```py
def all_finished(self):
    # 检查是否所有路径都已完成（所有登机口已访问且步骤数足够）
    return self.i.item() >= self.demand.size(-1) and self.visited.all()  # 步骤数 >= 登机口数 且 所有节点已访问
```

这里判断所有的节点被访问和步骤数大于登机口数即可

但我们这里并不是单单看步骤数比较总节点数即可，每一次训练数据的大小都是不一样的，这里我们使用的是掩码来处理掉不需要的节点，所以单单判断总数是不可行的，这里也做了很多尝试，最终放弃

在探索的时候发现self.ids一直理解错误了，刚开始看self.tw_left[self.ids, selected]的时候，以为select是一个数值，想当然的认为self.ids也是一个数值了，其实打印self.ids可以看到，形状为[bath_size,1]，select也是一个储存了bath_size个数值，visited是三维的，

```py
self.tw_right[self.ids]
```

也是三维的

这样两者才好比较

```py
def all_finished(self):
    # 检查是否所有路径都已完成（所有登机口已访问且步骤数足够）
    mask = (self.tw_right[self.ids] != 0).int()
    flag = (mask==self.visited).all()  # [4]，True 表示该批次完成
    return flag
```

```
mask: tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]],

        [[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]], device='cuda:0', dtype=torch.int32),shape: torch.Size([4, 1, 51])
visited:tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]],

        [[1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]], device='cuda:0', dtype=torch.uint8),shape: torch.Size([4, 1, 51])
```



rollout处理好过后就是处理train_batch_agh

更深层次的都处理完了，这里也好办了照着rollout的思路删除了demand，加上了车队和节点之间的映射关系，并使用时间掩码来去除掉了不用访问的节点

现在最大的问题就是，当每个批次，节点的数量大小都不同，模型能不能很好的训练起来，还在等训练结果，紧张，不行的话，也可以改，在数据那里，但这个对不访问节点进行时间掩码，这里要改的话还是要设计很多文件代码，这就不好搞了



这个先放一边，现在来处理一下初始数据



先来看看论文中怎么写的

![image-20250619115032872](https://s2.loli.net/2025/06/19/2HW7PGOAvYxS3oR.png)

可以看到C(患者)分为C^s和C^d，V(医护人员)

论文作者设计了单个服务C^s和双重服务C^d

1.单个服务是指一个患者只需要一个医生就行，其中指定不同的医生

2.双重服务是指一个患者需要两种服务，这两种服务由不同的医生来提供，这里很不一样的就是，一个医生具备3种技能，可以提供三种不同的服务，而我们的代码项目中是一个车队只提供一种服务，还有个最大的不同点在于这里的双重服务里还分了同时服务和具有先后服务（先后服务的时间窗又不是毫无交集的，他们属于：下图可见）

3.服务时长都是2个小时，120分钟

<img src="https://s2.loli.net/2025/06/19/9JWUuXZLslyKa2H.jpg" alt="img" style="zoom:50%;" />

其中，地图范围：100个×100距离单位的区域内

**适当修改**

1.我想的是将双重服务中更细的分类直接删掉，沿用项目中的规则，使用优先级（两个服务可以同时进行，并且没有时间先后，车队到了就开始服务，两者之间没有约束）

2.删掉一个医生具备3种技能，可以提供三种不同的服务这个规定，为了简便，就只规定一个车队只提供一种服务



那接着看看代码项目中的数据吧

```py
n_gate, n_hour, n_min, prob = 91, 24, 60, np.load('problems/agh/arrival_prob.npy') # 登机口数、小时、分钟、到达概率
loc = 1 + np.random.choice(n_gate, size=(num_samples, size))  # 随机生成登机口索引 (1-91)
arrival = 60 * np.random.choice(n_hour, size=(num_samples, size),
                                p=prob) + np.random.randint(0, n_min, size=(num_samples, size))  # 到达时间
stay = torch.tensor([130, 124, 123]).repeat(num_samples, 1)  # 停留时间（根据类型）
type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)  # 节点类型 (0-2)
departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)  # 离开时间 = 到达 + 停留
service_to_select = torch.tensor([1, 2, 3, 4, 5])
need = service_to_select[torch.randint(0, 5, size=(num_samples, size))]
data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), need.tolist()))


SPEED = 110.0  # 车辆速度（单位：假设为公里/小时，用于时间计算）
NODE_SIZE = 92  # 总节点数（91 个登机口 + 1 个车库）


fleet_info= {'order': [1, 2, 3, 4],
             'precedence': {1: 0, 2: 1, 3: 1, 4: 2},
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 3: [35.0, 35.0, 36.0], 4: [30.0, 29.0, 27.0]},
             'next_duration': {2: [0.0, 0.0, 0.0],   1: [45.0, 44.0, 42.0], 0: [85.0, 87.0, 84.0]}}
```



最开始先设计最简单那种吧

1.V=4,即4种不同服务的车队

2.C=50,其中C^s=35,C^d=15

3.在100*100的距离范围内

4.服务时长设计成120分钟

5.一天设计工作时间为区间[10-20],共10小时，患者的时间窗口在这个时间段随机生成





在距离生成过程发现，本项目代码中1到0和0到1之间储存到距离值不同，是一个非对称的

我生成的是对称的100*100的矩阵，存入字典中



把所有数据改完过后，发现get_cost这里竟然出现"Time window violation"的报错



找到问题所在了，哈哈

我最先疑惑的就是为什么是problem_agh文件中的get_cost中检测到时间窗口有误，我想：在节点选择的时候get_mask就将时间窗口不合适的过滤了，为啥后面在计算奖励再一次检查的时候错了呢

思来想去，想到是get_cost中的速度speed改成了80，但get_mask中的代码并没有改，仍为110，这样的话，在get_mask中速度更大，花费的路程时间更短，能在时间窗口中，但在进一步计算奖励时，还要检验一次时间窗口是否正确，这里速度是80，那么路程时间就会更长，就很可能造成超出右时间窗口



果然不出所料，把get_mask中的speed改为80后就可以了，哈哈



记录一下数据生成部分

```py
# 随机生成数据
n_hour = np.arange(10, 20)
n_min = 60
n_gate, prob = (91, np.load('problems/agh/arrival_prob.npy')) # 登机口数、小时、分钟、到达概率
loc = 1 + np.random.choice(n_gate, size=(num_samples, size))  # 随机生成登机口索引 (1-91)
arrival = (60 * np.random.choice(n_hour, size=(num_samples, size),p=prob)
           + np.random.randint(0, n_min, size=(num_samples, size)))  # 到达时间
stay = torch.tensor([120, 120, 120]).repeat(num_samples, 1)  # 停留时间（根据类型）
type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)  # 节点类型 (0-2)
departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)  # 离开时间 = 到达 + 停留
# 按 7:3 比例生成 need
# service_to_select = torch.tensor([1, 2, 3, 4, 5])
# prob = torch.tensor([0.175, 0.175, 0.175, 0.175, 0.3])

service_to_select = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
prob = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1])
# 按照概率逐行采样
need = torch.stack([
    service_to_select[torch.multinomial(prob, size, replacement=True)]
    for _ in range(num_samples)
])# 生成的数据是按照这个概率来的，因为50较小，所有实际的比例大小会上下波动

data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), need.tolist()))
```



其中distance.pkl生成的是在100*100距离范围内对称的数据

```py
import numpy as np
import pickle

num_nodes = 92 # 1个车库加上91个节点
coords = np.random.rand(num_nodes, 2) * 100

dist_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

# 转成字典形式 {(i, j): distance}
distance_info = {}
for i in range(num_nodes):
    for j in range(num_nodes):
        distance_info[(i, j)] = dist_matrix[i, j]

# 保存字典到文件
with open('distance.pkl', 'wb') as f:
    pickle.dump(distance_info, f)


with open('distance.pkl', 'rb') as f:
    distance_info = pickle.load(f)
    print('fleet_info:', distance_info)
```



这里要好好讲讲feet_info的设计

```py
fleet_info= {'order': [1, 2, 3, 4, 5, 6, 7],
             'precedence': {1: 0, 2: 1, 3: 1, 4: 2, 5:3, 6:3, 7:1},
             'duration': {1: [30.0, 30.0, 30.0], 2: [36.0, 35.0, 35.0], 3: [35.0, 35.0, 36.0], 4: [30.0, 29.0, 27.0],
                          5: [30.0, 30.0, 28.0], 6: [30.0, 30.0, 28.0], 7: [28.0, 28.0, 26.0]},
             'next_duration': {3:[0.0, 0.0, 0.0],  2: [0.0, 0.0, 0.0],   1: [0.0, 0.0, 0.0], 0: [0.0, 0.0, 0.0]}}
```

写到最后才发现，一共就两种服务，这里的next_duration根本没有用，没有必要写，也懒得删掉了，可以为以后做拓展

主要来看这里```'precedence': {1: 0, 2: 1, 3: 1, 4: 2, 5:3, 6:3, 7:1},```

其中2，3是一个组合，4，5是一个组合，2，7是一个组合

我在代码中怎么实现这些组合的呢

```py
if f == 2:
    mask = (need == 2) | (need == 9) | (need == 11)
elif f == 3:
    mask = (need == 3) | (need == 9)
elif f == 5:
    mask = (need == 5) | (need == 10)
elif f == 6:
    mask = (need == 6) | (need == 10)
elif f == 8:
    mask = (need == 8) | (need == 11)
else:
    mask = (need == f)
```

```service_to_select = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])```患者一一对应其中一个值

我将2，3设为一个组合8，4，5设一个组合9，设2，7是一个组合10

因为总的代码是车队去选择要服务的节点，所以针对车队f，若need中需要f，则f选择改need的节点



**汇报的点**

1.论文中的车队数量和single服务和double服务，是按照7:3的比率去生成的，我是将双重服务中更细的分类直接删掉，沿用项目中的规则，使用优先级（两个服务可以同时进行，并且没有时间先后，车队到了就开始服务，两者之间没有约束）

2.规定的是一个车队只提供一种服务，设计的节点，例如：患者需要1，2，3，4，5这样的服务，1，2，3，4这个四个服务就对应1，2，3，4车队，5对应2，3车队的组合，为什么这样设计呢？是因为在代码是车队去选择要服务的节点，这样做的话，在选择时，规定需求为5的患者需要2，3车队的服务即可

3.数据生成上，我看之前agh的距离数据是非对称的，就是0=>1不等于1=>0，这里我没有找到这篇论文对应的代码，论文中也没有说，所以我设计的是对称的，还有就是论文中车队的行驶速度也没有说，我这里设计的是80

4.还有个很重要的点就是，single服务和double服务，是按照7:3的比率去生成的，但凑不成整数，所以实际是在7:3上下波动，这样的话每个车队服务的节点数量是不同的，不好设计输入的大小，不知道这里我有没有想错，所以我干脆直接输入的是整个节点，不同的是，该车队不会去服务的节点，我把他的左右时间窗口设计成的0，这样在之后的掩码机制会把这些点排除掉



<img src="https://s2.loli.net/2025/12/01/GU37W6qnYxEz5gA.png" alt="image-20250620164711115" style="zoom:50%;" />



## 进一步修改

汇报完后我，在双重服务这里处理得太简单，要加强复杂性

在增强双重服务复杂性的时候，突然发现一个超级大的错误，是之前没有注意到的：

这里就拿rollout来举例

```py
bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
```

bat_tw_left是在arrival的基础上复制len(model.fleet_info['next_duration']) + 1份得来的，也就是说arrival为[batch_size=64, graph_size=50],如果next_duration有三个优先级，那么bat_tw_left就为[4，batch_size=64, graph_size=50]，至于这里为什么要+1，是因为在之后处理bat_tw_left是递增的，是为了避免```model.fleet_info['precedence'][f]```超出范围

```py
tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
```

针对每一个车队f，tw_left的值是用f对应的优先级在bat_tw_left中取出来的

```py
bat_tw_left[model.fleet_info['precedence'][f] + 1] = torch.max(
                    bat_tw_left[model.fleet_info['precedence'][f] + 1], serve_time[:, 1:])
```

在之后遍历所有需要的节点后，bat_tw_left的更新方式是，直接更新f对应优先级的下一个优先级的bat_tw_left，这样全局的所有左时间窗后bat_tw_left都改变了，但在我们的问题中，我们只是想更新当前车队f需要服务的节点的左时间窗口



所以我门要做的就是，更新当前车队f需要服务的节点的左时间窗口

```py
# 获取下一个阶段的索引
next_stage = model.fleet_info['precedence'][f] + 1

# 当前阶段实际服务的节点 mask（跳过 depot，即从第 1 列开始）
real_mask = mask[:, 1:]  # [batch_size, graph_size]

# 旧的 tw_left 值
prev_tw_left = bat_tw_left[next_stage]  # [batch_size, graph_size]

# 当前 serve_time（从第 1 列开始）
cur_serve_time = serve_time[:, 1:]  # [batch_size, graph_size]

# 只更新被服务节点的时间窗
updated_tw_left = torch.where(real_mask, torch.max(prev_tw_left, cur_serve_time), prev_tw_left)

# 覆盖更新
bat_tw_left[next_stage] = updated_tw_left
```

在这里又有个小小的疑问，在agh_state中serve_time最开始是全0，更新完过后，那些没有被访问的节点的值是什么？

在 `update()` 中：

```py
serve_time = self.serve_time.scatter_(1, selected, cur_free_time.float())
```

这表示：

只有当前被选中访问的节点（`selected`）的 `serve_time` 会被更新为 `cur_free_time`

其它节点（即未被选中访问的节点）保持原样

所以，**未被访问的节点的 serve_time 依然是 0.0**

还有个问题就是，我们注意到的，`serve_time`包含仓库节点，bat_tw_left不包含，在最开始的代码中使用```serve_time[:, 1:]```从仓库之后的节点开始提取



最终设计的是6种车队提供6种服务：single服务1，2，3，4，5，6，double服务2，3组合，4，5组合，2，6组合，其中2，3组合，4，5组合具有优先级，2，6组合同步进行



**要注意的是要更改attention_model文件中嵌入输入的的大小为6**

## 对比实验

整个代码算是完了，现在要进行对比试验





同时处理多个临时表，内存占用太大，改进：

主要的内存优化在于将 `_apply_fleet_need_specific_constraints` 内部的逻辑从一次性处理整个批次改为**逐样本处理**。这意味着在任何给定时间点，GPU 上只保留一个样本的中间计算结果，显著降低了内存峰值。



场景一
        (fleet == 1) & ((need == 1) | (need == 9)),
        (fleet == 2) & ((need == 2) | (need == 7)),
        (fleet == 4) & ((need == 4) | (need == 8)),
        (fleet == 6) & (need == 6),
        (fleet == 3) & (need == 3),
        (fleet == 5) & (need == 5),

```
exceeds_tw_scenario1 = (
    (self.cur_free_time.expand(self.coords.size(0), self.coords.size(1) - 1) +  # 完成上一个服务后的时间
     self.distance.gather(1, distance_index) / self.SPEED)  # 加上旅行时间
    + self.duration[:, 1:]  # 加上服务持续时间
    > self.tw_right[:, 1:]
)[:, None, :]  # [batch_size, 1, graph_size]
```



 场景二

​	(fleet == 3) & (need == 7)) | ((fleet == 5) & (need == 8)

```
arrival_time = self.cur_free_time.expand(self.coords.size(0), self.coords.size(1) - 1) + \
               self.distance.gather(1, distance_index) / self.SPEED

tw_left = self.tw_left[:, 1:]
tw_right = self.tw_right[:, 1:]
duration = self.duration[:, 1:]

# 定义此场景的下限和上限
lower_bound_scenario2 = tw_left
upper_bound_scenario2 = tw_left + 30

# 如果早于tw_left，则强制拉到tw_left（即lower_bound_scenario2）
adjusted_arrival_time_scenario2 = torch.max(arrival_time, lower_bound_scenario2)

# 如果arrival_time超过upper_bound_scenario2，则标记为不可行
exceeds_upper_bound_scenario2 = (arrival_time > upper_bound_scenario2)[:, None, :]

# 如果adjusted_arrival_time + duration > tw_right，也标记为不可行
exceeds_tw_scenario2 = ((adjusted_arrival_time_scenario2 + duration) > tw_right)[:, None, :]

# 组合场景二中不可行的条件
mask_scenario2 = exceeds_upper_bound_scenario2 | exceeds_tw_scenario2
```

场景三

​	(fleet == 6) & (need == 9)

```
arrival_time = self.cur_free_time.expand(self.coords.size(0), self.coords.size(1) - 1) + \
               self.distance.gather(1, distance_index) / self.SPEED

tw_left = self.tw_left[:, 1:]
tw_right = self.tw_right[:, 1:] # 新增：需要tw_right来检查

# 如果早于或等于tw_left，则强制拉到tw_left
adjusted_arrival_time_scenario3 = torch.max(arrival_time, tw_left)

# 如果晚于tw_left，则标记为不可行 (条件1)
mask_condition1_scenario3 = (arrival_time > tw_left)[:, None, :]

# 如果调整后的时间加上服务持续时间超过tw_right，则标记为不可行 (条件2)
# 需要duration变量
duration = self.duration[:, 1:]
mask_condition2_scenario3 = ((adjusted_arrival_time_scenario3 + duration) > tw_right)[:, None, :]

# 组合两个不可行的条件
mask_scenario3 = mask_condition1_scenario3 | mask_condition2_scenario3
```



```py
import torch
import pickle
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

# 定义 StateAGH 类，使用 NamedTuple 存储 AGH 问题的状态
class StateAGH(NamedTuple):
    # 固定输入属性（来自问题实例的不可变数据）
    coords: torch.Tensor  # [batch_size, graph_size+1, 2]，车库（节点 0）+登机口的位置坐标
    tw_left: torch.Tensor  # [batch_size, graph_size+1]，时间窗口左边界（最早服务时间）
    tw_right: torch.Tensor  # [batch_size, graph_size+1]，时间窗口右边界（最晚服务时间）
    distance: torch.Tensor  # [batch_size, graph_size+1]，节点间距离矩阵
    duration: torch.Tensor  # [batch_size, graph_size+1]，每个节点的服务时长（车库为 0）
    need: torch.Tensor
    # 状态跟踪索引，用于支持束搜索（beam search）等场景，避免重复存储 coords 和 demand
    ids: torch.Tensor  # [batch_size, 1]，跟踪原始批次索引，用于索引固定数据行

    # 动态状态属性（随路径生成更新）
    prev_a: torch.Tensor  # [batch_size, 1]，上一个访问的节点
    # used_capacity: torch.Tensor  # [batch_size, 1]，已使用的车辆容量
    visited_: torch.Tensor  # [batch_size, 1, graph_size+1]，已访问节点掩码（uint8 或 int64）
    lengths: torch.Tensor  # [batch_size, 1]，当前路径的总距离
    cur_coord: torch.Tensor  # [batch_size, 1, 2]，当前节点的坐标
    cur_free_time: torch.Tensor  # [batch_size, 1]，当前可用时间（上一个节点服务完成后的时间）
    serve_time: torch.Tensor  # [batch_size, graph_size+1]，每个节点的服务开始时间
    tour: torch.Tensor  # [batch_size, steps]，当前路径（访问的节点序列）
    i: torch.Tensor  # [1]，当前步骤计数器
    fleet: torch.Tensor

    # 硬编码的常量
    VEHICLE_CAPACITY = 1.0  # 车辆容量，固定为 1.0
    SPEED = 80.0  # 车辆速度，单位为分钟（用于将距离转换为旅行时间）
    NODE_SIZE = 92  # 节点总数（graph_size + 1，通常为 91 个登机口 + 1 个车库）

    @property
    def visited(self):
        # 返回已访问节点的布尔掩码，处理不同数据类型（uint8 或 int64）
        if self.visited_.dtype == torch.uint8:
            return self.visited_  # 如果是 uint8，直接返回 visited_（0=未访问，1=已访问）
        else:
            # 如果是 int64，使用 mask_long2bool 转换为布尔掩码，n 为节点数（demand 的最后一个维度）
            return mask_long2bool(self.visited_, n=self.duration.size(-1)-1)

    def __getitem__(self, key):
        # 支持通过索引（张量或切片）获取子状态，保持 NamedTuple 结构
        assert torch.is_tensor(key) or isinstance(key, slice)  # 确保 key 是张量或切片
        # 返回新的 StateAGH 实例，所有张量按 key 索引
        return self._replace(
            ids=self.ids[key],  # 索引批次 ID
            prev_a=self.prev_a[key],  # 索引上一个节点
            # used_capacity=self.used_capacity[key],  # 索引已使用容量
            visited_=self.visited_[key],  # 索引已访问掩码
            lengths=self.lengths[key],  # 索引路径长度
            cur_coord=self.cur_coord[key],  # 索引当前坐标
            cur_free_time=self.cur_free_time[key],  # 索引当前可用时间
            tour=self.tour[key],  # 索引路径序列
            serve_time=self.serve_time[key],  # 索引服务时间
            fleet=self.fleet[key],  # 索引车队标号
            need=self.need[key],  # 索引需求量
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        # 初始化 StateAGH 状态，基于输入数据创建初始状态
        loc = input['loc']  # [batch_size, graph_size, 2]，登机口坐标
        tw_left, tw_right = input['tw_left'], input['tw_right']  # [batch_size, graph_size+1]，时间窗口
        batch_distance = input['distance']  # [batch_size, graph_size+1]，距离矩阵
        fleet = input['fleet']  # [batch_size, 1]，车队标号
        need = input['need']  # [batch_size, graph_size+1]，需求量


        batch_size, n_loc = loc.size()  # batch_size 和登机口数（graph_size）
        return StateAGH(
            # 固定输入
            fleet=fleet,  # [batch_size, 1]，车队标号
            need=need,
            coords=torch.cat((torch.zeros_like(loc[:, :1], device=loc.device), loc), dim=1),  # [batch_size, graph_size+1, 2]，添加车库坐标 (0,0)
            tw_left=tw_left,  # [batch_size, graph_size+1]，时间窗口左边界
            tw_right=tw_right,  # [batch_size, graph_size+1]，时间窗口右边界
            distance=batch_distance,  # [batch_size, graph_size+1]，距离矩阵
            duration=torch.cat((torch.zeros_like(loc[:, :1], device=loc.device), input['duration']), dim=1),  # [batch_size, graph_size+1]，服务时长（车库为 0）
            # 状态初始化
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # [batch_size, 1]，批次索引
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),  # [batch_size, 1]，初始上一个节点为车库 (0)
            visited_=(
                # 初始化已访问掩码，uint8 更节省内存，int64 支持更多节点
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                ) if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # [batch_size, 1, ceil(n_loc/64)]，int64 掩码
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),  # [batch_size, 1]，初始路径长度为 0
            cur_coord=torch.zeros_like(loc[:, :1], device=loc.device),  # [batch_size, 1, 2]，初始坐标为车库 (0,0)
            cur_free_time=torch.full((batch_size, 1), -60, device=loc.device),  # [batch_size, 1]，初始可用时间为 -60（占位符，车库）
            serve_time=torch.zeros(batch_size, n_loc+1, device=loc.device).float(),  # [batch_size, graph_size+1]，初始服务时间为 0
            tour=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),  # [batch_size, 1]，初始路径为空
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # [1]，初始步骤为 0
        )

    def get_final_cost(self):
        # 获取最终路径成本（未实现，留待束搜索等未来扩展）
        # TODO: 结合束搜索，未来完善
        pass


    def update(self, selected):
        # 更新状态，基于选择的节点（selected）更新路径和状态
        assert self.i.size(0) == 1, "Can only update if state represents single step"  # 确保当前状态只表示单步

        # 更新状态
        selected = selected[:, None]  # [batch_size, 1]，扩展维度以匹配状态
        prev_a = selected  # [batch_size, 1]，更新上一个节点为当前选择
        n_loc = self.duration.size(-1)-1  # 登机口数（不包括车库）

        # 计算路径长度
        cur_coord = self.coords[self.ids, selected]  # [batch_size, 1, 2]，当前节点的坐标
        distance_index = self.NODE_SIZE * self.coords[self.ids, self.prev_a] + cur_coord  # [batch_size, 1]，距离矩阵索引
        # print(f"distance_index = {distance_index},shape: {distance_index.shape}")
        lengths = self.lengths + self.distance.gather(1, distance_index)  # [batch_size, 1]，累加路径距离
        #
        # # 更新已使用容量
        # selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]  # [batch_size, 1]，当前节点的需求（车库需求为 0）
        # used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()  # [batch_size, 1]，非车库时累加需求，车库时重置为 0

        # 计算车队在该节点完成服务的时间
        cur_free_time = (torch.max(
            self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,  # 到达时间 = 上一个完成时间 + 旅行时间
            self.tw_left[self.ids, selected])  # 确保不小于时间窗口左边界
                         + self.duration[self.ids, selected]) * (prev_a != 0).float() - 60 * (prev_a == 0).float()  # [batch_size, 1]，非车库时加服务时长，车库时设为 -60

        # 计算车队在该节点开始服务的时间
        cur_start_time = torch.max(
            self.cur_free_time + self.distance.gather(1, distance_index) / self.SPEED,  # 到达时间 = 上一个完成时间 + 旅行时间
            self.tw_left[self.ids, selected])

        # 更新已访问掩码
        if self.visited_.dtype == torch.uint8:
            # 对于 uint8 掩码，直接使用 scatter 设置当前节点为已访问
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)  # [batch_size, 1, graph_size+1]
        else:
            # 对于 int64 掩码，使用 mask_long_scatter 设置（忽略车库，prev_a - 1）
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        # 更新服务时间
        serve_time = self.serve_time.scatter_(1, selected, cur_start_time.float())  # [batch_size, graph_size+1]，记录当前节点的服务时间

        # print(f"locations: {self.coords}")
        # print(f"update.tw_left: {self.tw_left[:4].int()},shape: {self.tw_left.shape}")
        # print(f"update.tw_right: {self.tw_right[:4].int()},shape: {self.tw_right.shape}")
        # print(f"sver_time: {serve_time[:4]},shape: {serve_time.shape}")
        # print(f"distance_time: {self.distance.gather(1, distance_index) / self.SPEED}")
        # print(f"duration: {self.duration[:4]},shape: {self.duration.shape}")
        # 更新路径序列
        tour = torch.cat((self.tour, selected), dim=1)  # [batch_size, steps+1]，添加当前节点到路径
        torch.set_printoptions(threshold=float('inf'), linewidth=1000)# 设置张量（Tensor）的打印选项，可有可无
        # print(f"tour: {tour[:4]},shape: {tour.shape}\n")
        # 返回更新后的 StateAGH 实例
        return self._replace(
            prev_a=prev_a,  # 更新上一个节点
            # used_capacity=used_capacity,  # 更新已使用容量
            visited_=visited_,  # 更新已访问掩码
            lengths=lengths,  # 更新路径长度
            cur_coord=cur_coord,  # 更新当前坐标
            i=self.i + 1,  # 步骤计数器加 1
            cur_free_time=cur_free_time,  # 更新当前可用时间
            tour=tour,  # 更新路径序列
            serve_time=serve_time  # 更新服务时间
        )


    def all_finished(self):
        # 检查是否所有路径都已完成（所有登机口已访问且步骤数足够）
        mask = (self.tw_right[self.ids] != 0).int()
        flag = (mask==self.visited).all()  # [4]，True 表示该批次完成
        # print(f"ids: {self.ids},shape: {self.ids.shape}")
        # print(f"mask: {mask},shape: {mask.shape}")
        # print(f"visited:{self.visited},shape: {mask.shape}")
        # print(f"flag: {flag},shape: {flag.shape}")
        return flag

    # def all_finished(self):
    #     # 检查是否所有路径都已完成（所有登机口已访问且步骤数足够）
    #     print(f"visited:{self.visited}")
    #     return self.i.item() >= self.duration.size(-1)-1 and self.visited.all()  # 步骤数 >= 登机口数 且 所有节点已访问

    def get_finished(self):
        # 检查每个批次是否完成（所有节点已访问）
        return self.visited.sum(-1) == self.visited.size(-1)  # [batch_size, 1]，True 表示该批次完成

    def get_current_node(self):
        # 获取当前节点（上一个访问的节点）
        return self.prev_a  # [batch_size, 1]

    def get_mask(self):
        """
        生成掩码，标记不可行节点（0=可行，1=不可行）
        - 形状: [batch_size, 1, graph_size+1]
        - 考虑已访问节点、时间窗口约束
        - 禁止连续两次访问车库，除非所有登机口已访问
        """
        # 1.处理已访问节点掩码
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, graph_size]，排除车库
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.duration.size(-1)-1)  # [batch_size, 1, graph_size]，转换为布尔掩码
        # print(f"visited_loc: {visited_loc[:4].int()},shape: {visited_loc.shape}")
        # 检查容量约束
        # exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)  # [batch_size, 1, graph_size]，需求超过容量
        # print(f"exceeds_cap: {exceeds_cap[:4].int()},shape: {exceeds_cap.shape}")
        # 计算距离索引
        pre_coord = self.coords[self.ids, self.prev_a]  # [batch_size, 1, 2]，上一个节点坐标
        all_coord = self.coords[:, 1:]  # [batch_size, graph_size, 2]，所有登机口坐标
        distance_index = self.NODE_SIZE * pre_coord.expand(self.coords.size(0), self.coords.size(1)-1) + all_coord  # [batch_size, graph_size]，距离矩阵索引

        # 检查时间窗口约束
        # exceeds_tw = ((torch.max(
        #     self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 完成上一个服务后的时间
        #     self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
        #     self.tw_left[:, 1:])  # 确保不早于时间窗口左边界
        #     + self.duration[:, 1:]) > self.tw_right[:, 1:])[:, None, :]  # [batch_size, 1, graph_size]，服务时间超过右边界
        # print(f"cur_free_time: {self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1)}")
        # print(f"time_distance: {self.distance.gather(1, distance_index) / self.SPEED}")
        # print(f"mask.tw_left: {self.tw_left[:, 1:].int()},shape: {self.tw_left.shape}")
        # print(f"mask.tw_right: {self.tw_right[:, 1:].int()},shape: {self.tw_right.shape}")
        # print(f"exceeds_tw: {exceeds_tw[:4].int()},shape: {exceeds_tw.shape}")
        # （完成上一个服务后的时间 + 到达该节点的距离/速度 + 服务时间 ）三者之和来判断是否大于右边界
        # 不可行节点 = 已访问 或 容量超限 或 时间窗口不可行

        # 2.处理时间窗口约束
        arrival_time = self.cur_free_time.expand(self.coords.size(0), self.coords.size(1) - 1) + \
                       self.distance.gather(1, distance_index) / self.SPEED

        tw_left = self.tw_left[:, 1:]
        tw_right = self.tw_right[:, 1:]
        duration = self.duration[:, 1:]

        # 设置上下界
        lower_bound = tw_left
        upper_bound = tw_left + 30

        # 如果早于 lower_bound，则强制拉到 lower_bound
        adjusted_arrival_time = torch.max(arrival_time, lower_bound)

        # 如果晚于 upper_bound，则设为不可行
        exceeds_upper_bound = (arrival_time > upper_bound)[:, None, :]  # 掩码

        # 如果 adjusted_arrival_time + duration > tw_right，也设为不可行
        exceeds_tw = ((adjusted_arrival_time + duration) > tw_right)[:, None, :]

        # 总掩码 = 已访问 或 超出时间窗口 或 超出允许到达范围
        mask_loc = visited_loc.to(exceeds_tw.dtype) | exceeds_tw | exceeds_upper_bound  # [batch_size, 1, graph_size]

        # 3.处理车库约束
        # 车库掩码：如果刚访问车库且仍有未访问登机口
        # 禁止再次访问车库，当刚访问车库（prev_a == 0）且仍有可访问登机口（mask_loc == 0 的节点数 > 0）时，车库被掩码
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size+1]，合并车库和登机口掩码

    def construct_solutions(self, actions):
        # 构建完整路径（未实现，留待束搜索等未来扩展）
        # TODO: 结合束搜索，未来完善
        pass
```



现在有三种情况的掩码，分为情况一，情况二，情况三

当fleet=1时，此时有两种情况，need=1，need=9

```
[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 9., 0., 0., 9., 0., 1., 0.,
 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.]
```



当fleet=3时，此时有两种情况，need=3，need=7

在fleet下，首先将need=7置0之后得到掩码，对左时间窗口和右时间窗口进行掩码置零过后，再进行时间窗口掩码操作执行情况一

之后将need=3置0之后得到掩码，对左时间窗口和右时间窗口进行掩码置零过后，再进行时间窗口掩码操作执行情况二

最后将这两个合并





当fleet=5时，此时有两种情况，need=5，need=8

在fleet下，首先将need=8置0之后得到掩码，对左时间窗口和右时间窗口进行掩码置零过后，再进行时间窗口掩码操作执行情况一

之后将need=5置0之后得到掩码，对左时间窗口和右时间窗口进行掩码置零过后，再进行时间窗口掩码操作执行情况二

最后将这两个合并



当fleet=1，此时有两种情况，need=1，need=9

就没有上面那么复杂，这两个直接执行情况一

当fleet=2，此时有两种情况，need=2，need=7

就没有上面那么复杂，这两个直接执行情况一

当fleet=4，此时有两种情况，need=4，need=8

就没有上面那么复杂，这两个直接执行情况一



当fleet=6，此时有两种情况，need=6，need=9

在fleet下，首先将need=9置0之后得到掩码，对左时间窗口和右时间窗口进行掩码置零过后，再进行时间窗口掩码操作执行情况一

之后将need=6置0之后得到掩码，对左时间窗口和右时间窗口进行掩码置零过后，再进行时间窗口掩码操作执行情况三

最后将这两个合并



成功，使用上述逻辑成功

总结一下，最开始是因为我直接建立了几个全部数据临时表，直接把内存给干崩溃了

经过这样的更改后get_mask最终如下

```py
 def get_mask(self):
        """
        生成掩码，标记不可行节点（0=可行，1=不可行）
        - 形状: [batch_size, 1, graph_size+1]
        - 考虑已访问节点、时间窗口约束
        - 禁止连续两次访问车库，除非所有登机口已访问
        """
        # 1.处理已访问节点掩码
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]  # [batch_size, 1, graph_size]，排除车库
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.duration.size(-1)-1)  # [batch_size, 1, graph_size]，转换为布尔掩码
        # print(f"visited_loc: {visited_loc[:4].int()},shape: {visited_loc.shape}")
        # 检查容量约束
        # exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)  # [batch_size, 1, graph_size]，需求超过容量
        # print(f"exceeds_cap: {exceeds_cap[:4].int()},shape: {exceeds_cap.shape}")
        # 计算距离索引
        pre_coord = self.coords[self.ids, self.prev_a]  # [batch_size, 1, 2]，上一个节点坐标
        all_coord = self.coords[:, 1:]  # [batch_size, graph_size, 2]，所有登机口坐标
        distance_index = self.NODE_SIZE * pre_coord.expand(self.coords.size(0), self.coords.size(1)-1) + all_coord  # [batch_size, graph_size]，距离矩阵索引

        # 获取必要张量
        arrival_time = torch.max(
            self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 完成上一个服务后的时间
            self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
            self.tw_left[:, 1:])
        tw_left_base = self.tw_left[:, 1:]
        tw_right_base = self.tw_right[:, 1:]
        duration_base = self.duration[:, 1:]

        # 初始化掩码
        mask_loc = torch.zeros_like(tw_left_base, dtype=torch.bool)[:, None, :]  # [batch_size, 1, graph_size]

        fleet_ids = self.fleet.squeeze(-1)+1  # [batch_size]
        # print(f"fleet_ids: {fleet_ids[:4]},shape: {fleet_ids.shape}")
        need_tensor = self.need  # [batch_size, graph_size+1]

        for fleet_type in [1, 2, 3, 4, 5, 6]:
            batch_mask = (fleet_ids == fleet_type)
            if not batch_mask.any():
                continue

            # 获取当前 fleet 下的 tw, duration
            tw_left = tw_left_base[batch_mask]
            tw_right = tw_right_base[batch_mask]
            duration = duration_base[batch_mask]
            arrival = arrival_time[batch_mask]
            need = need_tensor[batch_mask]
            # print(f"tw_left: {tw_left[:4]},shape: {tw_left.shape}")
            # print(f"tw_right: {tw_right[:4]},shape: {tw_right.shape}")
            # print(f"arrival: {arrival[:4]},shape: {arrival.shape}")
            # print(f"need: {need[:4]},shape: {need.shape}")

            # 默认 scenario1
            mask_fleet = (
                (arrival + duration > tw_right)[:, None, :]
            )

            if fleet_type in [3, 5]:
                # --- scenario1: need = 7 or 8 ---
                need_alt = need.clone()
                need_alt[(need_alt == (7 if fleet_type == 3 else 8))] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                tw_left_alt[(need_alt == 0)] = 0
                tw_right_alt[(need_alt == 0)] = 0

                mask1 = (
                    (arrival + duration > tw_right_alt)[:, None, :]
                )


                # --- scenario2: need = 3 or 5 ---
                need_alt = need.clone()
                need_alt[(need_alt == (3 if fleet_type == 3 else 5))] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                tw_left_alt[(need_alt == 0)] = 0
                tw_right_alt[(need_alt == 0)] = 0

                lower_bound = tw_left_alt
                upper_bound = tw_left_alt + 30
                adjusted_arrival = torch.max(arrival, lower_bound)
                mask2 = ((arrival > upper_bound)[:, None, :] |
                         ((adjusted_arrival + duration) > tw_right_alt)[:, None, :])

                mask_fleet = mask1 & mask2



            elif fleet_type == 6:
                # --- scenario1: need = 9 ---
                need_alt = need.clone()
                need_alt[need_alt == 9] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                tw_left_alt[need_alt == 0] = 0
                tw_right_alt[need_alt == 0] = 0

                mask1 = ((arrival + duration > tw_right_alt)[:, None, :])

                # --- scenario3: need = 6 ---
                need_alt = need.clone()
                need_alt[need_alt == 6] = 0
                tw_left_alt = tw_left.clone()
                tw_right_alt = tw_right.clone()
                duration_alt = duration.clone()
                tw_left_alt[need_alt == 0] = 0
                tw_right_alt[need_alt == 0] = 0

                adjusted_arrival = torch.max(arrival, tw_left_alt)
                mask_condition1 = (arrival > tw_left_alt)[:, None, :]
                mask_condition2 = ((adjusted_arrival + duration_alt) > tw_right_alt)[:, None, :]
                mask2 = mask_condition1 | mask_condition2

                mask_fleet = mask1 & mask2


            elif fleet_type in [1, 2, 4]:
                # fleet 1、2、4 直接使用场景一
                mask_fleet = ((arrival + duration > tw_right)[:, None, :])

            # 填入总掩码
            mask_loc[batch_mask] = mask_fleet

        # 总掩码 = 已访问 或 时间窗口不可行
        mask_loc = visited_loc.to(mask_loc.dtype) | mask_loc  # [batch_size, 1, graph_size]

        # 3.处理车库约束
        # 车库掩码：如果刚访问车库且仍有未访问登机口
        # 禁止再次访问车库，当刚访问车库（prev_a == 0）且仍有可访问登机口（mask_loc == 0 的节点数 > 0）时，车库被掩码
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)  # [batch_size, 1, graph_size+1]，合并车库和登机口掩码
```



在该这个过程当中发现，车队所有节点都遍历了，这肯定不对，找了半天原因，在这里

```py
arrival_time = self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) + self.distance.gather(1, distance_index) / self.SPEED
```

应该写成这样

```py
arrival_time = torch.max(
            self.cur_free_time.expand(self.coords.size(0), self.coords.size(1)-1) +  # 完成上一个服务后的时间
            self.distance.gather(1, distance_index) / self.SPEED,  # 加上旅行时间
            self.tw_left[:, 1:])
```

这里没有使用max，导致arrival_time初始值为负数，而我们在对其他不需要访问节点的处理时是将其右时间窗口置0，让后在掩码mask这里于arrival_time比较如果arrival_time>右时间窗口,则被掩码，但这里arrival_time<0,导致所有窗口都没有被掩码

呼～，这个问题从昨天一直搞到现在，今天刚考完英语，又被辅导员教育到下午3点开始，干到晚上10点，可算是解决了！！



## 预测eval

这里有贪婪采样greedy和随机采样sample

### 贪婪 vs 采样

这里再细细讲讲

**贪婪（Greedy）**：每次选择概率最高的那个动作/结果，结果确定且固定。

**采样（Sampling）**：根据概率分布随机选择下一个动作，概率越大，被选中的机会越高，但有一定随机性。

结合代码中具体的来

```
# sample: eval_batch_size=1, width=1000, decode_strategy=sample
# greedy: eval_batch_size=1000, decode_strategy=greedy (width=0)
```

这里的width代表路径的数量，

这里的策略是

针对一个问题（实例）

贪婪采样：针对这个问题只生成一条路径，每次选择概率最高的那个动作

随机采样：针对这个问题生成1000条路径，并从中选取最佳的路径，每条路径根据概率分布随机选择下一个动作



还要注意的是eval_batch_size=1000，这意思是代表贪婪采样解决问题时，一次性解决1000个问题，大大提升速度

### 类比：

- 采样模式：像你给每个样本试了1000种方案，但每次只能同时专心试一个样本。
- 贪婪模式：你一次性处理1000个样本，但每个样本只选最简单的1种方案。

### 结果

agh_50:

贪婪解码

![image-20250708151319148](https://s2.loli.net/2025/12/01/b2kJfB6dGjvc4AK.png)



采样解码









从上述结果可以看到

采样解码的效果要好165左右，花费时间更长

贪婪解码的效果略不好，但花费时间短



## 其他基线_AGH50

```py
 parser.add_argument('--val_method', type=str, default='farthest_insert', 
                     choices=['cws', 
                              'nearest_insert', 
                              'farthest_insert',
                              'random_insert', 
                              'nearest_neighbor', 
                              'sa'])
```

大概解释一下：

### `cws` (Clarke and Wright Savings)

**原理：** CWS 算法是解决车辆路径问题（VRP）的经典启发式算法。它的核心思想是计算将两个独立路径合并成一个路径所能节省的距离（或成本）。最初，每个客户点都有一条从仓库出发再回到仓库的独立路径。算法通过迭代地将节省最多的两个路径合并，直到无法再进行有效合并或满足车辆容量限制



看这个原理肯定是什么也看不懂的

其实这个算法就是围绕 合并和节省 这两个核心思想去做的

接着细细讲讲：



CWS 算法的出发点是一个非常低效的场景：

**初始状态：** 假设每个客户点都有一辆专门的车去服务它。这辆车从仓库出发，送到货后立刻返回仓库。

这样的做法会导致大量的车队资源被消耗，那就会想，我们可不可以A车去给A客户送东西的时候顺便就把离得近的B客户的东西给送了呢

这就是节省思想



合并是什么意思呢？举个例子

```
两个独立的客户 A 和 B，它们各自的路线是：
路线 A：仓库 → 客户 A → 仓库
路线 B：仓库 → 客户 B → 仓库
```

把这两条路线合并成一条

```
合并后的路线：仓库 → 客户 A → 客户 B → 仓库
```

这就是合并

那这样做**节省**的距离（或成本、时间）就是：

(A回仓库的距离+仓库到B的距离)−A到B的距离

**cws会将所有合并的情况列出来，根据节省的值进行合并**



这里在一个地方卡了很久

我们判断循环结束的依据是：

```py
def all_finished(self):
    # 检查是否所有路径都已完成（所有登机口已访问且步骤数足够）
    mask = (self.tw_right[self.ids] != 0).int()
    flag = (mask==self.visited).all()  # [4]，True 表示该批次完成
    return flag
```

已访问节点和右时间窗口不等于0的节点（也就是应访问的的节点）做比较来判断是否结束循环



在cws算法实现中，先按右时间窗口排序，把第一个节点默认为第一个要访问的节点，

但在我们的设置中，有些节点的右时间窗口为0，排序过后自然就在前面了，这样就会始终访问到不该访问的节点，visited始终会多出一个不该访问的节点，导致visited和mask相等不了，循环结束不了

这里做的改进就是

```py
# 策略：选择右时间窗口最小的客户点作为起始。
mask = tw_right != 0 # 过滤掉右时间窗为0的客户点
# 方法：将0替换为一个很大的值，这样排序后0会排在最后
tw_right_masked = tw_right.clone()
tw_right_masked[~mask] = float('inf')  # 使用 ~ 取反掩码，找出零值位置，并替换为一个很大的数，如1e9

_, selected = tw_right_masked.sort(dim=1) # 对时间窗右边界排序
selected = 1 + selected[:, 0] # 获取排序后第一个客户点（索引从1开始）

state = state.update(selected) # 更新状态，加入选定节点
sequences.append(selected) # 记录节点
```

将右时间窗口为0的值替换为无穷大，这样就不会排在前面被选进去了



val_size=1000的结果:

![image-20250708184645855](https://s2.loli.net/2025/12/01/G6x2Vyo4KpeOdXI.png)





### 插入

#### `nearest_insert` (最近插入)

​	**原理：** 这是一种构建启发式算法。它通常从一个空的路径（或只包含起点的路径）开始，然后逐步将未访问的节点插入到路径中，每次插入都选择能使路径成本增加最小的那个节点和插入位置

还是有大白话讲一下将可能被访问的节点依次进行插入，最终找到路径最短的路径(最近插入的最近，最近意思是指使得路径最短)

举个简单的例子：

```
仓库： [W]
未访问节点： {A, B}
```

第一次进行插入：

可能的选择是：

W->A，距离为3

或W->B，距离为4

3<4，选择W->A



第二次插入：

可能的选择是：

W->B->A,距离为6

或W->A->B，距离为3

3<6,所以选择W->A->B



在实际训练中，val_size和bath_size的关系

val_size代表要处理的问题实例的数量，bath_size代表，算法一次型处理多少实例，真正的循环次数等于val_size/bath_size



AGH_50

val_size=1000的结果

![image-20250709153021028](https://s2.loli.net/2025/12/01/zfbMSLXRBwiveGr.png)

val_size=100的结果

![image-20250709154009803](https://s2.loli.net/2025/12/01/zIJAN9Ft8EeDG5T.png)



**val_size越大，越接近真实情况**







#### `farthest_insert` (最远插入)

​	**原理：** 同样是一种构建启发式算法。与最近插入相反，它每次选择距离当前路径上所有已访问节点中最远的未访问节点进行插入。这样做的目的是让路径尽可能地“均匀”扩展，避免一开始就挤在某个区域



选择离当前路径**最远**的点来“扩展”路线，但它在插入这个点时，仍然会选择能使**当前路径总成本增加最少**的那个位置



还是有点难懂，比如有一条直线，此时有A，B两点，首先是计算A到直线的垂线距离为10，B到直线的垂线距离为20，此时会选择距离更远的B点

有可能要问了，这样不会导致最终下来走的路径最长吗，并不会，我们计算A到路径的距离时计算的是最短距离（这里是垂线段而不是弯弯曲曲的线段）

这里对比一下：



`nearest_insert` (最近插入)：通常能较快得到一个合理的初始解，但可能陷入局部最优。

相较于`nearest_insert` (最近插入)而言，`farthest_insert` (最远插入)有时能跳出局部最优，但仍是启发式，不保证全局最优。



AGH_50

val_size=100的结果

![image-20250709153433208](https://s2.loli.net/2025/12/01/LbugKDa3OiGUBAM.png)

val_size=1000的结果

![image-20250710094633214](https://s2.loli.net/2025/12/01/KHVGsjk82drlwoi.png)



可以看到100和1000得到的结果相差不大，为了节省预测时间可以选用100



#### `random_insert` (随机插入)

​	**原理：** 这也是一种构建启发式算法。它每次随机选择一个未访问的节点，并将其插入到当前路径的某个位置（通常是随机选择或最优选择）



val_size=100的结果

![image-20250709155534381](https://s2.loli.net/2025/12/01/HIQ2TZvLXGBVuRq.png)



以上理解肯定和实际有很大出入，我们的关注点不在这里，就大致了解一下





### `nearest_neighbor` (最近邻)

​	**原理：** 这是一种非常简单且快速的构建启发式算法。它从一个起始点开始，然后重复选择距离当前点最近的未访问点，直到所有点都被访问

这个好理解，举个例子：

从你当前所在的城市（最初是仓库），你环顾四周，找到所有你还没去过的城市中**离你当前位置最近的那个城市**

比如，如果你在城市 A，而你还没去过城市 B、C、D。如果 B 离 A 最近，那么下一步你就去 B

是一种典型的**贪婪算法**



val_size=1000的结果

![image-20250709160308495](https://s2.loli.net/2025/12/01/a7EYOVF5Hte9wNc.png)



### `sa` (Simulated Annealing，模拟退火)

​	**原理：** 模拟退火是一种元启发式算法，灵感来源于物理学中的退火过程（金属冷却，原子达到最低能量状态）。它从一个初始解开始，通过迭代地对当前解进行小幅度的扰动（生成邻域解），并以一定的概率接受比当前解更差的解。这个接受差解的概率会随着“温度”（T）的降低而减少，从而在前期允许更大的探索，后期则趋向于收敛

其实核心原理就一点，温度

**高温时 (T 大)：** 允许算法**大胆探索**，甚至可以以较高的概率接受“更差”的解（即目标函数值更大/更高的“能量”状态）。这有助于算法跳出局部最优

温度会慢慢降低，随着温度的降低

**低温时 (T 小)：** 算法变得**保守**，主要在当前解的附近进行**局部搜索**，接受“更差”解的概率大大降低，几乎只接受“更好”的解。这有助于算法收敛到最优解或局部最优解

val_size=1的结果

![image-20250709163244523](https://s2.loli.net/2025/12/01/utvC6yzlaBM7OJ8.png)



**val_size=200的结果**

![image-20250710091246772](https://s2.loli.net/2025/12/01/mZ4v2enRLK3IPa7.png)

可以看到只取一次实例得到的结果太具有偶然性了

我随便取200个中的几个结果：

```
[3605.4727840156706, 3616.9605836275796, 3021.338180891031, 3519.867430197459, 3349.768065577911, 3632.0716534408066, 3582.433868675217, 3335.853329674824, 3501.1945420558304, 3424.192845966682, 3641.4098729464945, 3849.076609771741, 3425.6033844919666, 3691.486041336502, 3331.619590817787, 3476.9821696854906]
```







### CPLEX

(IBM ILOG CPLEX Optimizer)





### LNS 

(Large Neighborhood Search - 大邻域搜索)





### LNS_SA 

(Large Neighborhood Search with Simulated Annealing - 结合模拟退火的大邻域搜索)





## AGH20

### ours

贪婪解码

![image-20250710141525662](https://s2.loli.net/2025/12/01/v4MVkERsFPAbX7O.png)





采样解码

![image-20250710141424361](https://s2.loli.net/2025/12/01/H3ebVchkTgXQrFB.png)



### `cws` (Clarke and Wright Savings)



### `nearest_neighbor` (最近邻)

![image-20250710160417511](https://s2.loli.net/2025/12/01/CMfTv9zr2RcxWLw.png)



### nearest_insert

AGH_20

val_size=200的结果

![image-20250710142014544](https://s2.loli.net/2025/12/01/tybafjHuxiIXUhL.png)



### farthest_insert

AGH_20

val_size=200的结果

![image-20250710142127430](https://s2.loli.net/2025/12/01/wWaA6BhPN93VCpv.png)



### random_insert

val_size=200的结果

![image-20250710142517328](https://s2.loli.net/2025/12/01/Vay4dFLp6Gn1mrl.png)



## `sa` (Simulated Annealing，模拟退火)

![image-20250710151405591](https://s2.loli.net/2025/12/01/Edwj17K38OQWLU2.png)





## AGH100

### ours

贪婪解码

![image-20250710173451354](https://s2.loli.net/2025/12/01/OSFonTRQPdA6g9j.png)



采样解码

![image-20250710174540063](https://s2.loli.net/2025/12/01/XSYIcGM16umE3v9.png)



### `cws` (Clarke and Wright Savings)

![image-20250710182141247](https://s2.loli.net/2025/12/01/MaPJ3wszvgRHSft.png)



### `nearest_neighbor` (最近邻)

![image-20250711093747765](https://s2.loli.net/2025/12/01/Idea6njhNASRiTp.png)





### nearest_insert

val_size=100的结果

![image-20250710184316972](https://s2.loli.net/2025/12/01/Jukmo4gvht5MSlC.png)



### farthest_insert

val_size=100的结果

![image-20250710184926160](https://s2.loli.net/2025/12/01/kMz6KDq4iRVwS98.png)

val_size=100的结果

### random_insert

![image-20250710185341716](https://s2.loli.net/2025/12/01/LZMAC9XRxkqclst.png)



###  `sa` (Simulated Annealing，模拟退火)

**val_size=200的结果**

![image-20250711091443356](https://s2.loli.net/2025/12/01/h4dxQeEBoIHSFGt.png)





| 方法名称                         | 类型                     |
| -------------------------------- | ------------------------ |
| CPLEX                            | 精确算法（混合整数规划） |
| LNS（Large Neighborhood Search） | 元启发式                 |
| SA（Simulated Annealing）        | 元启发式                 |
| GA（Genetic Algorithm）          | 元启发式                 |
| CWS（Clarke-Wright Savings）     | 启发式                   |
| Nearest Insertion                | 启发式                   |
| Farthest Insertion               | 启发式                   |
| Random Insertion                 | 启发式                   |
| Nearest Neighbor                 | 启发式                   |
| Ours (Greedy)                    | AI                       |
| Ours (Sampling)                  | AI                       |



## 原理详解

![image-20250805154525387](https://s2.loli.net/2025/12/01/4B8uTrVWvFxJmEM.png)

### **编码器 (Encoder) 模块详解**

### 1. **输入模块**

- 左上角输入

  ：包含三个部分

  - `δ^f`：航班j对操作f的需求量
  - `a^p`：时间窗开始时间
  - `b^p`：时间窗结束时间

### 2. **嵌入层 (Embedding)**

- **Concatenate操作**：将三个输入特征拼接
- **Linear Projection**：通过线性投影层处理拼接后的特征
- **位置嵌入**：右上方显示的是位置嵌入表，为每个登机口位置学习固定的嵌入向量

### 3. **特征融合**

- **⊕符号**：表示将线性投影结果与位置嵌入相加
- **Concatenate**：将所有节点的初始嵌入拼接成矩阵形式

### 4. **多层注意力处理**

- **MHA (Multi-Head Attention)**：8头注意力机制，学习节点间的关系
- **FF (Feed Forward)**：前馈神经网络，进一步处理特征
- **右侧显示**：经过N=3层处理后得到的最终节点嵌入h^(N)



## 损失函数和优化器

### 优化器

先讲讲优化器

优化器的是要和损失函数搭配起来使用的，它知道模型如何更新参数从而最小化损失函数

说白了，就是在我们训练模型时，就是朝着使损失函数最小的方向去更新参数，那朝着什么方向更新呢？更新多少呢？这就是优化器要做的事

贴一个大家常常看到的生动的举例：把模型想象成一个登山者，损失函数就是山谷，而登山者的目标就是找到山谷的最低点。优化器就是告诉登山者“下一步往**哪个方向走，走多远**”的策略



核心思想基本都是：**梯度下降**（Gradient Descent）

梯度就是参数相对于损失函数的偏导数向量



接下来介绍几个常见的优化器：

1.SGD，随机梯度下降

这个优化器的特点一个字就是就是，快！

快的原因就是它随机选取一小批样本来计算梯度，并更新参数，

然而这样带来的缺点也很明显就是不稳定，很容易震荡，



2.Momentum（动量）

这个优化器的提出就是为了解决SGD容易震荡的问题

引入了物理学中动量的概念：一个物体在运动时会保持其方向和速度，除非有足够大的力改变它

SGD容易震荡的原因就是参数更新的方向不定，左右摇摆，当加入动量时，就可以让左右摇摆的力量相互抵消掉

举个很生动的例子：你是一个从高山上下来的滑雪者，你看得到脚下的小斜坡，但还是会因为之前的滑行速度继续向前滑行，这样就不会为了脚下小小的偏差就立刻改变方向，从而顺利的滑向山脚



参数更新公式：
![image-20250807151031625](https://s2.loli.net/2025/12/01/CvxuniZDtjygWIE.png)

w是参数，阿尔法是学习率，v是速度



速度更新公式：

![image-20250807151233758](https://s2.loli.net/2025/12/01/BnYFS7tpymaH5Wk.png)

β：**动量系数**（Momentum Coefficient），是一个超参数，通常取值在 `[0.9, 0.99]` 之间

β 的值越大，表示我们越依赖于**过去的梯度方向**，即动量效果越强



这个公式是一个**指数加权移动平均**



3.AdaGrad（自适应梯度, Adaptive Gradient）

核心思想就是自适应学习率

每个参数有各自对应的学习率，对于频繁更新（梯度较大）的参数，其学习率会**逐渐减小**；对于不经常更新（梯度较小）的参数，其学习率会**保持或减小得更慢**



4.RMSprop（均方根传播）



RMSprop引入了一个新的变量 St，它代表梯度平方的移动平均值

公式

![image-20250807171422226](https://s2.loli.net/2025/12/01/qECJBdmGAV7X5jU.png)

这里还是采用了指数加权移动平均

ρ 是一个介于 `[0, 1]` 之间的超参数，通常设置为 `0.9` 左右。它决定了我们对**过去信息的保留程度**

- ρ 值越大，越依赖过去的梯度平方
- ρ 值越小，越看重当前的梯度平方

参数更新公式

![image-20250807172048834](https://s2.loli.net/2025/12/01/98ac6re4SoQdwDf.png)

看到这里，是不是觉得和动量优化器很像：

这里区分一下：

**动量（Momentum）**：

​	**应用对象**：对**梯度本身**进行移动平均，**来获得一个平滑的“速度”向量**

​	**解决的问题**：解决 SGD 在狭窄山谷中的**震荡问题**，让模型能够更平稳地沿着正确的方向前进

**RMSprop**：

​	**应用对象**：对**梯度的平方**进行移动平均，**来获得一个自适应的学习率**

​	**解决的问题**：解决 AdaGrad 的**学习率递减过快问题**，避免学习率在训练后期变得过小，导致模型无法继续学习



5.Adam（自适应矩估计, Adaptive Moment Estimation）

用一句话来概括 Adam 的核心思想：**Adam 结合了动量来优化更新的**方向***，同时结合了 RMSprop 来优化更新的***步长**（学习率）**









## 总结

我们将 `Ours(Sampling)` 的目标值 **2807.16** 作为参考最优值（Best Obj.），并用如下公式计算每个方法的 Gap：

<img src="https://s2.loli.net/2025/12/01/QfiL6APE4bUJldk.png" alt="image-20250710093935029" style="zoom:50%;" />

AGH_200

| Method              | Obj.        | Gap       | Time       |
| ------------------- | ----------- | --------- | ---------- |
| CWS                 | 11825.91    | 56.45%    | 3.87s      |
| Nearest Insertion   | 12455.32    | 64.79%    | 1719.57s   |
| Farthest Insertion  | 12821.75    | 69.58%    | 1877.88s   |
| Random Insertion    | 12913.99    | 70.90%    | 1142.61s   |
| Nearest Neighbor    | 11845.60    | 56.74%    | 2.87s      |
| SA                  | 11833.08    | 56.58%    | 2506.50s   |
| **Ours (Greedy)**   | **8168.61** | 8.10%     | **4.88s**  |
| **Ours (Sampling)** | **7559.24** | **0.00%** | **54.54s** |



AGH_300

| Method              | Obj.         | Gap       | Time       |
| ------------------- | ------------ | --------- | ---------- |
| CWS                 | 16696.87     | 62.05%    | 3.41s      |
| Nearest Insertion   | 17444.41     | 69.33%    | 5366.73s   |
| Farthest Insertion  | 17985.94     | 74.58%    | 9169.26s   |
| Random Insertion    | 17697.37     | 71.72%    | 6255.44s   |
| Nearest Neighbor    | 15790.44     | 53.21%    | 3.56s      |
| SA                  | 15782.23     | 53.14%    | 2699.62s   |
| **Ours (Greedy)**   | **11161.73** | 8.31%     | **5.71s**  |
| **Ours (Sampling)** | **10302.7**  | **0.00%** | **93.42s** |





|                     |             | AGH20     |           |             | AGH50     |            |             | AGH100    |            |
| ------------------- | ----------- | --------- | --------- | ----------- | --------- | ---------- | ----------- | --------- | ---------- |
| Method              | Obj.        | Gap       | Time      | Obj.        | Gap       | Time       | Obj.        | Gap       | Time       |
| SA                  | 1450.74 | 0.00% | 30min | 3202.59 | 15.95% | 30min |5448.33s|16.18%|30min|
| CWS                 | 1551.81 |  | 3.19s     | 3527.81     | 27.73%    | 4.57s      | 6470.46     | 38.01%    | 8.07s      |
| Nearest Insertion   | 1991.54     | 37.28% | 19.89s    | 4103.50     | 48.63%    | 579.67s    | 7083.87     | 51.09%    | 286.46s    |
| Farthest Insertion  | 1891.21     | 30.36% | 19.61s    | 3947.21     | 42.97%    | 62.74s     | 7076.28     | 50.93%    | 293.51s    |
| Random Insertion    | 2125.87     | 46.54% | 21.46s    | 4404.38     | 59.52%    | 43.26s     | 7472.61     | 59.34%    | 192.35s    |
| Nearest Neighbor    | 1614.58 | 143.19% | 4.57s     | 4191.31     | 51.75%    | 5.53s      | 7053.08     | 50.43%    | 8.73s    |
| SA                  | 1559.68     | 7.51% | 2186.72s  | 3599.76     | 30.30%    | 4481.02s   | 7032.01     | 49.95%    | 4566.01s   |
| **Ours (Greedy)**   | **1515.08** | **4.43%** | **2.99s** | **2965.80** | **7.38%** | **3.29s**  | **4977.11** | **6.13%** | **3.85s**  |
| **Ours (Sampling)** | **1481.84** | **2.14%** | **7.20s** | **2761.98** | **0.00%** | **12.45s** | **4689.55** | **0.00%** | **25.24s** |



|    |    | AGH200 |        |               |      AGH300    |                  |
| ------------------- | ----------- | ------ | ---------- | ------------ | ------ | ---------- |
| Method | Obj.        | Gap    | Time       | Obj.         | Gap    | Time       |
| SA                  | 9306.88 | 23.11% | 30min | 14128.06 | 37.18% | 30min |
| CWS                 | 11825.91    | 56.45% | 3.87s      | 16696.87     | 62.05% | 3.41s      |
| Nearest Insertion   | 12455.32    | 64.79% | 1719.57s   | 17444.41     | 69.33% | 5366.73s   |
| Farthest Insertion  | 12821.75    | 69.58% | 1877.88s   | 17985.94     | 74.58% | 9169.26s   |
| Random Insertion    | 12913.99    | 70.90% | 1142.61s   | 17697.37     | 71.72% | 6255.44s   |
| Nearest Neighbor    | 11845.60    | 56.74% | 2.87s      | 15790.44     | 53.21% | 3.56s      |
| SA                  | 11833.08    | 56.58% | 2506.50s   | 15782.23     | 53.14% | 2699.62s   |
| **Ours (Greedy)**   | **8168.61** | **8.10%** | **4.88s**  | **11161.73** | **8.31%** | **5.71s**  |
| **Ours (Sampling)** | **7559.24** | **0.00%** | **54.54s** | **10302.7**  | **0.00%** | **93.42s** |

agh_200     [1:31:44<1:31:44, 5504.64s/it]



agh_300

<img src="https://s2.loli.net/2025/12/01/VK65XxBUOZ19uR2.png" alt="image-20250727174700665" style="zoom: 50%;" />

仅展示车队一

<img src="https://s2.loli.net/2025/12/01/BR8yMg3OVkZdpzc.png" alt="image-20250727174743802" style="zoom:25%;" />



这里的**Ours (Sampling)**是按照val_size=50预测的

在让chatgpt帮我生成Gap时，脑袋里有个问题，他是文本生成式AI，应该不具有计算能力，哈哈,这里问了他一下

```
模型会学习到 “2 + 3 → 5” 这样的文本模式，而不是学习“2加3等于5”这个逻辑规则
所以我是在语言意义上“生成”结果，而非真正“计算”结果
因为我没有数学内核，也没有中间结果缓存
```



|                     |             | AGH20   |           |             | AGH50  |           |             | AGH100 |           |
| ------------------- | ----------- | ------- | --------- | ----------- | ------ | --------- | ----------- | ------ | --------- |
| CPLEX               | 1399.28     | 0.00%   | 14.52s    | 3018.56     | 13.12% | 1823.58s  | 6523.64     | 42.46% | 2109.80s  |
| GA                  | 1436.39     | 2.65%   | 131.34s   | 3393.17     | 27.14% | 273.46s   | 6523.69     | 42.46% | 541.56s   |
| LNS                 | 1560.50     | 11.54%  | 14.10s    | 3286.37     | 23.18% | 276.45s   | 6117.91     | 33.64% | 605.74s   |
| SA                  | 1515.81     | 8.34%   | 75.24s    | 3268.10     | 22.47% | 215.49s   | 6068.71     | 32.51% | 473.39s   |
| AVNS                | 1404.07     | 0.34%   | 38.73s    | 3168.53     | 18.74% | 92.79s    | 6519.97     | 42.42% | 278.62s   |
| CWS                 | 1551.80     | 10.90%  | 3.19s     | 3527.81     | 32.19% | 4.57s     | 6470.46     | 41.32% | 8.07s     |
| Nearest Insertion   | 1991.54     | 42.33%  | 0.62s     | 4103.50     | 53.74% | 1.87s     | 6820.46     | 48.90% | 3.41s     |
| Farthest Insertion  | 1891.21     | 35.13%  | 0.59s     | 3947.21     | 47.95% | 1.94s     | 6823.66     | 48.97% | 3.36s     |
| Random Insertion    | 2125.87     | 51.98%  | 0.44s     | 4404.38     | 65.05% | 1.66s     | 6958.03     | 51.94% | 2.72s     |
| Nearest Neighbor    | 3527.81     | 152.06% | 1.09s     | 4191.31     | 57.15% | 1.58s     | 6494.51     | 41.82% | 1.89s     |
| **Ours (Greedy)**   | **1463.45** | 4.59%   | **2.57s** | **2815.35** | 5.50%  | **2.81s** | **4930.17** | 7.63%  | **3.69s** |
| **Ours (Sampling)** | **1422.28** | 1.64%   | **2.28s** | **2668.65** | 0.00%  | **2.94s** | **4580.51** | 0.00%  | **3.68s** |





|                    |             | Gaussian  |           |              | Poisson   |           |
| ------------------ | ----------- | --------- | --------- | ------------ | --------- | --------- |
| Method             | Obj.        | Gap       | Time      | Obj.         | Gap       | Time      |
| CPLEX              | 6525.69 | 42.04% | 2523.52s | 6439.37 | 41.53% | 2279.30s |
| GA                 | 6633.98 | 44.38% | 560.11s | 6675.03 | 46.74% | 579.89s |
| LNS          | 6789.02 | 47.74% | 626.94s | 6063.39 | 33.27% | 631.17s |
| SA                 | 6383.67     | 38.88% | 500.72s     | 5477.72     | 20.37% | 475.81s |
| AVNS            | 6882.09 | 49.75% | 309.44s | 6586.46 | 44.74% | 316.13s |
| CWS                | 6265.49 | 36.35% | 19.00s   | 6238.95 | 37.07% | 18.79s |
| Nearest Insertion  | 6823.61 | 48.48% | 276.44s | 7121.03 | 56.50% | 288.22s |
| Farthest Insertion | 6829.65 | 48.61% | 287.71s | 7107.16 | 56.19% | 295.45s |
| Random Insertion   | 6968.14 | 51.62% | 187.37s | 7017.95 | 54.26% | 194.04s |
| Nearest Neighbor   | 6455.27 | 40.45% | 8.55s | 6435.93 | 41.45% | 9.56s |
| **Ours (Greedy)**  | **4975.03** | 8.26% | 3.58s       | **4923.51** | 8.20% | 3.18s   |
| **Ours (Sampling)** | **4595.73** | 0.00% | 3.84s       | **4550.24** | 0.00% | 3.47s   |

GA有问题，跑出来8113，参数是没有问题的，因为我又去跑了一下原始的数据（不是高斯和泊松的数据），是6523.69没错



cplex0923_orign.log是cplex跑的原始的数据（不是高斯和泊松的数据），测试一下更改了3过后的约束

|    |    | AGH200 |        |               |      AGH300    |                  |
| ------------------- | ----------- | ------ | ---------- | ------------ | ------ | ---------- |
| CPLEX               | 12455.16 | 67.22% | 4233.20s | 34215.14 | 240.65% | 7733.76s |
| GA                  | 14036.28 | 88.49% | 1120.54s | 23614.03 | 135.07% | 1802.96s |
| LNS                 | 11647.58 | 56.39% | 3771.44s | 17560.08 | 74.74% | 3836.43s |
| SA                  | 10093.13 | 35.48% | 2337.47s | 15828.15 | 57.61% | 2742.10s |
| AVNS                  | 10780.54 | 60.53% | 755.10s | 18683.19 | 78.85% | 948.90s |
| CWS                 | 11924.53   | 60.11% | 30.71s   | 16811.29   | 67.39% | 44.71s   |
| Nearest Insertion   | 11834.28   | 59.04% | 554.86s  | 17051.70   | 69.72% | 886.19s  |
| Farthest Insertion  | 12190.16   | 63.67% | 604.36s  | 17985.94   | 78.94% | 954.91s  |
| Random Insertion    | 11978.26   | 60.83% | 353.78s  | 16758.41   | 66.77% | 566.20s  |
| Nearest Neighbor    | 10757.03   | 44.40% | 14.84s   | 14409.77   | 43.43% | 20.43s   |
| **Ours (Greedy)**   | **8106.30** | 8.87% | 4.42s | **11052.73** | 10.01% | 5.75s |
| **Ours (Sampling)** | **7446.30** | 0.00% | 3.19s | **10047.09** | 0.00% | 4.04s |






|                     | fleet |         | 20    |       |         | 50     |       |         | 100    |       |         | 200    |       |          | 300    |       |
| :------------------ | :---- | :------ | :---- | :---- | :------ | :----- | :---- | :------ | :----- | :---- | :------ | :----- | :---- | :------- | :----- | :---- |
| **Ours (Greedy)**   | No    | 1473.52 | 3.60% | 2.80s | 3049.58 | 14.27% | 4.84s | 4653.17 | 7.42%  | 4.62s | 7577.14 | 9.05%  | 6.62s | 10257.49 | 10.28% | 9.38s |
| **Ours (Sampling)** | No    | 1426.62 | 0.31% | 2.96s | 2881.87 | 8.00%  | 4.42s | 4331.86 | 0.00%  | 4.93s | 6948.19 | 0.00%  | 7.11s | 9301.28  | 0.00%  | 6.62s |
| **Ours (Greedy)**   | Yes   | 1463.45 | 2.89% | 2.57s | 2815.35 | 5.50%  | 2.81s | 4930.17 | 13.82% | 3.69s | 8106.30 | 16.64% | 4.42s | 11052.73 | 18.82% | 5.75s |
| **Ours (Sampling)** | Yes   | 1422.28 | 0.00% | 2.28s | 2668.65 | 0.00%  | 2.94s | 4580.51 | 5.73%  | 3.68s | 7446.30 | 7.17%  | 3.19s | 10047.09 | 8.02%  | 4.04s |






|                    |         | 50->100 |         |
| ------------------ | :------ | ------- | ------- |
| Method             | Obj.    | Gap     | Time    |
| Nearest Insertion  | 6599.40 | 45.29%  | 395.87s |
| Farthest Insertion | 6760.36 | 48.91%  | 439.04s |
| Random Insertion   | 7008.56 | 54.32%  | 305.29s |
| Nearest Neighbor   | 6695.80 | 47.48%  | 14.32s  |
| **Ours(Greedy)**   | 5521.71 | 21.61%  | 135.05s |
| **Ours(Sampling)** | 4541.68 | 0.00%   | 131.22s |

 4150.63



**Ours(Greedy)**      Val_size:200  eval_batch_size:10

**Ours(Sampling)**    val_size:100   eval_batch_size:1   width:50





这次这个next_duration害得不惨，重新训练

老实了，把这个详细信息列出来

```
fleet_info= {'order': [1, 2, 3, 4, 5, 6],
             'precedence': {1: 2, 2: 0, 3: 1, 4: 0, 5: 1, 6: 3},
             'duration': {1: [30.0, 30.0, 30.0], 2: [30.0, 31.0, 31.0], 3: [34.0, 33.0, 32.0], 4: [29.0, 31.0, 30.0], 5:[32.0, 31.0, 30.0], 6: [30.0, 30.0, 30.0]},
             'next_duration': {   3: [0.0, 0.0, 0.0],   2: [0.0, 0.0, 0.0],   1: [0.0, 0.0, 0.0],   0: [60.0, 60.0, 60.0],}}
```

1.no fleet  output_50.log



## **随机车辆路径问题（Stochastic Vehicle Routing Problem, SVRP）**

传统VRP：所有客户信息（位置、需求量、时间窗口）**提前已知**，可以精确规划最优路径
现在面临的情况就是客户可能**随机出现**，需求量**可能波动**

这里记录一下别给忘了

这个是动态

![image-20250813180250630](https://s2.loli.net/2025/12/01/mpYj2IEydO7eTra.png)





这个是静态

![image-20250813180239212](https://s2.loli.net/2025/12/01/qw7mAubFJDR5Wi2.png)

静态的最终版是

![image-20251201231140317](https://s2.loli.net/2025/12/01/5TfbP612rwaSjkm.png)



## 论文

## 格式

I. 导论

II. 相关作品

III. 问题陈述

IV. 方法学

V. 实验

VI. 结论及未来工作



### III. 问题陈述

现在主要是来把公式全部写明白

```latex
\begin{align}
\text{min.} \quad & \sum_{i \in C_0} \sum_{j \in C_0} \sum_{v \in V} d_{ij} x_{ijv} \tag{1}
\end{align}
This objective function minimizes the total travel distance for all staff members, thereby minimizing total travel costs.


\begin{align}
\text{s.t.} \quad & \sum_{j \in C_0} x_{0jv} = \sum_{i \in C_0} x_{i0v} = 1, \quad \forall v \in V \tag{2}
\end{align}
Constraint (2) ensures that each staff member's route starts and ends at the central office (node 0).

\begin{align}
& \sum_{i \in C_0} x_{ijv} = \sum_{k \in C_0} x_{jkv}, \quad \forall j \in C, \forall v \in V \tag{3}
\end{align}
Constraint (3) is a flow balance condition. If a staff member visits a patient, they must also leave that location.

\begin{align}
& \sum_{v \in V} y_{isv} = N_s \cdot r_{is}, \quad \forall i \in C, \forall s \in S \tag{4}
\end{align}
Constraint (4) ensures that each patient's service demand is fully met by assigning the exact number of qualified staff members ($N_s$) required for that service.

\begin{align}
& y_{isv} \leq a_{vs}, \quad \forall i \in C, \forall s \in S, \forall v \in V \tag{5}
\end{align}
Constraint (5) links service assignments to staff qualifications. A staff member can only be assigned to a service they are qualified to perform.

\begin{align}
& y_{isv} \leq \sum_{j \in C_0} x_{ijv}, \quad \forall i \in C, \forall s \in S, \forall v \in V \tag{6}
\end{align}
Constraint (6) connects service provision to routing. A staff member can only provide a service if they actually visit the patient's location.


\begin{align}
& t_{isv} \geq e_i y_{isv}, \quad \forall i \in C, \forall s \in S, \forall v \in V \tag{7}
\end{align}
Constraint (7) stipulates that a service start time cannot be earlier than the patient's earliest acceptable time ($e_i$).

\begin{align}
& t_{isv} + p_{is} \leq l_i + M(1 - y_{isv}), \quad \forall i \in C, \forall s \in S, \forall v \in V \tag{8}
\end{align}
Constraint (8) ensures that the service completion time falls within the patient's time window.

\begin{align}
& t_{jsv'} \geq t_{isv} + p_{is} + d_{ij} - M(2 - x_{ijv} - y_{isv}), \notag \\
& \quad \forall i,j \in C, \forall s,s' \in S, \forall v,v' \in V \tag{9}
\end{align}
Constraint (9) ensures that service start times along a staff member's route increase with travel and service duration, while also preventing cycles.

\begin{align}
& |t_{is_1v_1} - t_{is_2v_2}| \leq M(2 - y_{is_1v_1} - y_{is_2v_2}), \notag \\
& \quad \forall i \in C^{\text{sim}}, \forall v_1,v_2 \in V \tag{10}
\end{align}
Constraint (10) addresses simultaneous services. It requires two different staff members providing a simultaneous service at the same location to start their services at roughly the same time.

\begin{align}
& \delta_i^{\min} y_{is_1v_1} \leq t_{is_2v_2} - t_{is_1v_1} \leq \delta_i^{\max} + M(2 - y_{is_1v_1} - y_{is_2v_2}), \notag \\
& \quad \forall i \in C^{\text{prec}}, \forall v_1,v_2 \in V \tag{11}
\end{align}
Constraint (11) addresses precedence-based services, ensuring the time between the start of the first and second services falls within a specified range.


\begin{align}
& x_{ijv}, y_{isv} \in \{0,1\}, \quad t_{isv}, w_{iv} \geq 0, \notag \\
& \quad \forall i,j \in C_0, \forall s \in S, \forall v \in V \tag{12}
\end{align}
Constraint (12) defines the variable types. The routing ($x_{ijv}$) and service assignment ($y_{isv}$) variables are binary, while the time variables ($t_{isv}$) are non-negative continuous.
```



### IV. 方法学

encoder和decoder

首先使用encoder将位置信息，左右时间窗口的信息编码，这个就是K，V了

之后给到decoder，decoder在此基础上加上fleet，当前节点和上一个节点完成的时间都嵌入，



这里要严格说明：decoder阶段使用了两次注意力机制

### 第一阶段：多头注意力更新上下文

```python
# _one_to_many_logits 函数中
# 1. 构造查询向量（包含当前节点嵌入 + 剩余容量/时间等约束信息）
query = fixed.context_node_projected + \
        self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state).float())

# 2. 多头注意力计算glimpse（上下文更新）
glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

# 3. 合并多头输出得到更新的上下文
glimpse = self.project_out(heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
```

**作用**:

- **输入**: h^(N)_c（当前上下文：当前节点嵌入 + 剩余容量/时间）
- **过程**: 通过MHA让上下文节点从所有飞行节点收集信息
- **输出**: h^(N+1)_c（更新后的上下文嵌入）

### 第二阶段：单头交叉注意力计算选择logits

```python
# 4. 使用更新后的上下文计算最终logits
final_Q = glimpse  # 这就是h^(N+1)_c
logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
```

**作用**:

- **输入**: h^(N+1)_c（更新后的上下文）和 logit_K（所有节点的键）
- **过程**: 单头交叉注意力计算每个节点的选择分数
- **输出**: 节点选择的logits分布



**这里要解释一下query, glimpse_K, glimpse_V, logit_K这些参数是什么意思**

在**第一阶段对话** (多头注意力)中

- **query**: "我现在在这个位置，还有这些约束，该关注什么？"
- **glimpse_K**: "我是这样的节点（身份标签）"
- **glimpse_V**: "如果你关注我，我能提供这些信息"

**结果**: 得到 `glimpse` - 综合所有节点信息后的更新理解

在**第二阶段对话** (单头选择)中

- **glimpse**: "基于刚才的分析，我现在的理解是..."
- **logit_K**: "选择我的话，收益是这样的"



还有个很重要的点没关注到，就是最后的单头交叉注意力机制是没有V的，为什么？

**标准注意力** (第一阶段):

```
output = Attention(Q,K,V) = softmax(QK^T)V
```

目的是**聚合信息**

**相似度计算** (第二阶段):

```
score = similarity(Q,K) = QK^T
```

目的是**评估匹配度**

只需要知道下一步操作是什么，不需要知道下一步操作对应的具体内容

举一个生动的例子：

这就像：

**第一阶段**：学生向各科老师收集信息

- Query: "我现在的学习状态如何？"
- Key: 各科目的识别标签
- Value: 各科目的具体建议和信息 
- 输出: 综合的学习建议

**第二阶段**：根据综合建议**选择下一步行动**

- Query: 综合的学习建议
- Key: 各个可选行动的特征
- Value: 不需要！只要知道哪个行动最匹配即可 
- 输出: 各行动的匹配分数



## DH和CH

**DH (Distance-based Heuristic) 本质上就是最近插入算法在家庭医疗护理调度问题中的应用**。两者在核心思想上是一致的：

1. 枚举所有可行的插入位置
2. 计算每个位置的距离增加成本
3. 选择成本增加最小的位置

**区别主要在于应用场景的复杂性：**

- **经典最近插入**：通常用于简单的TSP或VRP问题

- 论文中的DH

  ：应用于复杂的家庭医疗护理场景，需要考虑：

  - 多个护士
  - 时间窗约束
  - 护士技能等级
  - 多天重复访问
  - 护理连续性等额外约束



CH同样使用贪心插入启发式的框架（Algorithm 1），但与DH的区别在于成本计算函数`insertCost()`：

- **DH的insertCost**：计算插入患者后护士旅行距离的增加量
- **CH的insertCost**：计算插入患者后护士剩余服务容量的减少量（或者说，选择能保持最大剩余容量的插入方案）





## cplex核心约束

### 1. **服务指示变量与访问次数约束**

python

```python
# need=7/8/9需要两次访问，其他需要一次访问
required_visits = 2 if need_i in [7, 8, 9] else 1

# 链接服务指示变量y与到达边
y[i, f] == sum(x[j, i, k, f] for all j, k)

# 总访问次数必须满足需求
sum(y[i, f] for all f) == required_visits
```

### 2. **车辆流平衡约束**

python

```python
# 每辆车最多从起始depot出发一次
sum(x[depot_start, j, k, f]) <= 1

# 每辆车最多到达结束depot一次  
sum(x[i, depot_end, k, f]) <= 1

# 客户节点流平衡：进入边数 = 离开边数
sum(x[i, h, k, f]) == sum(x[h, j, k, f])
```

### 3. **时间连续性约束**

python

```python
# 如果车辆从i到j，那么j的开始时间 >= i的完成时间 + 旅行时间
t[j, f] >= t[i, f] + tau[i, j, f] - BIG_M * (1 - x[i, j, k, f])
```

### 4. **时间窗约束**

python

```python
# 硬时间窗：最早/最晚开始时间
t[i, f] >= start_early[i]
t[i, f] + service_duration <= start_late[i][f]

# 通用右时间窗
arrival + service_duration <= tw_right[f] + BIG_M * (1 - x[i,j,k,f])
```

## 特殊约束

### 5. **优先级约束（条件激活）**

python

```python
# 基于前驱映射：3->2, 5->4, 6->1
# 只有当两个车队都服务同一客户时才激活
precedence_mapping = {3: 2, 5: 4, 6: 1}
act = 2 - y[i, prev_fleet] - y[i, next_fleet]
t[i, next_fleet] >= t[i, prev_fleet] - BIG_M * act
```

### 6. **组合服务差分时间窗约束**

python

```python
# (2->3) need=7: 时间差必须在[10,30]分钟内
if f == 3 and target_need_j == 7:
    act = 2 - y[j, 2] - y[j, 3]
    t[j, 3] - t[j, 2] >= 10 - BIG_M * act
    t[j, 3] - t[j, 2] <= 30 + BIG_M * act

# (4->5) need=8: 时间差必须在[10,30]分钟内
# (1//6) need=9: 同时服务，时间差为0
```

### 7. **车队访问限制约束**

python

```python
# 根据need类型限制车队访问权限（可选择放宽）
if not relax_access:
    if not can_visit:
        x[i, j, k, f] == 0  # 强制不可访问的边为0
```

### 8. **最小服务间隔约束**

python

```python
MIN_SERVICE_INTERVAL = 120.0
t[j, f] >= t[i, f] + duration[i] + 120 - BIG_M * (1 - x[i, j, k, f])
```





/home/user/anaconda3/bin/conda run -p /home/user/anaconda3 --no-capture-output python /home/user/1gniT42e/HHC/cplex.py 
命令行参数:
{'filename': './data/agh/agh50_validation_seed4321.pkl',
 'graph_size': 50,
 'no_progress_bar': False,
 'offset': 0,
 'problem': 'agh',
 'relax_access': 'True',
 'seed': 1234,
 'val_method': 'cplex',
 'val_size': 1}
正在验证数据集: ./data/agh/agh50_validation_seed4321.pkl
  0%|                                                     | 0/1 [00:00<?, ?it/s]
--- 实例 1 开始求解 ---
CPLEX 求解完成!
求解状态: JobSolveStatus.FEASIBLE_SOLUTION
状态解释: 可行解
目标函数值: 2689.5129154634924
Gap: 0.013389661170526186
二进制变量数量: 477660
整数变量数量: 312
求解时间: 1805.75 秒
分支定界节点数: 1248
最佳界限: 2653.501248811681

路径信息:
车队 1:
  车辆 1: Depot -> 20 -> 14 -> 17 -> 13 -> Depot
  车辆 4: Depot -> 26 -> 34 -> 45 -> Depot
  车辆 9: Depot -> 1 -> 25 -> 21 -> 27 -> 8 -> Depot
  车辆 23: Depot -> 40 -> 12 -> 35 -> 50 -> Depot
  车辆 26: Depot -> 7 -> 49 -> 44 -> 6 -> Depot
  车辆 28: Depot -> 22 -> 47 -> 33 -> 48 -> Depot
车队 2:
  车辆 1: Depot -> 16 -> 18 -> 19 -> Depot
  车辆 14: Depot -> 26 -> 29 -> 28 -> Depot
车队 3:
  车辆 19: Depot -> 20 -> 14 -> 17 -> 13 -> Depot
  车辆 24: Depot -> 37 -> 2 -> 44 -> 6 -> Depot
车队 4:
  车辆 18: Depot -> 9 -> 43 -> 10 -> Depot
  车辆 30: Depot -> 16 -> 19 -> Depot
车队 5:
  车辆 2: Depot -> 39 -> 47 -> 5 -> Depot
  车辆 3: Depot -> 1 -> 25 -> 21 -> 27 -> 8 -> Depot
  车辆 4: Depot -> 9 -> 10 -> Depot
  车辆 22: Depot -> 46 -> 12 -> 35 -> 50 -> Depot
  车辆 23: Depot -> 22 -> 45 -> Depot
  车辆 27: Depot -> 37 -> 2 -> 36 -> Depot
  车辆 28: Depot -> 11 -> 24 -> 23 -> 32 -> 30 -> 3 -> Depot
  车辆 30: Depot -> 38 -> 31 -> 41 -> 15 -> Depot
车队 6:
  车辆 8: Depot -> 42 -> 24 -> 23 -> 32 -> 30 -> 3 -> Depot
  车辆 10: Depot -> 46 -> 29 -> 28 -> Depot
  车辆 29: Depot -> 40 -> 4 -> 36 -> Depot
  车辆 30: Depot -> 38 -> 31 -> 41 -> 15 -> Depot
--- 实例 1 求解结束，耗时 1807.18 秒 ---

✓ 实例 1 求解成功: 2689.5129154634924
100%|███████████████████████████████████████████| 1/1 [31:43<00:00, 1903.17s/it]

=============================

成功求解: 1/1
所有实例的成本列表: [2689.5129154634924]

结果统计:
   平均成本: 2689.51
   最小成本: 2689.51
   最大成本: 2689.51
>> 验证结束，总耗时 1903.19 秒

进程已结束，退出代码为 0















/home/user/anaconda3/bin/conda run -p /home/user/anaconda3 --no-capture-output python /home/user/1gniT42e/HHC/cplex.py 
命令行参数:
{'filename': './data/agh/agh50_validation_seed4321.pkl',
 'graph_size': 50,
 'no_progress_bar': False,
 'offset': 0,
 'problem': 'agh',
 'relax_access': 'True',
 'seed': 1234,
 'val_method': 'cplex',
 'val_size': 1}
正在验证数据集: ./data/agh/agh50_validation_seed4321.pkl
  0%|                                                     | 0/1 [00:00<?, ?it/s]
--- 实例 1 开始求解 ---
CPLEX 求解完成!
求解状态: JobSolveStatus.FEASIBLE_SOLUTION
状态解释: 可行解
目标函数值: 1833.1282273948818
Gap: 0.3228401798325015
二进制变量数量: 477660
整数变量数量: 312
求解时间: 1812.27 秒
分支定界节点数: 1961
最佳界限: 1241.320780806651

路径信息:
车队 1:
  车辆 13: Depot -> 42 -> 11 -> 40 -> 12 -> 2 -> 44 -> 32 -> 30 -> 6 -> 36 -> Depot
  车辆 17: Depot -> 34 -> 47 -> 45 -> 33 -> 5 -> 48 -> Depot
  车辆 23: Depot -> 1 -> 20 -> 25 -> 17 -> 13 -> Depot
车队 2:
  车辆 5: Depot -> 34 -> Depot
车队 3:
  车辆 8: Depot -> 46 -> Depot
  车辆 17: Depot -> 9 -> 43 -> 10 -> Depot
  车辆 27: Depot -> 14 -> 31 -> 41 -> 15 -> Depot
  车辆 29: Depot -> 26 -> 4 -> Depot
车队 4:
  车辆 14: Depot -> 42 -> 16 -> 18 -> 29 -> Depot
车队 5:
  车辆 5: Depot -> 7 -> 20 -> 14 -> 38 -> 15 -> Depot
  车辆 17: Depot -> 39 -> 37 -> 24 -> 35 -> 50 -> Depot
  车辆 18: Depot -> 22 -> 12 -> 23 -> 2 -> 44 -> 8 -> 5 -> Depot
  车辆 26: Depot -> 10 -> 28 -> 19 -> 4 -> Depot
车队 6:
  车辆 7: Depot -> 26 -> 46 -> Depot
  车辆 10: Depot -> 7 -> 49 -> 21 -> 27 -> 3 -> Depot
--- 实例 1 求解结束，耗时 1813.42 秒 ---

✓ 实例 1 求解成功: 1833.1282273948818
100%|███████████████████████████████████████████| 1/1 [31:16<00:00, 1876.77s/it]

============================================================

所有实例求解完成!
============================================================
成功求解: 1/1
所有实例的成本列表: [1833.1282273948818]

结果统计:
   平均成本: 1833.13
   最小成本: 1833.13
   最大成本: 1833.13
>> 验证结束，总耗时 1876.79 秒

进程已结束，退出代码为 0











/home/user/anaconda3/bin/conda run -p /home/user/anaconda3 --no-capture-output python /home/user/1gniT42e/HHC/cplex_solver.py 
命令行参数:
{'filename': './data/agh/agh50_validation_seed4321.pkl',
 'graph_size': 50,
 'no_progress_bar': False,
 'offset': 0,
 'problem': 'agh',
 'relax_access': 'True',
 'seed': 1234,
 'val_method': 'cplex',
 'val_size': 1}
正在验证数据集: ./data/agh/agh50_validation_seed4321.pkl
  0%|                                                     | 0/1 [00:00<?, ?it/s]
--- 实例 1 开始求解 ---
CPLEX 求解完成!
求解状态: JobSolveStatus.OPTIMAL_SOLUTION
状态解释: 最优解
目标函数值: 1239.7389462166398
Gap: 0.0
二进制变量数量: 477660
整数变量数量: 312
求解时间: 35.72 秒
分支定界节点数: 0
最佳界限: 1239.7389462166398

路径信息:
车队 1:
  车辆 1: Depot -> 22 -> 39 -> 7 -> 49 -> 21 -> 27 -> 15 -> Depot
车队 2: 未使用
车队 3: 未使用
车队 4:
  车辆 7: Depot -> 9 -> 43 -> 10 -> Depot
车队 5:
  车辆 30: Depot -> 42 -> 11 -> 40 -> 12 -> 23 -> 2 -> 44 -> 8 -> 48 -> 5 -> Depot
车队 6:
  车辆 1: Depot -> 42 -> 16 -> 18 -> 29 -> Depot
  车辆 20: Depot -> 14 -> 31 -> 41 -> 17 -> 10 -> Depot
  车辆 30: Depot -> 34 -> 12 -> 35 -> 2 -> 44 -> 32 -> 30 -> 6 -> 36 -> 5 -> 33 -> 45 -> Depot
--- 实例 1 求解结束，耗时 36.80 秒 ---

✓ 实例 1 求解成功: 1239.7389462166398
100%|█████████████████████████████████████████████| 1/1 [01:28<00:00, 88.39s/it]


============================================================
成功求解: 1/1
所有实例的成本列表: [1239.7389462166398]

结果统计:
   平均成本: 1239.74
   最小成本: 1239.74
   最大成本: 1239.74
>> 验证结束，总耗时 88.41 秒

进程已结束，退出代码为 0







  车辆 9: Depot -> 18 -> 9 -> 14 -> 1 -> 7 -> 20 -> 15 -> 13 -> 10 -> Depot



车队 5:
  车辆 1: Depot -> 14 -> 15 -> 13 -> Depot
  车辆 2: Depot -> 2 -> 3 -> 11 -> Depot
  车辆 8: Depot -> 9 -> 17 -> 7 -> 8 -> Depot





```
parser.add_argument('--pop_size', type=int, default=300, help='种群大小')
parser.add_argument('--cx_pb', type=float, default=0.8, help='交叉概率')
parser.add_argument('--mut_pb', type=float, default=0.2, help='变异概率')
parser.add_argument('--n_gen', type=int, default=4500, help='进化代数')
```



约束11

related work

数据更新

时间分布







p1

```
{
    1: '4', 2: '2&3', 3: '1', 4: '1&6', 5: '1&6', 6: '3', 7: '1&6', 8: '1', 9: '1', 10: '2&3',
    11: '2', 12: '2&3', 13: '6', 14: '1&6', 15: '1&6', 16: '1', 17: '2', 18: '6', 19: '1', 20: '4&5',
    21: '4', 22: '2', 23: '4', 24: '2', 25: '5', 26: '2&3', 27: '4', 28: '1', 29: '5', 30: '4',
    31: '1', 32: '5', 33: '4', 34: '4&5', 35: '3', 36: '5', 37: '1', 38: '6', 39: '6', 40: '4',
    41: '5', 42: '4&5', 43: '2', 44: '2&3', 45: '3', 46: '2&3', 47: '4', 48: '4', 49: '5', 50: '6'
}
```

```
[ 0, 49,  0, 20, 29, 36,  0, 25, 41,  0, 34, 32,  0, 42,  0]
```

```
[0, 20, 49, 29, 36, 0, 42, 25, 41, 0, 34, 32, 0]
```





/home/user/anaconda3/bin/conda run -p /home/user/anaconda3 --no-capture-output python /home/user/1gniT42e/HHC/eval_routes.py 
调用AttentionModel
  [*] Loading model from ./data/50/epoch-99.pt
  0%|                                                    | 0/10 [00:00<?, ?it/s]
Node-Service Mapping:
{1: '4', 2: '2&3', 3: '1', 4: '1&6', 5: '1&6', 6: '3', 7: '1&6', 8: '1', 9: '1', 10: '2&3', 11: '2', 12: '2&3', 13: '6', 14: '1&6', 15: '1&6', 16: '1', 17: '2', 18: '6', 19: '1', 20: '4&5', 21: '4', 22: '2', 23: '4', 24: '2', 25: '5', 26: '2&3', 27: '4', 28: '1', 29: '5', 30: '4', 31: '1', 32: '5', 33: '4', 34: '4&5', 35: '3', 36: '5', 37: '1', 38: '6', 39: '6', 40: '4', 41: '5', 42: '4&5', 43: '2', 44: '2&3', 45: '3', 46: '2&3', 47: '4', 48: '4', 49: '5', 50: '6'}
tour:tensor([[37,  5,  3,  0, 16, 31,  8, 19,  0, 14, 15,  4,  0,  9,  7, 28]], device='cuda:0')
tour:tensor([[11,  2, 17, 44,  0, 22, 12, 24, 43, 10,  0, 46, 26,  0,  0]], device='cuda:0')
tour:tensor([[12, 35, 45,  0, 46, 26, 10,  0,  2,  6,  0, 44,  0]], device='cuda:0')
tour:tensor([[40, 47, 34, 23, 30,  0, 20, 21, 27,  0, 42,  1, 33, 48,  0,  0]], device='cuda:0')
tour:tensor([[20, 49, 29, 36,  0, 42, 25, 41,  0, 34, 32,  0]], device='cuda:0')
tour:tensor([[ 7, 38, 18, 50,  0, 39, 14, 15,  0,  5, 13,  0,  4,  0,  0]], device='cuda:0')
 10%|████▍                                       | 1/10 [00:00<00:07,  1.26it/s]
Node-Service Mapping:
{1: '2&3', 2: '3', 3: '1', 4: '1&6', 5: '4&5', 6: '2&3', 7: '2&3', 8: '5', 9: '2', 10: '2', 11: '5', 12: '4', 13: '5', 14: '6', 15: '3', 16: '1&6', 17: '4&5', 18: '2&3', 19: '2', 20: '2', 21: '1', 22: '2&3', 23: '2', 24: '2&3', 25: '5', 26: '1', 27: '3', 28: '2', 29: '2&3', 30: '2', 31: '5', 32: '6', 33: '2', 34: '5', 35: '2&3', 36: '2', 37: '5', 38: '4', 39: '4&5', 40: '1&6', 41: '6', 42: '6', 43: '3', 44: '6', 45: '2&3', 46: '1', 47: '5', 48: '1&6', 49: '6', 50: '2'}
tour:tensor([[46, 16, 40, 48,  3,  0, 26,  0, 21,  4,  0]], device='cuda:0')
tour:tensor([[28, 23, 50, 20, 10,  0, 22,  9,  7, 19, 33,  0, 35, 45, 36,  1, 29,  0,  6, 18, 24,  0, 30,  0]], device='cuda:0')
tour:tensor([[ 2, 22,  7, 24,  0, 35, 45,  1, 27,  0, 43,  6, 18, 29,  0, 15,  0]], device='cuda:0')
tour:tensor([[38, 12,  0,  5, 17, 39,  0]], device='cuda:0')
tour:tensor([[37, 25,  8, 34,  0, 31, 13, 11, 17, 39,  0, 47,  0,  5]], device='cuda:0')
tour:tensor([[49, 14, 32, 42,  0, 41,  4, 44,  0, 16, 40, 48,  0]], device='cuda:0')
 20%|████████▊                                   | 2/10 [00:01<00:03,  2.16it/s]
Node-Service Mapping:
{1: '1', 2: '1&6', 3: '2&3', 4: '4&5', 5: '1', 6: '6', 7: '6', 8: '5', 9: '4', 10: '2&3', 11: '2', 12: '1', 13: '3', 14: '2&3', 15: '5', 16: '1&6', 17: '4', 18: '5', 19: '5', 20: '2', 21: '1', 22: '5', 23: '3', 24: '2', 25: '6', 26: '2&3', 27: '2', 28: '4&5', 29: '2&3', 30: '5', 31: '2&3', 32: '6', 33: '2&3', 34: '2&3', 35: '3', 36: '1', 37: '6', 38: '4&5', 39: '2', 40: '3', 41: '4&5', 42: '6', 43: '2', 44: '6', 45: '1', 46: '5', 47: '5', 48: '3', 49: '3', 50: '1'}
tour:tensor([[21, 50,  1,  0, 12,  2, 36, 16,  0, 45,  0,  5,  0]], device='cuda:0')
tour:tensor([[27, 11, 14, 20, 24, 39, 10,  0, 31, 33,  0, 34,  3, 43, 26,  0, 29,  0,  0,  0]], device='cuda:0')
tour:tensor([[31, 40, 33,  0, 34,  3, 35,  0, 14, 49,  0, 29, 48, 26, 23, 13,  0, 10,  0]], device='cuda:0')
tour:tensor([[ 4, 28, 17,  0,  9, 41,  0, 38]], device='cuda:0')
tour:tensor([[ 4, 19, 28,  0, 22, 41, 30,  0, 18,  8, 38, 47,  0, 15, 46,  0]], device='cuda:0')
tour:tensor([[25,  7, 32, 42,  0,  6, 44,  2, 16,  0, 37,  0]], device='cuda:0')
 30%|█████████████▏                              | 3/10 [00:01<00:02,  2.78it/s]
Node-Service Mapping:
{1: '2&3', 2: '1', 3: '4', 4: '1', 5: '6', 6: '2', 7: '6', 8: '5', 9: '5', 10: '6', 11: '5', 12: '2&3', 13: '2', 14: '4', 15: '2', 16: '1', 17: '1', 18: '2&3', 19: '1', 20: '2&3', 21: '4', 22: '3', 23: '2&3', 24: '4', 25: '3', 26: '6', 27: '4', 28: '6', 29: '2', 30: '5', 31: '1&6', 32: '5', 33: '6', 34: '2&3', 35: '5', 36: '4', 37: '4', 38: '1&6', 39: '3', 40: '2&3', 41: '1', 42: '4', 43: '3', 44: '5', 45: '4&5', 46: '1', 47: '5', 48: '2', 49: '4', 50: '4'}
tour:tensor([[ 4, 41,  2, 19,  0, 17, 16, 46,  0, 38,  0, 31]], device='cuda:0')
tour:tensor([[40, 12, 34, 18, 15,  0, 23,  1,  0, 29, 20, 13,  0,  6, 48,  0]], device='cuda:0')
tour:tensor([[40, 12, 34, 18,  1,  0, 25, 20, 39,  0, 23, 22, 43,  0,  0]], device='cuda:0')
tour:tensor([[49, 36,  0, 37, 50, 45, 42, 27, 14,  0,  3, 21,  0, 24,  0]], device='cuda:0')
tour:tensor([[47,  9, 30, 44,  0, 35, 32,  0,  8, 11,  0, 45,  0]], device='cuda:0')
tour:tensor([[10, 28, 26, 33,  0, 38,  5,  0, 31,  7,  0]], device='cuda:0')
 40%|█████████████████▌                          | 4/10 [00:01<00:01,  3.29it/s]
Node-Service Mapping:
{1: '4&5', 2: '4&5', 3: '2', 4: '4&5', 5: '5', 6: '4&5', 7: '4', 8: '1&6', 9: '3', 10: '2', 11: '1', 12: '1', 13: '4&5', 14: '4', 15: '4', 16: '3', 17: '1&6', 18: '1', 19: '1', 20: '5', 21: '4&5', 22: '1&6', 23: '4&5', 24: '3', 25: '3', 26: '6', 27: '2', 28: '1', 29: '1&6', 30: '5', 31: '6', 32: '4&5', 33: '4', 34: '2', 35: '5', 36: '1&6', 37: '5', 38: '1', 39: '2', 40: '1&6', 41: '4', 42: '6', 43: '1', 44: '5', 45: '2', 46: '2', 47: '2&3', 48: '4&5', 49: '4', 50: '2&3'}
tour:tensor([[12, 11, 29, 43,  0, 36, 18,  8, 22,  0, 28, 17, 40, 38, 19,  0,  0,  0]], device='cuda:0')
tour:tensor([[27, 45, 34,  3,  0, 39, 46, 50,  0, 47, 10,  0,  0]], device='cuda:0')
tour:tensor([[16, 24, 50, 25,  0,  9, 47,  0]], device='cuda:0')
tour:tensor([[48, 32, 23,  0, 41, 14, 21,  0, 15, 49, 33,  0,  1, 13,  6,  7,  4,  0,  2,  0,  0,  0]], device='cuda:0')
tour:tensor([[48, 37, 32, 23, 20,  0,  5,  2,  0, 35, 30,  4,  0, 44,  1, 13,  6, 21,  0,  0]], device='cuda:0')
tour:tensor([[26, 22,  0, 36, 42, 40,  0, 31,  8,  0, 29,  0, 17,  0]], device='cuda:0')
 50%|██████████████████████                      | 5/10 [00:01<00:01,  3.52it/s]
Node-Service Mapping:
{1: '4', 2: '2&3', 3: '1', 4: '1&6', 5: '2&3', 6: '4&5', 7: '2', 8: '4&5', 9: '3', 10: '4&5', 11: '1&6', 12: '4&5', 13: '4', 14: '1', 15: '2', 16: '6', 17: '3', 18: '4', 19: '1&6', 20: '5', 21: '5', 22: '4&5', 23: '2', 24: '2', 25: '1&6', 26: '6', 27: '1', 28: '5', 29: '2', 30: '6', 31: '1', 32: '6', 33: '4&5', 34: '1', 35: '4&5', 36: '4', 37: '4', 38: '1&6', 39: '1', 40: '4&5', 41: '5', 42: '4&5', 43: '6', 44: '2&3', 45: '2&3', 46: '5', 47: '4&5', 48: '6', 49: '4&5', 50: '1'}
tour:tensor([[ 3, 11, 31, 19, 50,  0, 39, 38, 25,  0, 27, 14,  0, 34,  4,  0]], device='cuda:0')
tour:tensor([[29,  7, 45,  0, 44, 15, 24,  5,  0,  2, 23,  0]], device='cuda:0')
tour:tensor([[44,  5,  0,  9, 17,  0,  2, 45,  0]], device='cuda:0')
tour:tensor([[35, 42, 36, 12,  0, 18, 40,  6, 47, 37,  8,  0, 13, 22, 33,  1,  0, 10, 49,  0,  0]], device='cuda:0')
tour:tensor([[28, 42, 21, 20,  8,  0, 35, 41, 46,  0, 22, 33,  0, 49, 40,  6, 47,  0, 10, 12,  0]], device='cuda:0')
tour:tensor([[43, 11, 48, 16, 26,  4, 19,  0, 32, 38, 25,  0, 30,  0]], device='cuda:0')
 60%|██████████████████████████▍                 | 6/10 [00:01<00:01,  3.70it/s]
Node-Service Mapping:
{1: '2&3', 2: '2', 3: '2', 4: '4', 5: '5', 6: '2&3', 7: '4&5', 8: '3', 9: '4&5', 10: '4', 11: '2', 12: '4&5', 13: '1&6', 14: '2', 15: '6', 16: '1&6', 17: '1&6', 18: '5', 19: '6', 20: '2&3', 21: '4', 22: '5', 23: '6', 24: '4&5', 25: '1&6', 26: '1', 27: '2', 28: '1', 29: '5', 30: '1&6', 31: '2', 32: '6', 33: '4', 34: '6', 35: '2', 36: '4', 37: '2', 38: '2&3', 39: '3', 40: '4', 41: '5', 42: '2&3', 43: '5', 44: '5', 45: '1', 46: '4', 47: '1', 48: '2', 49: '4&5', 50: '2'}
tour:tensor([[47, 26, 13, 28, 16, 45,  0, 25, 17, 30,  0,  0]], device='cuda:0')
tour:tensor([[ 1, 35, 38,  0, 20, 31, 50, 14, 37,  0, 11,  6,  3, 42,  0, 48,  2,  0, 27,  0,  0]], device='cuda:0')
tour:tensor([[ 1, 39,  8, 38,  0, 20,  6, 42,  0]], device='cuda:0')
tour:tensor([[12,  4, 40, 33, 36,  7,  0, 21, 46,  0,  9, 10, 49, 24,  0]], device='cuda:0')
tour:tensor([[12,  7,  0,  9, 18, 22, 43,  0, 44, 29,  0,  5, 49, 24, 41,  0]], device='cuda:0')
tour:tensor([[34, 25, 19, 30,  0, 13, 16,  0, 23, 17, 15, 32,  0]], device='cuda:0')
 70%|██████████████████████████████▊             | 7/10 [00:02<00:00,  4.00it/s]
Node-Service Mapping:
{1: '4&5', 2: '4', 3: '2&3', 4: '5', 5: '3', 6: '1&6', 7: '1', 8: '2', 9: '3', 10: '3', 11: '3', 12: '5', 13: '1&6', 14: '1', 15: '2&3', 16: '4&5', 17: '5', 18: '2', 19: '4&5', 20: '2&3', 21: '6', 22: '4&5', 23: '5', 24: '3', 25: '1', 26: '1&6', 27: '3', 28: '2&3', 29: '4', 30: '2', 31: '5', 32: '4&5', 33: '6', 34: '4&5', 35: '6', 36: '2&3', 37: '5', 38: '5', 39: '6', 40: '4', 41: '4', 42: '2', 43: '1&6', 44: '2&3', 45: '1&6', 46: '6', 47: '3', 48: '5', 49: '1', 50: '2&3'}
tour:tensor([[14, 25,  6, 49,  0,  7,  0, 45, 13, 43, 26,  0,  0]], device='cuda:0')
tour:tensor([[18, 20, 28, 36,  0,  3, 42,  0, 44,  8, 50, 30, 15,  0,  0,  0]], device='cuda:0')
tour:tensor([[47, 20, 28, 36,  0,  3,  5, 11,  9,  0, 44, 27, 50, 10, 15,  0, 24,  0,  0]], device='cuda:0')
tour:tensor([[32, 41, 29, 22, 16,  2,  0, 34,  1, 19,  0, 40,  0,  0]], device='cuda:0')
tour:tensor([[23, 38, 17, 22, 16,  0, 34,  1, 48,  4,  0, 37, 12, 31,  0, 32, 19,  0]], device='cuda:0')
tour:tensor([[35, 43, 26,  0, 21, 46,  6,  0, 45, 39, 13, 33,  0,  0]], device='cuda:0')
 80%|███████████████████████████████████▏        | 8/10 [00:02<00:00,  4.11it/s]
Node-Service Mapping:
{1: '2', 2: '4&5', 3: '2&3', 4: '2', 5: '4', 6: '4&5', 7: '6', 8: '1', 9: '1', 10: '2', 11: '5', 12: '2&3', 13: '1', 14: '4&5', 15: '2', 16: '6', 17: '1&6', 18: '6', 19: '2', 20: '6', 21: '5', 22: '4', 23: '1', 24: '4', 25: '6', 26: '1&6', 27: '1', 28: '4', 29: '4', 30: '1', 31: '1', 32: '1', 33: '6', 34: '4', 35: '2', 36: '4', 37: '4', 38: '1', 39: '3', 40: '4&5', 41: '5', 42: '1', 43: '2&3', 44: '4', 45: '2', 46: '1', 47: '5', 48: '2', 49: '2', 50: '3'}
tour:tensor([[30, 23,  8, 32,  0, 46, 13, 26,  9,  0, 38, 27, 31, 17,  0, 42,  0]], device='cuda:0')
tour:tensor([[15,  3, 48,  0, 43,  0, 45,  1,  4, 19, 12,  0, 49, 10, 35,  0,  0]], device='cuda:0')
tour:tensor([[39, 50,  3,  0, 43,  0, 12]], device='cuda:0')
tour:tensor([[44,  5, 24, 34, 29, 40,  0, 37,  2, 22,  6, 14,  0, 36, 28,  0,  0]], device='cuda:0')
tour:tensor([[41,  2, 40, 47,  0, 21,  6, 14,  0, 11,  0]], device='cuda:0')
tour:tensor([[18,  7, 17,  0, 25, 33, 26,  0, 16, 20]], device='cuda:0')
 90%|███████████████████████████████████████▌    | 9/10 [00:02<00:00,  4.38it/s]
Node-Service Mapping:
{1: '2&3', 2: '6', 3: '6', 4: '4', 5: '4', 6: '5', 7: '4&5', 8: '1&6', 9: '3', 10: '1', 11: '1', 12: '1', 13: '2', 14: '2&3', 15: '6', 16: '5', 17: '1&6', 18: '1&6', 19: '4', 20: '1', 21: '6', 22: '2&3', 23: '1&6', 24: '5', 25: '5', 26: '2', 27: '4&5', 28: '2', 29: '2', 30: '2', 31: '2', 32: '6', 33: '4&5', 34: '3', 35: '3', 36: '6', 37: '3', 38: '6', 39: '2&3', 40: '1&6', 41: '2&3', 42: '4', 43: '1', 44: '3', 45: '5', 46: '3', 47: '1', 48: '3', 49: '2', 50: '4'}
tour:tensor([[18, 47, 17,  0, 12, 20, 23, 11,  0,  8, 43, 40,  0, 10,  0]], device='cuda:0')
tour:tensor([[49, 14, 26, 30, 29, 22,  0, 31, 41, 28,  0, 39,  1,  0, 13,  0]], device='cuda:0')
tour:tensor([[ 9, 14, 39, 22,  0, 34, 48, 35, 46, 44,  0, 37, 41,  1,  0]], device='cuda:0')
tour:tensor([[42, 33,  5,  4, 50,  0, 19,  7, 27,  0]], device='cuda:0')
tour:tensor([[ 6, 24,  0, 33, 16, 45,  7, 27,  0, 25,  0]], device='cuda:0')
tour:tensor([[18, 21, 17,  2,  3,  0, 15, 38, 36, 23, 32,  0,  8, 40,  0,  0]], device='cuda:0')
100%|███████████████████████████████████████████| 10/10 [00:02<00:00,  3.56it/s]
[2644.924285479495, 2431.840834213349, 2544.2897843379105, 2170.8921973966576, 2817.9109883646247, 2656.9763866219128, 2506.7165769206167, 2478.0525181051203, 2481.6433200211113, 2201.3963803887423]
Using sample strategy: Average cost: 2493.464327184954 +- 62.470613281038325

>> End of validation within 4.43s

进程已结束，退出代码为 0





```
# [0, 21, 12, 36, 1, 0, 45, 2, 16, 0, 50, 5,0],
# [0, 42, 0, 25, 44, 7, 37, 32, 0, 6, 0, 2, 16,0],

# [0,21, 50, 1, 0, 12, 36, 2, 16, 0, 45, 0, 5, 0],
# [0,25, 32, 42, 0, 7, 0, 6, 44, 2, 16, 0, 37,0],
```





sa

批次 1 汇总:
  Fleet 1: [0, 16, 7, 9, 37, 8, 19, 0, 14, 31, 3, 0, 4, 28, 5, 15]
  Fleet 2: [0, 11, 24, 0, 12, 0, 22, 0, 46, 26, 10, 0, 43, 2, 17, 44]
  Fleet 3: [0, 26, 44, 0, 46, 12, 10, 6, 45, 0, 2, 35]
  Fleet 4: [0, 42, 1, 0, 34, 0, 20, 33, 0, 47, 21, 27, 0, 23, 30, 0, 40, 48]
  Fleet 5: [0, 42, 25, 34, 0, 41, 0, 20, 32, 0, 49, 29, 36]
  Fleet 6: [0, 7, 5, 0, 38, 18, 4, 0, 15, 0, 14, 39, 13, 50]

批次 2 汇总:
  Fleet 1: [0, 40, 26, 0, 21, 0, 46, 4, 16, 48, 3]
  Fleet 2: [0, 33, 0, 22, 0, 18, 50, 20, 29, 0, 6, 19, 1, 0, 28, 36, 0, 45, 0, 9, 0, 7, 24, 0, 23, 30, 0, 35, 10]
  Fleet 3: [0, 35, 0, 45, 0, 6, 1, 0, 2, 29, 0, 43, 7, 24, 27, 0, 22, 18, 15]
  Fleet 4: [0, 12, 0, 38, 5, 17, 39]
  Fleet 5: [0, 8, 0, 37, 47, 39, 0, 5, 31, 17, 11, 0, 25, 34, 0, 13]
  Fleet 6: [0, 14, 0, 41, 0, 49, 4, 44, 0, 40, 0, 32, 42, 0, 16, 48]

批次 3 汇总:
  Fleet 1: [0, 21, 12, 36, 1, 0, 45, 2, 16, 0, 50, 5]
  Fleet 2: [0, 26, 0, 43, 0, 27, 39, 0, 31, 20, 24, 0, 11, 0, 34, 3, 10, 33, 0, 29, 14]
  Fleet 3: [0, 34, 10, 33, 0, 31, 0, 29, 3, 14, 23, 0, 40, 35, 49, 48, 13, 0, 26]
  Fleet 4: [0, 28, 38, 0, 41, 17, 0, 4, 9]
  Fleet 5: [0, 18, 19, 38, 47, 0, 4, 22, 0, 15, 0, 8, 30, 41, 28, 46]
  Fleet 6: [0, 42, 0, 25, 44, 7, 37, 32, 0, 6, 0, 2, 16]

批次 4 汇总:
  Fleet 1: [0, 4, 17, 16, 31, 19, 0, 41, 38, 2, 46]
  Fleet 2: [0, 20, 0, 23, 1, 15, 0, 13, 0, 12, 34, 6, 0, 40, 0, 18, 48, 29]
  Fleet 3: [0, 12, 25, 34, 0, 20, 0, 23, 0, 22, 43, 39, 0, 40, 0, 18, 1]
  Fleet 4: [0, 21, 0, 50, 45, 42, 0, 49, 24, 0, 36, 27, 0, 14, 0, 3, 37]
  Fleet 5: [0, 30, 0, 45, 32, 44, 0, 9, 35, 0, 47, 8, 11]
  Fleet 6: [0, 28, 38, 31, 0, 10, 33, 0, 26, 5, 7]

批次 5 汇总:
  Fleet 1: [0, 40, 0, 28, 11, 18, 38, 0, 22, 0, 8, 0, 29, 0, 36, 12, 43, 19, 17]
  Fleet 2: [0, 47, 0, 34, 0, 45, 0, 39, 46, 50, 10, 0, 27, 3]
  Fleet 3: [0, 25, 0, 9, 47, 0, 50, 0, 16, 24]
  Fleet 4: [0, 33, 0, 41, 14, 0, 6, 0, 1, 4, 0, 49, 0, 32, 15, 2, 7, 0, 21, 0, 48, 13, 23]
  Fleet 5: [0, 6, 0, 5, 0, 32, 30, 0, 1, 21, 4, 0, 35, 0, 48, 37, 0, 2, 20, 0, 44, 13, 23]
  Fleet 6: [0, 29, 0, 31, 17, 0, 26, 42, 40, 0, 8, 0, 36, 22]

批次 6 汇总:
  Fleet 1: [0, 31, 0, 38, 0, 50, 19, 4, 0, 39, 25, 0, 11, 34, 0, 3, 14, 27]
  Fleet 2: [0, 44, 45, 0, 29, 7, 0, 15, 0, 2, 0, 24, 23, 5]
  Fleet 3: [0, 2, 5, 0, 17, 0, 9, 0, 44, 45]
  Fleet 4: [0, 8, 0, 49, 1, 0, 42, 0, 10, 40, 37, 0, 18, 6, 0, 35, 47, 12, 0, 36, 0, 33, 0, 13, 22]
  Fleet 5: [0, 35, 41, 42, 49, 33, 40, 46, 0, 22, 6, 0, 10, 20, 8, 0, 21, 47, 12, 0, 28]
  Fleet 6: [0, 11, 16, 32, 48, 4, 19, 0, 26, 38, 25, 0, 43, 30]

批次 7 汇总:
  Fleet 1: [0, 13, 45, 30, 0, 17, 0, 47, 28, 16, 0, 26, 25]
  Fleet 2: [0, 11, 0, 3, 0, 20, 35, 38, 0, 14, 27, 0, 1, 48, 0, 6, 42, 0, 31, 0, 50, 2, 37]
  Fleet 3: [0, 8, 0, 20, 38, 0, 39, 0, 1, 0, 6, 42]
  Fleet 4: [0, 9, 4, 10, 33, 24, 0, 36, 49, 0, 7, 0, 40, 46, 0, 12, 21]
  Fleet 5: [0, 29, 7, 0, 22, 0, 9, 12, 41, 0, 49, 24, 0, 18, 0, 44, 5, 43]
  Fleet 6: [0, 23, 0, 16, 25, 15, 0, 17, 30, 0, 13, 0, 34, 32, 19]

批次 8 汇总:
  Fleet 1: [0, 45, 0, 13, 0, 25, 0, 7, 0, 43, 0, 14, 6, 49, 26]
  Fleet 2: [0, 15, 28, 36, 0, 8, 0, 20, 0, 44, 3, 50, 0, 18, 42, 30]
  Fleet 3: [0, 50, 24, 0, 27, 0, 10, 0, 44, 0, 47, 9, 0, 11, 0, 3, 5, 15, 0, 20, 28, 36]
  Fleet 4: [0, 41, 16, 2, 0, 40, 22, 0, 34, 1, 0, 32, 19, 29]
  Fleet 5: [0, 34, 0, 23, 38, 19, 17, 0, 1, 16, 4, 0, 22, 31, 0, 32, 37, 12, 48]
  Fleet 6: [0, 39, 43, 33, 0, 21, 26, 0, 46, 6, 0, 13, 0, 45, 35]

批次 9 汇总:
  Fleet 1: [0, 42, 0, 38, 23, 8, 0, 13, 0, 30, 0, 26, 32, 0, 17, 9, 0, 46, 27, 31]
  Fleet 2: [0, 19, 48, 0, 49, 10, 0, 15, 45, 43, 1, 4, 12, 0, 3, 35]
  Fleet 3: [0, 50, 0, 39, 0, 3, 0, 43, 12]
  Fleet 4: [0, 37, 2, 0, 40, 0, 24, 6, 28, 0, 36, 29, 14, 0, 34, 0, 44, 5, 22]
  Fleet 5: [0, 21, 0, 40, 0, 41, 47, 0, 2, 0, 6, 11, 14]
  Fleet 6: [0, 25, 17, 0, 20, 16, 0, 18, 33, 26, 0, 7]

批次 10 汇总:
  Fleet 1: [0, 8, 0, 12, 47, 0, 43, 0, 20, 40, 0, 18, 10, 17, 0, 23, 11]
  Fleet 2: [0, 26, 22, 0, 30, 1, 0, 49, 28, 0, 14, 13, 29, 0, 39, 31, 41]
  Fleet 3: [0, 44, 0, 46, 0, 9, 1, 22, 0, 39, 48, 0, 34, 14, 0, 37, 41, 35]
  Fleet 4: [0, 5, 0, 19, 0, 4, 27, 0, 42, 0, 7, 0, 33, 50]
  Fleet 5: [0, 7, 0, 6, 16, 27, 0, 33, 25, 0, 45, 24]
  Fleet 6: [0, 17, 32, 2, 0, 18, 38, 15, 36, 8, 0, 21, 23, 40, 3]

>> End of printing within 2134.24s

进程已结束，退出代码为 0







cplex

{1: '4.0', 2: '2&3', 3: '1.0', 4: '1&6', 5: '1&6', 6: '3.0', 7: '1&6', 8: '1.0', 9: '1.0', 10: '2&3', 11: '2.0', 12: '2&3', 13: '6.0', 14: '1&6', 15: '1&6', 16: '1.0', 17: '2.0', 18: '6.0', 19: '1.0', 20: '4&5', 21: '4.0', 22: '2.0', 23: '4.0', 24: '2.0', 25: '5.0', 26: '2&3', 27: '4.0', 28: '1.0', 29: '5.0', 30: '4.0', 31: '1.0', 32: '5.0', 33: '4.0', 34: '4&5', 35: '3.0', 36: '5.0', 37: '1.0', 38: '6.0', 39: '6.0', 40: '4.0', 41: '5.0', 42: '4&5', 43: '2.0', 44: '2&3', 45: '3.0', 46: '2&3', 47: '4.0', 48: '4.0', 49: '5.0', 50: '6.0'}

 10%|█     | 1/10 [01:32<13:56, 92.98s/it]

--- 实例 1 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2748.675499621526

Gap: 0.0009570602911252007

二进制变量数量: 478020

整数变量数量: 312

求解时间: 9.11 秒

分支定界节点数: 231

最佳界限: 2746.0448514476493



路径信息:

车队 1:

 车辆 3: Depot -> 9(need=1.0, L=628, T=628) -> 7(need=9.0, L=679, T=718) -> 28(need=1.0, L=1109, T=1109) -> 3(need=1.0, L=1135, T=1225) -> Depot

 tour:tensor([[ 9, 7, 28, 3, 0]])

 车辆 6: Depot -> 16(need=1.0, L=641, T=641) -> 31(need=1.0, L=877, T=967) -> 19(need=1.0, L=1143, T=1179) -> 8(need=1.0, L=1179, T=1269) -> Depot

 tour:tensor([[16, 31, 19, 8, 0]])

 车辆 20: Depot -> 14(need=9.0, L=742, T=832) -> 4(need=9.0, L=1104, T=1104) -> 15(need=9.0, L=1118, T=1194) -> Depot

 tour:tensor([[14, 4, 15, 0]])

 车辆 28: Depot -> 37(need=1.0, L=720, T=810) -> 5(need=9.0, L=1146, T=1236) -> Depot

 tour:tensor([[37, 5, 0]])

车队 2:

 车辆 3: Depot -> 46(need=7.0, L=668, T=668) -> 12(need=7.0, L=739, T=768) -> 17(need=2.0, L=1014, T=1044) -> Depot

 tour:tensor([[46, 12, 17, 0]])

 车辆 5: Depot -> 26(need=7.0, L=671, T=671) -> 43(need=2.0, L=891, T=891) -> 10(need=7.0, L=1036, T=1036) -> Depot

 tour:tensor([[26, 43, 10, 0]])

 车辆 6: Depot -> 22(need=2.0, L=668, T=668) -> 24(need=2.0, L=763, T=792) -> Depot

 tour:tensor([[22, 24, 0]])

 车辆 25: Depot -> 11(need=2.0, L=627, T=627) -> 2(need=7.0, L=926, T=926) -> 44(need=7.0, L=1028, T=1028) -> Depot

 tour:tensor([[11, 2, 44, 0]])

车队 3:

 车辆 1: Depot -> 2(need=7.0, L=926, T=936) -> 6(need=3.0, L=1144, T=1232) -> Depot

 tour:tensor([[2, 6, 0]])

 车辆 4: Depot -> 26(need=7.0, L=671, T=681) -> 12(need=7.0, L=739, T=798) -> 35(need=3.0, L=935, T=935) -> 44(need=7.0, L=1028, T=1038) -> 45(need=3.0, L=1111, T=1139) -> Depot

 tour:tensor([[26, 12, 35, 44, 45, 0]])

 车辆 20: Depot -> 46(need=7.0, L=668, T=678) -> 10(need=7.0, L=1036, T=1046) -> Depot

 tour:tensor([[46, 10, 0]])

车队 4:

 车辆 4: Depot -> 1(need=4.0, L=630, T=630) -> 33(need=4.0, L=1100, T=1100) -> 48(need=4.0, L=1162, T=1191) -> Depot

 tour:tensor([[ 1, 33, 48, 0]])

 车辆 12: Depot -> 40(need=4.0, L=659, T=659) -> 34(need=8.0, L=790, T=821) -> Depot

 tour:tensor([[40, 34, 0]])

 车辆 23: Depot -> 20(need=8.0, L=673, T=673) -> 47(need=4.0, L=751, T=782) -> 21(need=4.0, L=900, T=931) -> 27(need=4.0, L=1106, T=1106) -> Depot

 tour:tensor([[20, 47, 21, 27, 0]])

 车辆 27: Depot -> 42(need=8.0, L=625, T=625) -> 23(need=4.0, L=846, T=875) -> 30(need=4.0, L=1071, T=1071) -> Depot

 tour:tensor([[42, 23, 30, 0]])

车队 5:

 车辆 1: Depot -> 34(need=8.0, L=790, T=851) -> 32(need=5.0, L=1006, T=1096) -> Depot

 tour:tensor([[34, 32, 0]])

 车辆 4: Depot -> 20(need=8.0, L=673, T=683) -> 49(need=5.0, L=830, T=832) -> 29(need=5.0, L=923, T=1010) -> 36(need=5.0, L=1181, T=1271) -> Depot

 tour:tensor([[20, 49, 29, 36, 0]])

 车辆 11: Depot -> 42(need=8.0, L=625, T=635) -> 25(need=5.0, L=694, T=783) -> 41(need=5.0, L=973, T=1063) -> Depot

 tour:tensor([[42, 25, 41, 0]])

车队 6:

 车辆 3: Depot -> 7(need=9.0, L=679, T=718) -> 38(need=6.0, L=777, T=867) -> 18(need=6.0, L=868, T=958) -> 50(need=6.0, L=1152, T=1242) -> Depot

 tour:tensor([[ 7, 38, 18, 50, 0]])

 车辆 4: Depot -> 13(need=6.0, L=1116, T=1116) -> 5(need=9.0, L=1146, T=1236) -> Depot

 tour:tensor([[13, 5, 0]])

 车辆 22: Depot -> 39(need=6.0, L=683, T=687) -> 14(need=9.0, L=742, T=832) -> 4(need=9.0, L=1104, T=1104) -> 15(need=9.0, L=1118, T=1194) -> Depot

 tour:tensor([[39, 14, 4, 15, 0]])

--- 实例 1 求解结束，耗时 10.31 秒 ---



✓ 实例 1 求解成功: 2748.675499621526

Node-Service Mapping:

{1: '2&3', 2: '3.0', 3: '1.0', 4: '1&6', 5: '4&5', 6: '2&3', 7: '2&3', 8: '5.0', 9: '2.0', 10: '2.0', 11: '5.0', 12: '4.0', 13: '5.0', 14: '6.0', 15: '3.0', 16: '1&6', 17: '4&5', 18: '2&3', 19: '2.0', 20: '2.0', 21: '1.0', 22: '2&3', 23: '2.0', 24: '2&3', 25: '5.0', 26: '1.0', 27: '3.0', 28: '2.0', 29: '2&3', 30: '2.0', 31: '5.0', 32: '6.0', 33: '2.0', 34: '5.0', 35: '2&3', 36: '2.0', 37: '5.0', 38: '4.0', 39: '4&5', 40: '1&6', 41: '6.0', 42: '6.0', 43: '3.0', 44: '6.0', 45: '2&3', 46: '1.0', 47: '5.0', 48: '1&6', 49: '6.0', 50: '2.0'}

 20%|██    | 2/10 [03:04<12:15, 91.96s/it]

--- 实例 2 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2620.127290852651

Gap: 0.0

二进制变量数量: 478020

整数变量数量: 312

求解时间: 7.76 秒

分支定界节点数: 0

最佳界限: 2620.127290852651



路径信息:

车队 1:

 车辆 21: Depot -> 21(need=1.0, L=788, T=788) -> 4(need=9.0, L=877, T=878) -> 26(need=1.0, L=949, T=996) -> Depot

 tour:tensor([[21, 4, 26, 0]])

 车辆 30: Depot -> 46(need=1.0, L=726, T=726) -> 16(need=9.0, L=827, T=827) -> 40(need=9.0, L=907, T=917) -> 48(need=9.0, L=1074, T=1086) -> 3(need=1.0, L=1169, T=1259) -> Depot

 tour:tensor([[46, 16, 40, 48, 3, 0]])

车队 2:

 车辆 1: Depot -> 28(need=2.0, L=602, T=602) -> 23(need=2.0, L=910, T=940) -> 20(need=2.0, L=1011, T=1034) -> Depot

 tour:tensor([[28, 23, 20, 0]])

 车辆 6: Depot -> 36(need=2.0, L=813, T=817) -> 1(need=7.0, L=952, T=952) -> Depot

 tour:tensor([[36, 1, 0]])

 车辆 10: Depot -> 9(need=2.0, L=758, T=758) -> 24(need=7.0, L=907, T=907) -> Depot

 tour:tensor([[ 9, 24, 0]])

 车辆 12: Depot -> 30(need=2.0, L=922, T=948) -> Depot

 tour:tensor([[30, 0]])

 车辆 20: Depot -> 22(need=7.0, L=716, T=716) -> 7(need=7.0, L=828, T=828) -> 19(need=2.0, L=908, T=937) -> 33(need=2.0, L=1099, T=1129) -> Depot

 tour:tensor([[22, 7, 19, 33, 0]])

 车辆 21: Depot -> 6(need=7.0, L=763, T=763) -> 18(need=7.0, L=852, T=854) -> 29(need=7.0, L=1035, T=1038) -> Depot

 tour:tensor([[ 6, 18, 29, 0]])

 车辆 29: Depot -> 35(need=7.0, L=660, T=661) -> 45(need=7.0, L=773, T=773) -> 50(need=2.0, L=944, T=944) -> 10(need=2.0, L=1162, T=1191) -> Depot

 tour:tensor([[35, 45, 50, 10, 0]])

车队 3:

 车辆 1: Depot -> 43(need=3.0, L=620, T=620) -> 15(need=3.0, L=840, T=840) -> Depot

 tour:tensor([[43, 15, 0]])

 车辆 6: Depot -> 6(need=7.0, L=763, T=773) -> 18(need=7.0, L=852, T=865) -> 29(need=7.0, L=1035, T=1068) -> Depot

 tour:tensor([[ 6, 18, 29, 0]])

 车辆 21: Depot -> 35(need=7.0, L=660, T=691) -> 45(need=7.0, L=773, T=783) -> 1(need=7.0, L=952, T=976) -> 27(need=3.0, L=1002, T=1088) -> Depot

 tour:tensor([[35, 45, 1, 27, 0]])

 车辆 29: Depot -> 2(need=3.0, L=623, T=623) -> 22(need=7.0, L=716, T=726) -> 7(need=7.0, L=828, T=838) -> 24(need=7.0, L=907, T=930) -> Depot

 tour:tensor([[ 2, 22, 7, 24, 0]])

车队 4:

 车辆 8: Depot -> 17(need=8.0, L=885, T=914) -> Depot

 tour:tensor([[17, 0]])

 车辆 10: Depot -> 5(need=8.0, L=685, T=685) -> 39(need=8.0, L=901, T=931) -> Depot

 tour:tensor([[ 5, 39, 0]])

 车辆 25: Depot -> 38(need=4.0, L=672, T=672) -> 12(need=4.0, L=1132, T=1132) -> Depot

 tour:tensor([[38, 12, 0]])

车队 5:

 车辆 1: Depot -> 5(need=8.0, L=685, T=695) -> 11(need=5.0, L=859, T=869) -> 39(need=8.0, L=901, T=961) -> Depot

 tour:tensor([[ 5, 11, 39, 0]])

 车辆 7: Depot -> 31(need=5.0, L=670, T=695) -> 13(need=5.0, L=797, T=797) -> 17(need=8.0, L=885, T=944) -> Depot

 tour:tensor([[31, 13, 17, 0]])

 车辆 10: Depot -> 47(need=5.0, L=843, T=843) -> Depot

 tour:tensor([[47, 0]])

 车辆 30: Depot -> 37(need=5.0, L=605, T=605) -> 25(need=5.0, L=677, T=705) -> 8(need=5.0, L=1004, T=1051) -> 34(need=5.0, L=1086, T=1174) -> Depot

 tour:tensor([[37, 25, 8, 34, 0]])

车队 6:

 车辆 7: Depot -> 16(need=9.0, L=827, T=827) -> 40(need=9.0, L=907, T=917) -> 48(need=9.0, L=1074, T=1086) -> Depot

 tour:tensor([[16, 40, 48, 0]])

 车辆 10: Depot -> 14(need=6.0, L=709, T=744) -> 32(need=6.0, L=744, T=834) -> 42(need=6.0, L=939, T=996) -> Depot

 tour:tensor([[14, 32, 42, 0]])

 车辆 12: Depot -> 49(need=6.0, L=668, T=668) -> 41(need=6.0, L=739, T=758) -> 4(need=9.0, L=877, T=878) -> 44(need=6.0, L=1086, T=1176) -> Depot

 tour:tensor([[49, 41, 4, 44, 0]])

--- 实例 2 求解结束，耗时 8.99 秒 ---



✓ 实例 2 求解成功: 2620.127290852651

Node-Service Mapping:

{1: '1.0', 2: '1&6', 3: '2&3', 4: '4&5', 5: '1.0', 6: '6.0', 7: '6.0', 8: '5.0', 9: '4.0', 10: '2&3', 11: '2.0', 12: '1.0', 13: '3.0', 14: '2&3', 15: '5.0', 16: '1&6', 17: '4.0', 18: '5.0', 19: '5.0', 20: '2.0', 21: '1.0', 22: '5.0', 23: '3.0', 24: '2.0', 25: '6.0', 26: '2&3', 27: '2.0', 28: '4&5', 29: '2&3', 30: '5.0', 31: '2&3', 32: '6.0', 33: '2&3', 34: '2&3', 35: '3.0', 36: '1.0', 37: '6.0', 38: '4&5', 39: '2.0', 40: '3.0', 41: '4&5', 42: '6.0', 43: '2.0', 44: '6.0', 45: '1.0', 46: '5.0', 47: '5.0', 48: '3.0', 49: '3.0', 50: '1.0'}

 30%|███    | 3/10 [04:52<11:36, 99.56s/it]

--- 实例 3 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2680.254980160206

Gap: 0.0

二进制变量数量: 478020

整数变量数量: 312

求解时间: 21.15 秒

分支定界节点数: 8084

最佳界限: 2680.254980160206



路径信息:

车队 1:

 车辆 1: Depot -> 45(need=1.0, L=676, T=709) -> 12(need=1.0, L=714, T=799) -> 2(need=9.0, L=889, T=927) -> 36(need=1.0, L=945, T=1035) -> 16(need=9.0, L=1168, T=1258) -> Depot

 tour:tensor([[45, 12, 2, 36, 16, 0]])

 车辆 3: Depot -> 5(need=1.0, L=1043, T=1125) -> Depot

 tour:tensor([[5, 0]])

 车辆 29: Depot -> 21(need=1.0, L=613, T=613) -> 50(need=1.0, L=667, T=703) -> 1(need=1.0, L=1064, T=1154) -> Depot

 tour:tensor([[21, 50, 1, 0]])

车队 2:

 车辆 3: Depot -> 11(need=2.0, L=668, T=668) -> 20(need=2.0, L=889, T=889) -> 24(need=2.0, L=953, T=982) -> Depot

 tour:tensor([[11, 20, 24, 0]])

 车辆 6: Depot -> 27(need=2.0, L=608, T=608) -> 3(need=7.0, L=675, T=698) -> 26(need=7.0, L=1091, T=1119) -> Depot

 tour:tensor([[27, 3, 26, 0]])

 车辆 21: Depot -> 29(need=7.0, L=619, T=619) -> Depot

 tour:tensor([[29, 0]])

 车辆 26: Depot -> 31(need=7.0, L=641, T=641) -> 14(need=7.0, L=727, T=757) -> 39(need=2.0, L=1075, T=1075) -> 33(need=7.0, L=1177, T=1206) -> Depot

 tour:tensor([[31, 14, 39, 33, 0]])

 车辆 28: Depot -> 34(need=7.0, L=612, T=612) -> 43(need=2.0, L=723, T=753) -> 10(need=7.0, L=1116, T=1145) -> Depot

 tour:tensor([[34, 43, 10, 0]])

车队 3:

 车辆 1: Depot -> 34(need=7.0, L=612, T=622) -> 23(need=3.0, L=1110, T=1133) -> 13(need=3.0, L=1138, T=1226) -> Depot

 tour:tensor([[34, 23, 13, 0]])

 车辆 3: Depot -> 48(need=3.0, L=1056, T=1056) -> 26(need=7.0, L=1091, T=1149) -> Depot

 tour:tensor([[48, 26, 0]])

 车辆 5: Depot -> 31(need=7.0, L=641, T=651) -> 40(need=3.0, L=977, T=977) -> 33(need=7.0, L=1177, T=1236) -> Depot

 tour:tensor([[31, 40, 33, 0]])

 车辆 10: Depot -> 14(need=7.0, L=727, T=787) -> 49(need=3.0, L=1065, T=1065) -> 10(need=7.0, L=1116, T=1175) -> Depot

 tour:tensor([[14, 49, 10, 0]])

 车辆 22: Depot -> 29(need=7.0, L=619, T=629) -> 3(need=7.0, L=675, T=722) -> 35(need=3.0, L=1062, T=1062) -> Depot

 tour:tensor([[29, 3, 35, 0]])

车队 4:

 车辆 3: Depot -> 38(need=8.0, L=1105, T=1108) -> Depot

 tour:tensor([[38, 0]])

 车辆 5: Depot -> 9(need=4.0, L=860, T=860) -> 41(need=8.0, L=1023, T=1046) -> 17(need=4.0, L=1169, T=1198) -> Depot

 tour:tensor([[ 9, 41, 17, 0]])

 车辆 14: Depot -> 4(need=8.0, L=701, T=701) -> 28(need=8.0, L=1091, T=1121) -> Depot

 tour:tensor([[ 4, 28, 0]])

车队 5:

 车辆 7: Depot -> 4(need=8.0, L=701, T=711) -> 19(need=5.0, L=1048, T=1048) -> 28(need=8.0, L=1091, T=1151) -> Depot

 tour:tensor([[ 4, 19, 28, 0]])

 车辆 10: Depot -> 22(need=5.0, L=699, T=699) -> 30(need=5.0, L=984, T=984) -> 41(need=8.0, L=1023, T=1076) -> Depot

 tour:tensor([[22, 30, 41, 0]])

 车辆 15: Depot -> 8(need=5.0, L=762, T=852) -> 38(need=8.0, L=1105, T=1138) -> 47(need=5.0, L=1191, T=1280) -> Depot

 tour:tensor([[ 8, 38, 47, 0]])

 车辆 21: Depot -> 18(need=5.0, L=751, T=751) -> 15(need=5.0, L=801, T=891) -> 46(need=5.0, L=1110, T=1200) -> Depot

 tour:tensor([[18, 15, 46, 0]])

车队 6:

 车辆 1: Depot -> 7(need=6.0, L=890, T=890) -> 32(need=6.0, L=954, T=1044) -> Depot

 tour:tensor([[ 7, 32, 0]])

 车辆 3: Depot -> 6(need=6.0, L=649, T=649) -> 44(need=6.0, L=837, T=837) -> 2(need=9.0, L=889, T=927) -> 16(need=9.0, L=1168, T=1258) -> Depot

 tour:tensor([[ 6, 44, 2, 16, 0]])

 车辆 11: Depot -> 25(need=6.0, L=692, T=739) -> 42(need=6.0, L=936, T=936) -> 37(need=6.0, L=955, T=1045) -> Depot

 tour:tensor([[25, 42, 37, 0]])

--- 实例 3 求解结束，耗时 22.38 秒 ---



✓ 实例 3 求解成功: 2680.254980160206

Node-Service Mapping:

{1: '2&3', 2: '1.0', 3: '4.0', 4: '1.0', 5: '6.0', 6: '2.0', 7: '6.0', 8: '5.0', 9: '5.0', 10: '6.0', 11: '5.0', 12: '2&3', 13: '2.0', 14: '4.0', 15: '2.0', 16: '1.0', 17: '1.0', 18: '2&3', 19: '1.0', 20: '2&3', 21: '4.0', 22: '3.0', 23: '2&3', 24: '4.0', 25: '3.0', 26: '6.0', 27: '4.0', 28: '6.0', 29: '2.0', 30: '5.0', 31: '1&6', 32: '5.0', 33: '6.0', 34: '2&3', 35: '5.0', 36: '4.0', 37: '4.0', 38: '1&6', 39: '3.0', 40: '2&3', 41: '1.0', 42: '4.0', 43: '3.0', 44: '5.0', 45: '4&5', 46: '1.0', 47: '5.0', 48: '2.0', 49: '4.0', 50: '4.0'}

 40%|████   | 4/10 [06:27<09:45, 97.58s/it]

--- 实例 4 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2344.2510480052933

Gap: 0.00018747818855912148

二进制变量数量: 478020

整数变量数量: 312

求解时间: 7.21 秒

分支定界节点数: 77

最佳界限: 2343.8115520652855



路径信息:

车队 1:

 车辆 3: Depot -> 4(need=1.0, L=679, T=679) -> 41(need=1.0, L=699, T=789) -> 2(need=1.0, L=965, T=965) -> 31(need=9.0, L=976, T=1066) -> Depot

 tour:tensor([[ 4, 41, 2, 31, 0]])

 车辆 4: Depot -> 17(need=1.0, L=766, T=854) -> 16(need=1.0, L=944, T=944) -> 46(need=1.0, L=1053, T=1092) -> Depot

 tour:tensor([[17, 16, 46, 0]])

 车辆 26: Depot -> 38(need=9.0, L=662, T=662) -> 19(need=1.0, L=1092, T=1182) -> Depot

 tour:tensor([[38, 19, 0]])

车队 2:

 车辆 1: Depot -> 48(need=2.0, L=786, T=815) -> 20(need=7.0, L=1024, T=1053) -> Depot

 tour:tensor([[48, 20, 0]])

 车辆 5: Depot -> 40(need=7.0, L=650, T=650) -> 34(need=7.0, L=716, T=741) -> Depot

 tour:tensor([[40, 34, 0]])

 车辆 16: Depot -> 6(need=2.0, L=771, T=771) -> 29(need=2.0, L=877, T=906) -> 13(need=2.0, L=1036, T=1065) -> Depot

 tour:tensor([[ 6, 29, 13, 0]])

 车辆 20: Depot -> 23(need=7.0, L=670, T=670) -> 1(need=7.0, L=953, T=953) -> Depot

 tour:tensor([[23, 1, 0]])

 车辆 30: Depot -> 12(need=7.0, L=664, T=664) -> 18(need=7.0, L=733, T=762) -> 15(need=2.0, L=1073, T=1102) -> Depot

 tour:tensor([[12, 18, 15, 0]])

车队 3:

 车辆 2: Depot -> 22(need=3.0, L=663, T=663) -> 43(need=3.0, L=679, T=767) -> Depot

 tour:tensor([[22, 43, 0]])

 车辆 4: Depot -> 40(need=7.0, L=650, T=660) -> 18(need=7.0, L=733, T=792) -> Depot

 tour:tensor([[40, 18, 0]])

 车辆 18: Depot -> 25(need=3.0, L=705, T=793) -> 20(need=7.0, L=1024, T=1063) -> 39(need=3.0, L=1070, T=1156) -> Depot

 tour:tensor([[25, 20, 39, 0]])

 车辆 29: Depot -> 12(need=7.0, L=664, T=674) -> 34(need=7.0, L=716, T=771) -> Depot

 tour:tensor([[12, 34, 0]])

 车辆 30: Depot -> 23(need=7.0, L=670, T=680) -> 1(need=7.0, L=953, T=963) -> Depot

 tour:tensor([[23, 1, 0]])

车队 4:

 车辆 5: Depot -> 49(need=4.0, L=683, T=683) -> 36(need=4.0, L=1041, T=1056) -> Depot

 tour:tensor([[49, 36, 0]])

 车辆 16: Depot -> 37(need=4.0, L=711, T=742) -> 50(need=4.0, L=907, T=907) -> 27(need=4.0, L=1122, T=1122) -> 14(need=4.0, L=1182, T=1213) -> Depot

 tour:tensor([[37, 50, 27, 14, 0]])

 车辆 20: Depot -> 24(need=4.0, L=956, T=956) -> Depot

 tour:tensor([[24, 0]])

 车辆 23: Depot -> 45(need=8.0, L=998, T=1016) -> 42(need=4.0, L=1078, T=1107) -> Depot

 tour:tensor([[45, 42, 0]])

 车辆 25: Depot -> 3(need=4.0, L=631, T=631) -> 21(need=4.0, L=966, T=966) -> Depot

 tour:tensor([[ 3, 21, 0]])

车队 5:

 车辆 5: Depot -> 45(need=8.0, L=998, T=1046) -> Depot

 tour:tensor([[45, 0]])

 车辆 15: Depot -> 47(need=5.0, L=617, T=617) -> 9(need=5.0, L=641, T=709) -> 30(need=5.0, L=956, T=956) -> 44(need=5.0, L=1182, T=1270) -> Depot

 tour:tensor([[47, 9, 30, 44, 0]])

 车辆 24: Depot -> 8(need=5.0, L=658, T=671) -> 11(need=5.0, L=717, T=763) -> 35(need=5.0, L=763, T=853) -> 32(need=5.0, L=1088, T=1176) -> Depot

 tour:tensor([[ 8, 11, 35, 32, 0]])

车队 6:

 车辆 2: Depot -> 10(need=6.0, L=602, T=602) -> 28(need=6.0, L=615, T=692) -> 26(need=6.0, L=758, T=848) -> 33(need=6.0, L=1056, T=1146) -> Depot

 tour:tensor([[10, 28, 26, 33, 0]])

 车辆 3: Depot -> 38(need=9.0, L=662, T=662) -> 5(need=6.0, L=965, T=965) -> Depot

 tour:tensor([[38, 5, 0]])

 车辆 14: Depot -> 7(need=6.0, L=974, T=973) -> 31(need=9.0, L=976, T=1066) -> Depot

 tour:tensor([[ 7, 31, 0]])

--- 实例 4 求解结束，耗时 8.44 秒 ---



✓ 实例 4 求解成功: 2344.2510480052933

Node-Service Mapping:

{1: '4&5', 2: '4&5', 3: '2.0', 4: '4&5', 5: '5.0', 6: '4&5', 7: '4.0', 8: '1&6', 9: '3.0', 10: '2.0', 11: '1.0', 12: '1.0', 13: '4&5', 14: '4.0', 15: '4.0', 16: '3.0', 17: '1&6', 18: '1.0', 19: '1.0', 20: '5.0', 21: '4&5', 22: '1&6', 23: '4&5', 24: '3.0', 25: '3.0', 26: '6.0', 27: '2.0', 28: '1.0', 29: '1&6', 30: '5.0', 31: '6.0', 32: '4&5', 33: '4.0', 34: '2.0', 35: '5.0', 36: '1&6', 37: '5.0', 38: '1.0', 39: '2.0', 40: '1&6', 41: '4.0', 42: '6.0', 43: '1.0', 44: '5.0', 45: '2.0', 46: '2.0', 47: '2&3', 48: '4&5', 49: '4.0', 50: '2&3'}

 50%|█████   | 5/10 [08:05<08:09, 97.90s/it]

--- 实例 5 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 3046.7118074343666

Gap: 0.0009923996564695134

二进制变量数量: 478020

整数变量数量: 312

求解时间: 11.69 秒

分支定界节点数: 3195

最佳界限: 3043.688251683307



路径信息:

车队 1:

 车辆 1: Depot -> 36(need=9.0, L=735, T=735) -> 18(need=1.0, L=1054, T=1136) -> 38(need=1.0, L=1136, T=1226) -> Depot

 tour:tensor([[36, 18, 38, 0]])

 车辆 2: Depot -> 28(need=1.0, L=815, T=900) -> 17(need=9.0, L=1137, T=1137) -> 40(need=9.0, L=1158, T=1227) -> Depot

 tour:tensor([[28, 17, 40, 0]])

 车辆 10: Depot -> 12(need=1.0, L=720, T=720) -> 11(need=1.0, L=900, T=900) -> 29(need=9.0, L=902, T=992) -> 43(need=1.0, L=1019, T=1082) -> 19(need=1.0, L=1159, T=1249) -> Depot

 tour:tensor([[12, 11, 29, 43, 19, 0]])

 车辆 28: Depot -> 8(need=9.0, L=1043, T=1133) -> 22(need=9.0, L=1143, T=1233) -> Depot

 tour:tensor([[ 8, 22, 0]])

车队 2:

 车辆 1: Depot -> 46(need=2.0, L=763, T=763) -> 47(need=7.0, L=940, T=969) -> Depot

 tour:tensor([[46, 47, 0]])

 车辆 9: Depot -> 27(need=2.0, L=652, T=652) -> 50(need=7.0, L=797, T=815) -> 3(need=2.0, L=1146, T=1146) -> Depot

 tour:tensor([[27, 50, 3, 0]])

 车辆 16: Depot -> 10(need=2.0, L=994, T=994) -> Depot

 tour:tensor([[10, 0]])

 车辆 20: Depot -> 39(need=2.0, L=738, T=743) -> 45(need=2.0, L=826, T=855) -> 34(need=2.0, L=1122, T=1122) -> Depot

 tour:tensor([[39, 45, 34, 0]])

车队 3:

 车辆 12: Depot -> 16(need=3.0, L=639, T=639) -> 24(need=3.0, L=699, T=731) -> 50(need=7.0, L=797, T=825) -> 25(need=3.0, L=1094, T=1094) -> Depot

 tour:tensor([[16, 24, 50, 25, 0]])

 车辆 30: Depot -> 9(need=3.0, L=848, T=905) -> 47(need=7.0, L=940, T=999) -> Depot

 tour:tensor([[ 9, 47, 0]])

车队 4:

 车辆 2: Depot -> 13(need=8.0, L=873, T=901) -> 6(need=8.0, L=961, T=991) -> 4(need=8.0, L=1129, T=1159) -> Depot

 tour:tensor([[13, 6, 4, 0]])

 车辆 5: Depot -> 1(need=8.0, L=818, T=828) -> 15(need=4.0, L=904, T=934) -> 7(need=4.0, L=1082, T=1082) -> Depot

 tour:tensor([[ 1, 15, 7, 0]])

 车辆 18: Depot -> 41(need=4.0, L=764, T=783) -> 14(need=4.0, L=923, T=938) -> 21(need=8.0, L=1117, T=1126) -> Depot

 tour:tensor([[41, 14, 21, 0]])

 车辆 19: Depot -> 2(need=8.0, L=1036, T=1067) -> Depot

 tour:tensor([[2, 0]])

 车辆 24: Depot -> 32(need=8.0, L=695, T=726) -> 23(need=8.0, L=1167, T=1197) -> Depot

 tour:tensor([[32, 23, 0]])

 车辆 27: Depot -> 48(need=8.0, L=629, T=629) -> 49(need=4.0, L=1022, T=1036) -> 33(need=4.0, L=1133, T=1164) -> Depot

 tour:tensor([[48, 49, 33, 0]])

车队 5:

 车辆 6: Depot -> 32(need=8.0, L=695, T=736) -> 23(need=8.0, L=1167, T=1227) -> Depot

 tour:tensor([[32, 23, 0]])

 车辆 16: Depot -> 44(need=5.0, L=823, T=823) -> Depot

 tour:tensor([[44, 0]])

 车辆 20: Depot -> 5(need=5.0, L=661, T=661) -> 2(need=8.0, L=1036, T=1097) -> 20(need=5.0, L=1166, T=1254) -> Depot

 tour:tensor([[ 5, 2, 20, 0]])

 车辆 21: Depot -> 48(need=8.0, L=629, T=639) -> 37(need=5.0, L=671, T=731) -> 4(need=8.0, L=1129, T=1189) -> Depot

 tour:tensor([[48, 37, 4, 0]])

 车辆 24: Depot -> 1(need=8.0, L=818, T=841) -> 13(need=8.0, L=873, T=931) -> 6(need=8.0, L=961, T=1021) -> 21(need=8.0, L=1117, T=1136) -> Depot

 tour:tensor([[ 1, 13, 6, 21, 0]])

 车辆 30: Depot -> 35(need=5.0, L=822, T=822) -> 30(need=5.0, L=894, T=982) -> Depot

 tour:tensor([[35, 30, 0]])

车队 6:

 车辆 2: Depot -> 31(need=6.0, L=1018, T=1018) -> 8(need=9.0, L=1043, T=1133) -> Depot

 tour:tensor([[31, 8, 0]])

 车辆 3: Depot -> 36(need=9.0, L=735, T=735) -> 42(need=6.0, L=895, T=895) -> 17(need=9.0, L=1137, T=1137) -> 40(need=9.0, L=1158, T=1227) -> Depot

 tour:tensor([[36, 42, 17, 40, 0]])

 车辆 15: Depot -> 29(need=9.0, L=902, T=992) -> Depot

 tour:tensor([[29, 0]])

 车辆 23: Depot -> 26(need=6.0, L=909, T=999) -> 22(need=9.0, L=1143, T=1233) -> Depot

 tour:tensor([[26, 22, 0]])

--- 实例 5 求解结束，耗时 12.95 秒 ---



✓ 实例 5 求解成功: 3046.7118074343666

Node-Service Mapping:

{1: '4.0', 2: '2&3', 3: '1.0', 4: '1&6', 5: '2&3', 6: '4&5', 7: '2.0', 8: '4&5', 9: '3.0', 10: '4&5', 11: '1&6', 12: '4&5', 13: '4.0', 14: '1.0', 15: '2.0', 16: '6.0', 17: '3.0', 18: '4.0', 19: '1&6', 20: '5.0', 21: '5.0', 22: '4&5', 23: '2.0', 24: '2.0', 25: '1&6', 26: '6.0', 27: '1.0', 28: '5.0', 29: '2.0', 30: '6.0', 31: '1.0', 32: '6.0', 33: '4&5', 34: '1.0', 35: '4&5', 36: '4.0', 37: '4.0', 38: '1&6', 39: '1.0', 40: '4&5', 41: '5.0', 42: '4&5', 43: '6.0', 44: '2&3', 45: '2&3', 46: '5.0', 47: '4&5', 48: '6.0', 49: '4&5', 50: '1.0'}

 60%|██████  | 6/10 [09:41<06:29, 97.30s/it]

--- 实例 6 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2777.431808944002

Gap: 0.0

二进制变量数量: 478020

整数变量数量: 312

求解时间: 6.79 秒

分支定界节点数: 0

最佳界限: 2777.431808944002



路径信息:

车队 1:

 车辆 9: Depot -> 39(need=1.0, L=650, T=650) -> 38(need=9.0, L=930, T=954) -> 25(need=9.0, L=1109, T=1199) -> Depot

 tour:tensor([[39, 38, 25, 0]])

 车辆 16: Depot -> 3(need=1.0, L=631, T=631) -> 11(need=9.0, L=695, T=731) -> 31(need=1.0, L=882, T=882) -> 19(need=9.0, L=1095, T=1109) -> Depot

 tour:tensor([[ 3, 11, 31, 19, 0]])

 车辆 23: Depot -> 27(need=1.0, L=822, T=822) -> 14(need=1.0, L=851, T=912) -> Depot

 tour:tensor([[27, 14, 0]])

 车辆 28: Depot -> 34(need=1.0, L=892, T=954) -> 4(need=9.0, L=1044, T=1044) -> 50(need=1.0, L=1046, T=1134) -> Depot

 tour:tensor([[34, 4, 50, 0]])

车队 2:

 车辆 6: Depot -> 15(need=2.0, L=681, T=710) -> 23(need=2.0, L=878, T=878) -> Depot

 tour:tensor([[15, 23, 0]])

 车辆 9: Depot -> 44(need=7.0, L=618, T=618) -> 24(need=2.0, L=688, T=718) -> 5(need=7.0, L=1185, T=1214) -> Depot

 tour:tensor([[44, 24, 5, 0]])

 车辆 15: Depot -> 2(need=7.0, L=669, T=694) -> 45(need=7.0, L=1074, T=1103) -> Depot

 tour:tensor([[ 2, 45, 0]])

 车辆 21: Depot -> 29(need=2.0, L=603, T=603) -> 7(need=2.0, L=913, T=913) -> Depot

 tour:tensor([[29, 7, 0]])

车队 3:

 车辆 9: Depot -> 44(need=7.0, L=618, T=628) -> 5(need=7.0, L=1185, T=1244) -> Depot

 tour:tensor([[44, 5, 0]])

 车辆 12: Depot -> 9(need=3.0, L=716, T=804) -> 17(need=3.0, L=1021, T=1021) -> Depot

 tour:tensor([[ 9, 17, 0]])

 车辆 26: Depot -> 2(need=7.0, L=669, T=712) -> 45(need=7.0, L=1074, T=1133) -> Depot

 tour:tensor([[ 2, 45, 0]])

车队 4:

 车辆 15: Depot -> 10(need=8.0, L=716, T=716) -> 6(need=8.0, L=1016, T=1016) -> Depot

 tour:tensor([[10, 6, 0]])

 车辆 17: Depot -> 35(need=8.0, L=618, T=618) -> 37(need=4.0, L=1087, T=1116) -> Depot

 tour:tensor([[35, 37, 0]])

 车辆 21: Depot -> 18(need=4.0, L=648, T=648) -> 40(need=8.0, L=885, T=913) -> 47(need=8.0, L=1018, T=1018) -> 8(need=8.0, L=1104, T=1109) -> Depot

 tour:tensor([[18, 40, 47, 8, 0]])

 车辆 23: Depot -> 13(need=4.0, L=712, T=716) -> 22(need=8.0, L=806, T=806) -> Depot

 tour:tensor([[13, 22, 0]])

 车辆 25: Depot -> 49(need=8.0, L=842, T=842) -> 1(need=4.0, L=977, T=1007) -> Depot

 tour:tensor([[49, 1, 0]])

 车辆 28: Depot -> 33(need=8.0, L=850, T=878) -> Depot

 tour:tensor([[33, 0]])

 车辆 29: Depot -> 42(need=8.0, L=811, T=811) -> 36(need=4.0, L=955, T=955) -> 12(need=8.0, L=1146, T=1177) -> Depot

 tour:tensor([[42, 36, 12, 0]])

车队 5:

 车辆 2: Depot -> 22(need=8.0, L=806, T=816) -> 33(need=8.0, L=850, T=908) -> Depot

 tour:tensor([[22, 33, 0]])

 车辆 3: Depot -> 28(need=5.0, L=613, T=613) -> 42(need=8.0, L=811, T=821) -> 21(need=5.0, L=980, T=980) -> Depot

 tour:tensor([[28, 42, 21, 0]])

 车辆 7: Depot -> 49(need=8.0, L=842, T=852) -> 40(need=8.0, L=885, T=943) -> 47(need=8.0, L=1018, T=1033) -> Depot

 tour:tensor([[49, 40, 47, 0]])

 车辆 10: Depot -> 10(need=8.0, L=716, T=726) -> 6(need=8.0, L=1016, T=1026) -> 12(need=8.0, L=1146, T=1207) -> Depot

 tour:tensor([[10, 6, 12, 0]])

 车辆 20: Depot -> 35(need=8.0, L=618, T=628) -> 41(need=5.0, L=697, T=718) -> 46(need=5.0, L=897, T=897) -> 20(need=5.0, L=931, T=989) -> 8(need=8.0, L=1104, T=1119) -> Depot

 tour:tensor([[35, 41, 46, 20, 8, 0]])

车队 6:

 车辆 1: Depot -> 43(need=6.0, L=641, T=641) -> 11(need=9.0, L=695, T=731) -> 19(need=9.0, L=1095, T=1109) -> Depot

 tour:tensor([[43, 11, 19, 0]])

 车辆 16: Depot -> 30(need=6.0, L=855, T=864) -> Depot

 tour:tensor([[30, 0]])

 车辆 20: Depot -> 32(need=6.0, L=745, T=758) -> 38(need=9.0, L=930, T=954) -> 25(need=9.0, L=1109, T=1199) -> Depot

 tour:tensor([[32, 38, 25, 0]])

 车辆 21: Depot -> 16(need=6.0, L=687, T=687) -> 48(need=6.0, L=701, T=777) -> 26(need=6.0, L=848, T=867) -> 4(need=9.0, L=1044, T=1044) -> Depot

 tour:tensor([[16, 48, 26, 4, 0]])

--- 实例 6 求解结束，耗时 7.95 秒 ---



✓ 实例 6 求解成功: 2777.431808944002

Node-Service Mapping:

{1: '2&3', 2: '2.0', 3: '2.0', 4: '4.0', 5: '5.0', 6: '2&3', 7: '4&5', 8: '3.0', 9: '4&5', 10: '4.0', 11: '2.0', 12: '4&5', 13: '1&6', 14: '2.0', 15: '6.0', 16: '1&6', 17: '1&6', 18: '5.0', 19: '6.0', 20: '2&3', 21: '4.0', 22: '5.0', 23: '6.0', 24: '4&5', 25: '1&6', 26: '1.0', 27: '2.0', 28: '1.0', 29: '5.0', 30: '1&6', 31: '2.0', 32: '6.0', 33: '4.0', 34: '6.0', 35: '2.0', 36: '4.0', 37: '2.0', 38: '2&3', 39: '3.0', 40: '4.0', 41: '5.0', 42: '2&3', 43: '5.0', 44: '5.0', 45: '1.0', 46: '4.0', 47: '1.0', 48: '2.0', 49: '4&5', 50: '2.0'}

 70%|███████  | 7/10 [11:15<04:47, 95.98s/it]

--- 实例 7 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2643.802637693061

Gap: 0.000962524619285026

二进制变量数量: 478020

整数变量数量: 312

求解时间: 7.91 秒

分支定界节点数: 215

最佳界限: 2641.2579125657508



路径信息:

车队 1:

 车辆 6: Depot -> 16(need=9.0, L=864, T=899) -> 25(need=9.0, L=995, T=995) -> 17(need=9.0, L=1053, T=1143) -> 30(need=9.0, L=1174, T=1264) -> Depot

 tour:tensor([[16, 25, 17, 30, 0]])

 车辆 14: Depot -> 47(need=1.0, L=603, T=603) -> 26(need=1.0, L=719, T=719) -> 13(need=9.0, L=751, T=809) -> 28(need=1.0, L=895, T=899) -> 45(need=1.0, L=924, T=989) -> Depot

 tour:tensor([[47, 26, 13, 28, 45, 0]])

车队 2:

 车辆 5: Depot -> 35(need=2.0, L=734, T=734) -> 50(need=2.0, L=892, T=922) -> Depot

 tour:tensor([[35, 50, 0]])

 车辆 10: Depot -> 31(need=2.0, L=714, T=714) -> 14(need=2.0, L=894, T=894) -> 37(need=2.0, L=1132, T=1161) -> Depot

 tour:tensor([[31, 14, 37, 0]])

 车辆 14: Depot -> 20(need=7.0, L=682, T=682) -> 48(need=2.0, L=831, T=831) -> 2(need=2.0, L=893, T=922) -> 3(need=2.0, L=1028, T=1057) -> Depot

 tour:tensor([[20, 48, 2, 3, 0]])

 车辆 25: Depot -> 27(need=2.0, L=921, T=950) -> Depot

 tour:tensor([[27, 0]])

 车辆 30: Depot -> 1(need=7.0, L=717, T=717) -> 11(need=2.0, L=869, T=869) -> 6(need=7.0, L=978, T=978) -> 42(need=7.0, L=1065, T=1071) -> 38(need=7.0, L=1149, T=1178) -> Depot

 tour:tensor([[ 1, 11, 6, 42, 38, 0]])

车队 3:

 车辆 3: Depot -> 8(need=3.0, L=739, T=770) -> 38(need=7.0, L=1149, T=1208) -> Depot

 tour:tensor([[ 8, 38, 0]])

 车辆 4: Depot -> 1(need=7.0, L=717, T=727) -> 39(need=3.0, L=774, T=862) -> 42(need=7.0, L=1065, T=1101) -> Depot

 tour:tensor([[ 1, 39, 42, 0]])

 车辆 21: Depot -> 20(need=7.0, L=682, T=692) -> 6(need=7.0, L=978, T=988) -> Depot

 tour:tensor([[20, 6, 0]])

车队 4:

 车辆 5: Depot -> 10(need=4.0, L=921, T=921) -> 24(need=8.0, L=1061, T=1065) -> Depot

 tour:tensor([[10, 24, 0]])

 车辆 14: Depot -> 46(need=4.0, L=996, T=1013) -> Depot

 tour:tensor([[46, 0]])

 车辆 23: Depot -> 12(need=8.0, L=629, T=629) -> 21(need=4.0, L=734, T=763) -> 7(need=8.0, L=1073, T=1104) -> Depot

 tour:tensor([[12, 21, 7, 0]])

 车辆 24: Depot -> 9(need=8.0, L=602, T=602) -> 49(need=8.0, L=1058, T=1064) -> Depot

 tour:tensor([[ 9, 49, 0]])

 车辆 30: Depot -> 4(need=4.0, L=680, T=682) -> 40(need=4.0, L=740, T=771) -> 33(need=4.0, L=923, T=923) -> 36(need=4.0, L=1035, T=1066) -> Depot

 tour:tensor([[ 4, 40, 33, 36, 0]])

车队 5:

 车辆 2: Depot -> 9(need=8.0, L=602, T=612) -> 18(need=5.0, L=999, T=999) -> 49(need=8.0, L=1058, T=1094) -> Depot

 tour:tensor([[ 9, 18, 49, 0]])

 车辆 6: Depot -> 44(need=5.0, L=709, T=725) -> 29(need=5.0, L=853, T=908) -> 22(need=5.0, L=1078, T=1166) -> Depot

 tour:tensor([[44, 29, 22, 0]])

 车辆 7: Depot -> 5(need=5.0, L=766, T=816) -> 24(need=8.0, L=1061, T=1078) -> 41(need=5.0, L=1080, T=1169) -> Depot

 tour:tensor([[ 5, 24, 41, 0]])

 车辆 10: Depot -> 43(need=5.0, L=1036, T=1123) -> Depot

 tour:tensor([[43, 0]])

 车辆 20: Depot -> 12(need=8.0, L=629, T=639) -> 7(need=8.0, L=1073, T=1134) -> Depot

 tour:tensor([[12, 7, 0]])

车队 6:

 车辆 1: Depot -> 13(need=9.0, L=751, T=809) -> 16(need=9.0, L=864, T=899) -> 15(need=6.0, L=1003, T=1003) -> 32(need=6.0, L=1064, T=1154) -> Depot

 tour:tensor([[13, 16, 15, 32, 0]])

 车辆 3: Depot -> 34(need=6.0, L=622, T=622) -> 25(need=9.0, L=995, T=995) -> 19(need=6.0, L=1118, T=1208) -> Depot

 tour:tensor([[34, 25, 19, 0]])

 车辆 14: Depot -> 23(need=6.0, L=817, T=817) -> 17(need=9.0, L=1053, T=1143) -> 30(need=9.0, L=1174, T=1264) -> Depot

 tour:tensor([[23, 17, 30, 0]])

--- 实例 7 求解结束，耗时 9.25 秒 ---



✓ 实例 7 求解成功: 2643.802637693061

Node-Service Mapping:

{1: '4&5', 2: '4.0', 3: '2&3', 4: '5.0', 5: '3.0', 6: '1&6', 7: '1.0', 8: '2.0', 9: '3.0', 10: '3.0', 11: '3.0', 12: '5.0', 13: '1&6', 14: '1.0', 15: '2&3', 16: '4&5', 17: '5.0', 18: '2.0', 19: '4&5', 20: '2&3', 21: '6.0', 22: '4&5', 23: '5.0', 24: '3.0', 25: '1.0', 26: '1&6', 27: '3.0', 28: '2&3', 29: '4.0', 30: '2.0', 31: '5.0', 32: '4&5', 33: '6.0', 34: '4&5', 35: '6.0', 36: '2&3', 37: '5.0', 38: '5.0', 39: '6.0', 40: '4.0', 41: '4.0', 42: '2.0', 43: '1&6', 44: '2&3', 45: '1&6', 46: '6.0', 47: '3.0', 48: '5.0', 49: '1.0', 50: '2&3'}

 80%|████████ | 8/10 [12:48<03:09, 95.00s/it]

--- 实例 8 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2860.2919555348817

Gap: 0.0

二进制变量数量: 478020

整数变量数量: 312

求解时间: 6.25 秒

分支定界节点数: 0

最佳界限: 2860.2919555348817



路径信息:

车队 1:

 车辆 1: Depot -> 7(need=1.0, L=719, T=719) -> 45(need=9.0, L=757, T=809) -> 13(need=9.0, L=869, T=899) -> 43(need=9.0, L=919, T=1009) -> 26(need=9.0, L=1168, T=1212) -> Depot

 tour:tensor([[ 7, 45, 13, 43, 26, 0]])

 车辆 2: Depot -> 14(need=1.0, L=651, T=651) -> 25(need=1.0, L=812, T=812) -> 6(need=9.0, L=1032, T=1108) -> 49(need=1.0, L=1198, T=1288) -> Depot

 tour:tensor([[14, 25, 6, 49, 0]])

车队 2:

 车辆 9: Depot -> 20(need=7.0, L=703, T=732) -> 28(need=7.0, L=1026, T=1026) -> 36(need=7.0, L=1181, T=1210) -> Depot

 tour:tensor([[20, 28, 36, 0]])

 车辆 11: Depot -> 42(need=2.0, L=696, T=696) -> Depot

 tour:tensor([[42, 0]])

 车辆 14: Depot -> 18(need=2.0, L=650, T=650) -> 30(need=2.0, L=888, T=888) -> Depot

 tour:tensor([[18, 30, 0]])

 车辆 18: Depot -> 8(need=2.0, L=693, T=693) -> Depot

 tour:tensor([[8, 0]])

 车辆 19: Depot -> 3(need=7.0, L=685, T=702) -> Depot

 tour:tensor([[3, 0]])

 车辆 28: Depot -> 44(need=7.0, L=663, T=663) -> 50(need=7.0, L=755, T=785) -> 15(need=7.0, L=915, T=915) -> Depot

 tour:tensor([[44, 50, 15, 0]])

车队 3:

 车辆 1: Depot -> 3(need=7.0, L=685, T=715) -> 15(need=7.0, L=915, T=925) -> Depot

 tour:tensor([[ 3, 15, 0]])

 车辆 2: Depot -> 47(need=3.0, L=639, T=639) -> 20(need=7.0, L=703, T=762) -> 28(need=7.0, L=1026, T=1036) -> Depot

 tour:tensor([[47, 20, 28, 0]])

 车辆 9: Depot -> 5(need=3.0, L=664, T=664) -> 11(need=3.0, L=1048, T=1053) -> 9(need=3.0, L=1166, T=1253) -> Depot

 tour:tensor([[ 5, 11, 9, 0]])

 车辆 19: Depot -> 27(need=3.0, L=666, T=666) -> 24(need=3.0, L=1058, T=1145) -> Depot

 tour:tensor([[27, 24, 0]])

 车辆 27: Depot -> 44(need=7.0, L=663, T=673) -> 50(need=7.0, L=755, T=815) -> 10(need=3.0, L=898, T=909) -> 36(need=7.0, L=1181, T=1240) -> Depot

 tour:tensor([[44, 50, 10, 36, 0]])

车队 4:

 车辆 5: Depot -> 1(need=8.0, L=705, T=705) -> 19(need=8.0, L=986, T=986) -> 22(need=8.0, L=1092, T=1118) -> Depot

 tour:tensor([[ 1, 19, 22, 0]])

 车辆 18: Depot -> 32(need=8.0, L=659, T=659) -> 29(need=4.0, L=1070, T=1099) -> 2(need=4.0, L=1179, T=1209) -> Depot

 tour:tensor([[32, 29, 2, 0]])

 车辆 19: Depot -> 40(need=4.0, L=670, T=670) -> Depot

 tour:tensor([[40, 0]])

 车辆 22: Depot -> 34(need=8.0, L=693, T=704) -> 41(need=4.0, L=866, T=866) -> 16(need=8.0, L=1124, T=1154) -> Depot

 tour:tensor([[34, 41, 16, 0]])

车队 5:

 车辆 21: Depot -> 37(need=5.0, L=647, T=647) -> 12(need=5.0, L=759, T=849) -> 31(need=5.0, L=1070, T=1091) -> Depot

 tour:tensor([[37, 12, 31, 0]])

 车辆 23: Depot -> 1(need=8.0, L=705, T=715) -> 48(need=5.0, L=894, T=894) -> 4(need=5.0, L=1185, T=1273) -> Depot

 tour:tensor([[ 1, 48, 4, 0]])

 车辆 24: Depot -> 34(need=8.0, L=693, T=714) -> 19(need=8.0, L=986, T=996) -> 16(need=8.0, L=1124, T=1184) -> Depot

 tour:tensor([[34, 19, 16, 0]])

 车辆 27: Depot -> 32(need=8.0, L=659, T=669) -> 23(need=5.0, L=673, T=761) -> 38(need=5.0, L=905, T=905) -> 17(need=5.0, L=1056, T=1056) -> 22(need=8.0, L=1092, T=1148) -> Depot

 tour:tensor([[32, 23, 38, 17, 22, 0]])

车队 6:

 车辆 12: Depot -> 39(need=6.0, L=749, T=749) -> 13(need=9.0, L=869, T=899) -> 33(need=6.0, L=968, T=1058) -> Depot

 tour:tensor([[39, 13, 33, 0]])

 车辆 16: Depot -> 21(need=6.0, L=737, T=736) -> 46(need=6.0, L=780, T=870) -> 6(need=9.0, L=1032, T=1108) -> Depot

 tour:tensor([[21, 46, 6, 0]])

 车辆 21: Depot -> 35(need=6.0, L=701, T=701) -> 45(need=9.0, L=757, T=809) -> 43(need=9.0, L=919, T=1009) -> 26(need=9.0, L=1168, T=1212) -> Depot

 tour:tensor([[35, 45, 43, 26, 0]])

--- 实例 8 求解结束，耗时 7.42 秒 ---



✓ 实例 8 求解成功: 2860.2919555348817

Node-Service Mapping:

{1: '2.0', 2: '4&5', 3: '2&3', 4: '2.0', 5: '4.0', 6: '4&5', 7: '6.0', 8: '1.0', 9: '1.0', 10: '2.0', 11: '5.0', 12: '2&3', 13: '1.0', 14: '4&5', 15: '2.0', 16: '6.0', 17: '1&6', 18: '6.0', 19: '2.0', 20: '6.0', 21: '5.0', 22: '4.0', 23: '1.0', 24: '4.0', 25: '6.0', 26: '1&6', 27: '1.0', 28: '4.0', 29: '4.0', 30: '1.0', 31: '1.0', 32: '1.0', 33: '6.0', 34: '4.0', 35: '2.0', 36: '4.0', 37: '4.0', 38: '1.0', 39: '3.0', 40: '4&5', 41: '5.0', 42: '1.0', 43: '2&3', 44: '4.0', 45: '2.0', 46: '1.0', 47: '5.0', 48: '2.0', 49: '2.0', 50: '3.0'}

 90%|█████████ | 9/10 [14:18<01:33, 93.57s/it]

--- 实例 9 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2654.5494799846465

Gap: 0.0

二进制变量数量: 478020

整数变量数量: 312

求解时间: 4.55 秒

分支定界节点数: 0

最佳界限: 2654.5494799846465



路径信息:

车队 1:

 车辆 3: Depot -> 38(need=1.0, L=664, T=664) -> 26(need=9.0, L=989, T=989) -> 9(need=1.0, L=1169, T=1194) -> Depot

 tour:tensor([[38, 26, 9, 0]])

 车辆 4: Depot -> 46(need=1.0, L=694, T=744) -> 13(need=1.0, L=924, T=924) -> 42(need=1.0, L=1116, T=1116) -> 32(need=1.0, L=1137, T=1227) -> Depot

 tour:tensor([[46, 13, 42, 32, 0]])

 车辆 6: Depot -> 30(need=1.0, L=604, T=604) -> 23(need=1.0, L=770, T=860) -> 8(need=1.0, L=1194, T=1284) -> Depot

 tour:tensor([[30, 23, 8, 0]])

 车辆 15: Depot -> 27(need=1.0, L=684, T=684) -> 17(need=9.0, L=1067, T=1067) -> 31(need=1.0, L=1105, T=1195) -> Depot

 tour:tensor([[27, 17, 31, 0]])

车队 2:

 车辆 1: Depot -> 43(need=7.0, L=802, T=831) -> 49(need=2.0, L=917, T=922) -> 35(need=2.0, L=1180, T=1209) -> Depot

 tour:tensor([[43, 49, 35, 0]])

 车辆 2: Depot -> 15(need=2.0, L=734, T=734) -> 3(need=7.0, L=990, T=1019) -> 48(need=2.0, L=1117, T=1119) -> Depot

 tour:tensor([[15, 3, 48, 0]])

 车辆 6: Depot -> 1(need=2.0, L=929, T=929) -> 19(need=2.0, L=1023, T=1026) -> 12(need=7.0, L=1144, T=1144) -> Depot

 tour:tensor([[ 1, 19, 12, 0]])

 车辆 15: Depot -> 45(need=2.0, L=772, T=772) -> 4(need=2.0, L=932, T=932) -> 10(need=2.0, L=1152, T=1152) -> Depot

 tour:tensor([[45, 4, 10, 0]])

车队 3:

 车辆 2: Depot -> 39(need=3.0, L=744, T=744) -> 50(need=3.0, L=950, T=950) -> 3(need=7.0, L=990, T=1049) -> 12(need=7.0, L=1144, T=1154) -> Depot

 tour:tensor([[39, 50, 3, 12, 0]])

 车辆 6: Depot -> 43(need=7.0, L=802, T=841) -> Depot

 tour:tensor([[43, 0]])

车队 4:

 车辆 1: Depot -> 37(need=4.0, L=709, T=709) -> 22(need=4.0, L=787, T=800) -> 6(need=8.0, L=961, T=961) -> 14(need=8.0, L=1099, T=1130) -> Depot

 tour:tensor([[37, 22, 6, 14, 0]])

 车辆 3: Depot -> 44(need=4.0, L=659, T=659) -> 2(need=8.0, L=755, T=783) -> 34(need=4.0, L=896, T=896) -> 40(need=8.0, L=1095, T=1125) -> Depot

 tour:tensor([[44, 2, 34, 40, 0]])

 车辆 4: Depot -> 5(need=4.0, L=699, T=699) -> 24(need=4.0, L=777, T=800) -> 29(need=4.0, L=1032, T=1032) -> Depot

 tour:tensor([[ 5, 24, 29, 0]])

 车辆 25: Depot -> 36(need=4.0, L=873, T=889) -> 28(need=4.0, L=1180, T=1209) -> Depot

 tour:tensor([[36, 28, 0]])

车队 5:

 车辆 2: Depot -> 21(need=5.0, L=700, T=707) -> 6(need=8.0, L=961, T=971) -> 11(need=5.0, L=992, T=1061) -> 14(need=8.0, L=1099, T=1160) -> Depot

 tour:tensor([[21, 6, 11, 14, 0]])

 车辆 8: Depot -> 41(need=5.0, L=616, T=616) -> 40(need=8.0, L=1095, T=1155) -> Depot

 tour:tensor([[41, 40, 0]])

 车辆 12: Depot -> 2(need=8.0, L=755, T=813) -> 47(need=5.0, L=1187, T=1276) -> Depot

 tour:tensor([[ 2, 47, 0]])

车队 6:

 车辆 11: Depot -> 25(need=6.0, L=603, T=603) -> 33(need=6.0, L=673, T=693) -> 26(need=9.0, L=989, T=989) -> Depot

 tour:tensor([[25, 33, 26, 0]])

 车辆 22: Depot -> 16(need=6.0, L=892, T=892) -> 20(need=6.0, L=945, T=1003) -> Depot

 tour:tensor([[16, 20, 0]])

 车辆 24: Depot -> 18(need=6.0, L=679, T=712) -> 7(need=6.0, L=733, T=823) -> 17(need=9.0, L=1067, T=1067) -> Depot

 tour:tensor([[18, 7, 17, 0]])

--- 实例 9 求解结束，耗时 5.70 秒 ---



✓ 实例 9 求解成功: 2654.5494799846465

Node-Service Mapping:

{1: '2&3', 2: '6.0', 3: '6.0', 4: '4.0', 5: '4.0', 6: '5.0', 7: '4&5', 8: '1&6', 9: '3.0', 10: '1.0', 11: '1.0', 12: '1.0', 13: '2.0', 14: '2&3', 15: '6.0', 16: '5.0', 17: '1&6', 18: '1&6', 19: '4.0', 20: '1.0', 21: '6.0', 22: '2&3', 23: '1&6', 24: '5.0', 25: '5.0', 26: '2.0', 27: '4&5', 28: '2.0', 29: '2.0', 30: '2.0', 31: '2.0', 32: '6.0', 33: '4&5', 34: '3.0', 35: '3.0', 36: '6.0', 37: '3.0', 38: '6.0', 39: '2&3', 40: '1&6', 41: '2&3', 42: '4.0', 43: '1.0', 44: '3.0', 45: '5.0', 46: '3.0', 47: '1.0', 48: '3.0', 49: '2.0', 50: '4.0'}

100%|██████████| 10/10 [15:49<00:00, 94.90s/it]



--- 实例 10 开始求解 ---

CPLEX 求解完成!

求解状态: JobSolveStatus.OPTIMAL_SOLUTION

状态解释: 最优解

目标函数值: 2462.5500139616947

Gap: 0.00028518445561243245

二进制变量数量: 478020

整数变量数量: 312

求解时间: 5.94 秒

分支定界节点数: 0

最佳界限: 2461.8477329765446



路径信息:

车队 1:

 车辆 15: Depot -> 10(need=1.0, L=980, T=980) -> Depot

 tour:tensor([[10, 0]])

 车辆 23: Depot -> 12(need=1.0, L=668, T=668) -> 20(need=1.0, L=693, T=783) -> 23(need=9.0, L=876, T=911) -> 11(need=1.0, L=1137, T=1227) -> Depot

 tour:tensor([[12, 20, 23, 11, 0]])

 车辆 25: Depot -> 43(need=1.0, L=904, T=904) -> 8(need=9.0, L=908, T=994) -> 40(need=9.0, L=1070, T=1084) -> Depot

 tour:tensor([[43, 8, 40, 0]])

 车辆 28: Depot -> 18(need=9.0, L=606, T=606) -> 47(need=1.0, L=670, T=760) -> 17(need=9.0, L=1041, T=1041) -> Depot

 tour:tensor([[18, 47, 17, 0]])

车队 2:

 车辆 1: Depot -> 49(need=2.0, L=662, T=662) -> 26(need=2.0, L=772, T=772) -> 41(need=7.0, L=850, T=863) -> Depot

 tour:tensor([[49, 26, 41, 0]])

 车辆 2: Depot -> 39(need=7.0, L=797, T=815) -> 29(need=2.0, L=926, T=955) -> 22(need=7.0, L=1136, T=1136) -> Depot

 tour:tensor([[39, 29, 22, 0]])

 车辆 6: Depot -> 31(need=2.0, L=806, T=806) -> 28(need=2.0, L=891, T=920) -> Depot

 tour:tensor([[31, 28, 0]])

 车辆 20: Depot -> 14(need=7.0, L=674, T=700) -> 30(need=2.0, L=776, T=791) -> 1(need=7.0, L=927, T=956) -> Depot

 tour:tensor([[14, 30, 1, 0]])

 车辆 24: Depot -> 13(need=2.0, L=757, T=760) -> Depot

 tour:tensor([[13, 0]])

车队 3:

 车辆 5: Depot -> 37(need=3.0, L=633, T=633) -> 41(need=7.0, L=850, T=893) -> 22(need=7.0, L=1136, T=1146) -> Depot

 tour:tensor([[37, 41, 22, 0]])

 车辆 7: Depot -> 34(need=3.0, L=638, T=638) -> 48(need=3.0, L=923, T=929) -> 35(need=3.0, L=935, T=1023) -> 46(need=3.0, L=1125, T=1125) -> 44(need=3.0, L=1165, T=1251) -> Depot

 tour:tensor([[34, 48, 35, 46, 44, 0]])

 车辆 22: Depot -> 9(need=3.0, L=638, T=638) -> 14(need=7.0, L=674, T=730) -> 39(need=7.0, L=797, T=825) -> 1(need=7.0, L=927, T=986) -> Depot

 tour:tensor([[ 9, 14, 39, 1, 0]])

车队 4:

 车辆 2: Depot -> 42(need=4.0, L=661, T=661) -> 5(need=4.0, L=733, T=763) -> 50(need=4.0, L=1179, T=1179) -> Depot

 tour:tensor([[42, 5, 50, 0]])

 车辆 3: Depot -> 33(need=8.0, L=699, T=699) -> 4(need=4.0, L=770, T=799) -> 7(need=8.0, L=1141, T=1141) -> Depot

 tour:tensor([[33, 4, 7, 0]])

 车辆 7: Depot -> 19(need=4.0, L=876, T=906) -> 27(need=8.0, L=1180, T=1180) -> Depot

 tour:tensor([[19, 27, 0]])

车队 5:

 车辆 13: Depot -> 33(need=8.0, L=699, T=709) -> 16(need=5.0, L=818, T=818) -> 45(need=5.0, L=886, T=910) -> 7(need=8.0, L=1141, T=1151) -> Depot

 tour:tensor([[33, 16, 45, 7, 0]])

 车辆 16: Depot -> 25(need=5.0, L=966, T=1054) -> 27(need=8.0, L=1180, T=1190) -> Depot

 tour:tensor([[25, 27, 0]])

 车辆 18: Depot -> 6(need=5.0, L=703, T=703) -> 24(need=5.0, L=1192, T=1282) -> Depot

 tour:tensor([[ 6, 24, 0]])

车队 6:

 车辆 2: Depot -> 36(need=6.0, L=638, T=638) -> 23(need=9.0, L=876, T=911) -> 32(need=6.0, L=1001, T=1001) -> Depot

 tour:tensor([[36, 23, 32, 0]])

 车辆 20: Depot -> 18(need=9.0, L=606, T=606) -> 21(need=6.0, L=761, T=830) -> 17(need=9.0, L=1041, T=1041) -> 2(need=6.0, L=1072, T=1141) -> 3(need=6.0, L=1141, T=1231) -> Depot

 tour:tensor([[18, 21, 17, 2, 3, 0]])

 车辆 21: Depot -> 15(need=6.0, L=613, T=613) -> 38(need=6.0, L=650, T=740) -> 8(need=9.0, L=908, T=994) -> 40(need=9.0, L=1070, T=1084) -> Depot

 tour:tensor([[15, 38, 8, 40, 0]])

--- 实例 10 求解结束，耗时 7.17 秒 ---



✓ 实例 10 求解成功: 2462.5500139616947



============================================================

所有实例求解完成!

============================================================

成功求解: 10/10

所有实例的成本列表: [2748.675499621526, 2620.127290852651, 2680.254980160206, 2344.2510480052933, 3046.7118074343666, 2777.431808944002, 2643.802637693061, 2860.2919555348817, 2654.5494799846465, 2462.5500139616947]



结果统计:

  平均成本: 2683.86

  最小成本: 2344.25

  最大成本: 3046.71

  标准差: 196.24

\>> 验证结束，总耗时 949.06 秒
