import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

# === 获取模型核心 ===
def get_inner_model(model):
    """获取模型的核心实例，处理 DataParallel 包裹情况。
    - model: 模型实例（可能被 DataParallel 包裹）
    - 返回: 核心模型实例
    """
    return model.module if isinstance(model, DataParallel) else model

# === 验证函数 ===
def validate(model, dataset, opts):
    """验证模型性能，计算平均成本。
    - model: AttentionModel 实例
    - dataset: 验证数据集
    - opts: 配置选项（包含设备、批次大小等）
    - 返回: 平均成本
    """
    print('Validating...')
    cost = rollout(model, dataset, opts)  # 执行 rollout 评估
    if model.is_agh:
        cost = cost.sum(1)  # AGH：对所有车队成本求和
    avg_cost = cost.mean()  # 计算平均成本
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost

# === Rollout 评估 ===
def rollout(model, dataset, opts):
    set_decode_type(model, "greedy")
    model.eval()

    if model.is_agh:
        cost = []
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            bat_cost = []

            bat_tw_left = bat['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1).to(opts.device)
            bat_tw_right = bat['departure']
            need = bat['need']

            for f in model.fleet_info['order']:
                # print(f"fleet {f}")
                # 使用 type_as 保持与 bat['loc'] 同设备
                next_duration = torch.tensor(model.fleet_info['next_duration'][model.fleet_info['precedence'][f]]) \
                    .repeat(bat['loc'].size(0), 1).type_as(bat['loc'])

                tw_right = bat_tw_right - torch.gather(next_duration, 1, bat['type'])
                tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)

                tw_left = bat_tw_left[model.fleet_info['precedence'][f]]
                tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)

                duration = torch.tensor(model.fleet_info['duration'][f]) \
                    .repeat(bat['loc'].size(0), 1).type_as(bat['loc'])

                if f == 1:
                    mask = (need == 1) | (need == 9)  # 1单 + 1,6组合
                elif f == 2:
                    mask = (need == 2) | (need == 7)  # 2单 + 2,3组合
                elif f == 3:
                    mask = (need == 3) | (need == 7)  # 3单 + 2,3组合 #
                elif f == 4:
                    mask = (need == 4) | (need == 8)  # 4单 + 4,5组合
                elif f == 5:
                    mask = (need == 5) | (need == 8)  # 5单 + 4,5组合 #
                elif f == 6:
                    mask = (need == 6) | (need == 9)  # 6单 + 1,6组合 ##
                else:
                    mask = (need == f)  # 其他预留

                # 滤掉登机口节点
                tw_right_filtered = tw_right.clone()
                tw_right_filtered[:, 1:] = tw_right[:, 1:] * mask.type_as(tw_right).float()

                tw_left_filtered = tw_left.clone()
                tw_left_filtered[:, 1:] = tw_left[:, 1:] * mask.type_as(tw_left).float()

                # need掩码
                need_filtered = need.clone()
                need_filtered = need_filtered * mask.type_as(need).float()

                fleet_bat = {
                    'loc': bat['loc'],
                    'distance': model.distance.expand(bat['loc'].size(0), len(model.distance)),
                    'duration': torch.gather(duration, 1, bat['type']),
                    'tw_right': tw_right_filtered,
                    'tw_left': tw_left_filtered,
                    'fleet': torch.full((bat['loc'].size(0), 1), f - 1).type_as(bat['loc']),
                    'need': need_filtered,
                }

                if model.rnn_time:
                    model.pre_tw = None

                with torch.no_grad():
                    # print(f"fleet {f}")
                    fleet_cost, _, serve_time = model(move_to(fleet_bat, opts.device))
                bat_cost.append(fleet_cost.data.cpu().view(-1, 1))

                # 这个是在double service中根据优先级更新bat_tw_left的，暂时不用
                # # 获取下一个阶段的索引
                # next_stage = model.fleet_info['precedence'][f] + 1
                # # 当前阶段实际服务的节点 mask（跳过 depot，即从第 1 列开始）
                # real_mask = mask.to(opts.device) # [batch_size, graph_size]
                # # 旧的 tw_left 值
                # prev_tw_left = bat_tw_left[next_stage]  # [batch_size, graph_size]
                # # 当前 serve_time（从第 1 列开始）
                # cur_serve_time = serve_time[:, 1:]  # [batch_size, graph_size]
                # print(f"cur_serve_time: {cur_serve_time[:5]}")
                # # 只更新被服务节点的时间窗
                # updated_tw_left = torch.where(real_mask, torch.max(prev_tw_left, cur_serve_time), prev_tw_left)
                # # 覆盖更新
                # bat_tw_left[next_stage] = updated_tw_left

                next_stage = model.fleet_info['precedence'][f] + 1
                mask = mask.to(opts.device)  # [batch_size, graph_size]
                if f == 1:
                    # f=1 时的特殊逻辑：不加10
                    bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:], bat_tw_left[next_stage])
                else:
                    # 其他情况的原有逻辑：加10
                    bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:] + 10, bat_tw_left[next_stage])

            bat_cost = torch.cat(bat_cost, 1)
            cost.append(bat_cost)
        return torch.cat(cost, 0)

    def eval_model_bat(batch):
        with torch.no_grad():
            cost_, _ = model(move_to(batch, opts.device))
        return cost_.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

# === 梯度裁剪 ===
def clip_grad_norms(param_groups, max_norm=math.inf):
    """裁剪梯度范数，防止梯度爆炸。
    - param_groups: 优化器的参数组
    - max_norm: 最大范数（默认无限制）
    - 返回: 未裁剪和裁剪后的梯度范数
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

# === 训练一个 Epoch ===
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
    # 这里的wrap_dataset还要调用一次rollout,目的是计算基线b_vals
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    # 这里生成数据是没有带filename参数的
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    model.train()  # 切换到训练模式
    set_decode_type(model, "sampling")  # 使用采样解码

    total_batches = len(training_dataloader)
    print(f"开始train_epoch正式训练，总的 batch 大小: {total_batches}")
    print("Start training epoch {}...".format(epoch))
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        # print(f"batch_id {batch_id}")
        if model.is_agh:
            train_batch_agh(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        else:
            train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # 保存模型检查点
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('####Saving model and state...')
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
    print("调用validate")
    avg_reward = validate(model, val_dataset, opts)

    # 记录验证结果
    with open(os.path.join(opts.save_dir, 'validate_log.txt'), 'a') as f:
        f.write('####Validating Epoch {}, Validation avg_cost: {}\n'.format(epoch, avg_reward))
        f.write('\n')

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)  # 记录验证奖励

    baseline.epoch_callback(model, epoch)  # 基线回调

    lr_scheduler.step()  # 更新学习率

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
    # print(x.keys(),x)
    assert bl_val is not None
    x = move_to(x, opts.device)  # 移动到设备
    bl_val = move_to(bl_val, opts.device)

    set_decode_type(model, "sampling")  # 使用采样解码
    # 初始化时间窗口
    bat_tw_left = x['arrival'].repeat(len(model.fleet_info['next_duration']) + 1, 1, 1) # 全局时间窗口左边界
    bat_tw_right = x['departure'] # 全局时间窗口右边界
    need = x['need']
    fleet_cost_together, log_likelihood_together, fleet_cost_list, log_likelihood_list = None, None, [], []

    for f in model.fleet_info['order']:  # 按车队优先级
        # print(f"train_batch_agh下 车队: {f}")
        # 构造车队输入
        # print(model.fleet_info['precedence'][f])
        next_duration = torch.tensor(model.fleet_info['next_duration'][model.fleet_info['precedence'][f]],
                                    device=x['type'].device).repeat(x['loc'].size(0), 1)
        tw_right = bat_tw_right - torch.gather(next_duration, 1, x['type'])
        # torch.gather(next_duration, 1, x['type'])，从 next_duration 中沿 dim=1（类型维度）根据 x['type'] 提取值
        tw_right = torch.cat((torch.full_like(tw_right[:, :1], 1441), tw_right), dim=1)# 添加车库时间窗

        tw_left = bat_tw_left[model.fleet_info['precedence'][f]]# 单个车队时间窗左边界=全局时间窗口左边界
        tw_left = torch.cat((torch.zeros_like(tw_left[:, :1]), tw_left), dim=1)# 添加车队时间窗
        duration = torch.tensor(model.fleet_info['duration'][f], device=x['type'].device).repeat(x['loc'].size(0), 1)

        if f == 1:
            mask = (need == 1) | (need == 9)  # 1单 + 1,6组合
        elif f == 2:
            mask = (need == 2) | (need == 7)  # 2单 + 2,3组合
        elif f == 3:
            mask = (need == 3) | (need == 7)  # 3单 + 2,3组合#
        elif f == 4:
            mask = (need == 4) | (need == 8)  # 4单 + 4,5组合
        elif f == 5:
            mask = (need == 5) | (need == 8)  # 5单 + 4,5组合#
        elif f == 6:
            mask = (need == 6) | (need == 9)  # 6单 + 1,6组合#
        else:
            mask = (need == f)  # 其他预留

        # 滤掉登机口节点
        tw_right_filtered = tw_right.clone()
        tw_right_filtered[:, 1:] = tw_right[:, 1:] * mask.float()

        tw_left_filtered = tw_left.clone()
        tw_left_filtered[:, 1:] = tw_left[:, 1:] * mask.float()

        # need掩码
        need_filtered = need.clone()
        need_filtered = need_filtered * mask.type_as(need).float()


        fleet_bat = {'loc': x['loc'],
                     'distance': model.distance.expand(x['loc'].size(0), len(model.distance)),
                     'duration': torch.gather(duration, 1, x['type']),
                     'tw_right': tw_right_filtered,
                     'tw_left': tw_left_filtered,
                     'fleet': torch.full((x['loc'].size(0), 1), f - 1),
                     'need': need_filtered,
                     }

        if model.rnn_time:
            model.pre_tw = None  # 重置 RNN 隐藏状态

        # 前向传播
        # print(f"fleet {f}")
        fleet_cost, log_likelihood, serve_time = model(move_to(fleet_bat, opts.device))# 返回的是车队距离成本、对数似然、车队服务时间

        # 收集成本和对数似然
        fleet_cost_list.append(fleet_cost)
        log_likelihood_list.append(log_likelihood)

        if fleet_cost_together is None:
            fleet_cost_together, log_likelihood_together = fleet_cost, log_likelihood
        else:
            fleet_cost_together = fleet_cost_together + fleet_cost
            log_likelihood_together = log_likelihood_together + log_likelihood

        next_stage = model.fleet_info['precedence'][f] + 1
        mask = mask.to(opts.device)  # [batch_size, graph_size]
        if f == 1:
            # f=1 时的特殊逻辑：不加10
            bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:], bat_tw_left[next_stage])
        else:
            # 其他情况的原有逻辑：加10
            bat_tw_left[next_stage] = torch.where(mask, serve_time[:, 1:] + 10, bat_tw_left[next_stage])

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

# === 训练非 AGH 批次 ===
def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    """训练非 AGH 批次，使用 REINFORCE 优化。
    - 参数同 train_batch_agh
    """
    x, bl_val = baseline.unwrap_batch(batch)  # 解包数据和基线值
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # 前向传播
    cost, log_likelihood = model(x)

    # 计算基线值和损失
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # 计算 REINFORCE 损失
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss  # 总损失

    optimizer.zero_grad()
    loss.backward()  # 反向传播
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)  # 裁剪梯度
    optimizer.step()  # 更新参数

    # 记录日志
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)