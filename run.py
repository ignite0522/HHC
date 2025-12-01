import os
import json
import pprint as pp

import random
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
# 释放 GPU 缓存
torch.cuda.empty_cache()
# 指定空闲 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def run(opts):
    # 打印运行参数
    # Pretty print the run args
    global optimizer
    pp.pprint(vars(opts))

    # 设置随机种子
    # Set the random seed
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    # 可选配置 TensorBoard
    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    # 创建保存目录
    os.makedirs(opts.save_dir)
    # 保存参数以便随时找到确切的配置
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # 设置设备
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # 确定问题类型
    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # 从加载路径加载数据
    # Load data from load_path
    load_data = {}
    # 确保加载路径和恢复路径只有一个被指定
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # 初始化模型，这里在选择使用哪个模型
    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        wo_time=opts.wo_time,
        rnn_time=opts.rnn_time
    ).to(opts.device)

    # 使用要加载的参数覆盖模型参数
    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # 初始化基线，就是在选择使用哪个基线
    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        # 确保问题类型为 TSP
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        print("加载 rollout baseline")
        baseline = RolloutBaseline(model, problem, opts)
    else:
        # 确保基线类型为空或已知类型
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()


    # 从数据中加载基线，确保脚本调用时使用相同的基线类型
    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # 初始化优化器
    # Initialize optimizer

    if opts.optimizer == "Adam":
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
    elif opts.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
    elif opts.optimizer == 'SGD':
        optimizer = optim.SGD(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
    else:
        optimizer = None
        print("Optimizer {} not support ... ".format(opts.optimizer))
        exit()

    # 加载优化器状态
    # Load optimizer state
    if 'optimizer' in load_data and opts.fine_tune is False:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # 初始化学习率调度器，每轮衰减一次学习率
    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # 开始实际的训练循环


    # 生成验证数据集，生成验证集带了filename参数，在opts.val_dataset中没有指定文件类似于agh20_validation_seed7654321.pkl文件的话，默认自动生成
    val_dataset = problem.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution)


    if opts.resume:
        print("resuming")
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        if opts.fine_tune is False:
            torch.set_rng_state(load_data['rng_state'])
            if opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # 设置随机状态
        # Set the random states
        # 在 epoch 回调之前完成状态转储，现在执行（模型已加载）
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        print("Evaluating")
        assert load_path is not None, "No checkpoint to load!"
        validate(model, val_dataset, opts)
    else:
        print("Training")
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts)


if __name__ == "__main__":
    run(get_options())