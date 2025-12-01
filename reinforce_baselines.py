import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from train import rollout, get_inner_model

# === 基线基类 ===
class Baseline(object):
    """基线抽象基类，定义基线接口。
    - 用于 REINFORCE 优化，降低策略梯度方差。
    """
    def wrap_dataset(self, dataset):
        """包装数据集，添加基线值。
        - dataset: 输入数据集
        - 返回: 包装后的数据集
        """
        return dataset

    def unwrap_batch(self, batch):
        """解包批次数据，返回数据和基线值。
        - batch: 批次数据
        - 返回: (数据, 基线值)
        """
        return batch, None

    def eval(self, x, c):
        """计算基线值和损失。
        - x: 输入数据
        - c: 实际成本
        - 返回: (基线值, 基线损失)
        """
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        """获取可训练参数。
        - 返回: 参数列表
        """
        return []

    def epoch_callback(self, model, epoch):
        """每个 epoch 后的回调函数。
        - model: 当前模型
        - epoch: 当前 epoch
        """
        pass

    def state_dict(self):
        """获取基线状态字典。
        - 返回: 状态字典
        """
        return {}

    def load_state_dict(self, state_dict):
        """加载基线状态字典。
        - state_dict: 状态字典
        """
        pass

# === 预热基线 ===
class WarmupBaseline(Baseline):
    """预热基线，结合指数基线和主基线，逐步过渡。
    - baseline: 主基线（如 Critic）
    - n_epochs: 预热 epoch 数
    - warmup_exp_beta: 指数基线的衰减因子
    """
    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        super(Baseline, self).__init__()
        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)  # 指数基线
        self.alpha = 0  # 预热权重
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        """根据 alpha 选择基线包装数据集。
        - dataset: 输入数据集
        - 返回: 包装后的数据集
        """
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        """根据 alpha 解包批次数据。
        - batch: 批次数据
        - 返回: (数据, 基线值)
        """
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        """计算加权基线值和损失。
        - x: 输入数据
        - c: 实际成本
        - 返回: (加权基线值, 加权基线损失)
        """
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        """更新 alpha 并调用主基线回调。
        - model: 当前模型
        - epoch: 当前 epoch
        """
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)  # 线性增加 alpha
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        """获取主基线状态字典。
        - 返回: 状态字典
        """
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        """加载主基线状态字典。
        - state_dict: 状态字典
        """
        self.baseline.load_state_dict(state_dict)

# === 无基线 ===
class NoBaseline(Baseline):
    """无基线，直接返回 0。
    - 用于调试或不需基线的场景。
    """
    def eval(self, x, c):
        """返回零基线值和损失。
        - x: 输入数据
        - c: 实际成本
        - 返回: (0, 0)
        """
        return 0, 0

# === 指数基线 ===
class ExponentialBaseline(Baseline):
    """指数移动平均基线，跟踪成本均值。
    - beta: 衰减因子
    """
    def __init__(self, beta):
        super(Baseline, self).__init__()
        self.beta = beta
        self.v = None  # 移动平均值

    def eval(self, x, c):
        """计算指数移动平均基线值。
        - x: 输入数据
        - c: 实际成本
        - 返回: (基线值, 0)
        """
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()
        self.v = v.detach()  # 断开梯度
        return self.v, 0  # 无基线损失

    def state_dict(self):
        """获取基线状态字典。
        - 返回: 包含移动平均值的字典
        """
        return {'v': self.v}

    def load_state_dict(self, state_dict):
        """加载基线状态字典。
        - state_dict: 包含移动平均值的字典
        """
        self.v = state_dict['v']

# === Critic 基线 ===
class CriticBaseline(Baseline):
    """基于 Critic 网络的基线，预测状态价值。
    - critic: Critic 网络（如 CriticNetworkLSTM）
    """
    def __init__(self, critic):
        super(Baseline, self).__init__()
        self.critic = critic

    def eval(self, x, c):
        """计算 Critic 基线值和 MSE 损失。
        - x: 输入数据
        - c: 实际成本
        - 返回: (基线值, MSE 损失)
        """
        v = self.critic(x)  # 预测价值
        return v.detach(), F.mse_loss(v, c.detach())  # 断开梯度，计算 MSE

    def get_learnable_parameters(self):
        """获取 Critic 网络的可训练参数。
        - 返回: 参数列表
        """
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        """Critic 基线的 epoch 回调（空实现）。
        - model: 当前模型
        - epoch: 当前 epoch
        """
        pass

    def state_dict(self):
        """获取 Critic 网络状态字典。
        - 返回: 包含 Critic 状态的字典
        """
        return {'critic': self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        """加载 Critic 网络状态字典。
        - state_dict: 包含 Critic 状态的字典
        """
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # 向后兼容
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})

# === Rollout 基线 ===
class RolloutBaseline(Baseline):
    """基于模型 Rollout 的基线，使用贪婪解成本。
    - model: 基线模型
    - problem: 问题定义（TSP 或 AGH）
    - opts: 配置选项
    - epoch: 初始 epoch
    """
    def __init__(self, model, problem, opts, epoch=0):
        super(Baseline, self).__init__()
        self.problem = problem
        self.opts = opts
        self._update_model(model, epoch)  # 初始化基线模型

    def _update_model(self, model, epoch, dataset=None):
        """更新基线模型和数据集。
        - model: 新模型
        - epoch: 当前 epoch
        - dataset: 可选的验证数据集
        """
        self.model = copy.deepcopy(model)  # 深拷贝模型
        if dataset is not None:
            if len(dataset) != self.opts.val_size or (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since val_size or graph_size does not match")
                dataset = None
        if dataset is None:
            self.dataset = self.problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset\n")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()  # 计算基线成本
        if self.model.is_agh:
            self.bl_vals = self.bl_vals.sum(1)  # AGH：对车队成本求和
        self.mean = self.bl_vals.mean()  # 平均成本
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        """包装数据集，添加基线值。
        - dataset: 输入数据集
        - 返回: 包装后的数据集
        """
        print("Evaluating baseline on dataset...")
        if self.model.is_agh:
            print("Training baseline model...")
            output = BaselineDataset(dataset, rollout(self.model, dataset, self.opts))
            return BaselineDataset(dataset, rollout(self.model, dataset, self.opts))
        else:
            return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        """解包批次数据，返回数据和基线值。
        - batch: 批次数据
        - 返回: (数据, 基线值)
        """
        if self.model.is_agh:
            return batch['data'], batch['baseline']  # [batch_size, fleet_size=10]
        else:
            return batch['data'], batch['baseline'].view(-1)  # 展平基线值

    def eval(self, x, c):
        """使用基线模型计算成本。
        - x: 输入数据
        - c: 实际成本
        - 返回: (基线成本, 0)
        """
        with torch.no_grad():
            v, _ = self.model(x)  # 贪婪解码成本
        return v, 0  # 无基线损失

    def epoch_callback(self, model, epoch):
        """挑战基线模型，若新模型更优则更新。
        - model: 当前模型
        - epoch: 当前 epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        if model.is_agh:
            candidate_vals = candidate_vals.sum(1)
        candidate_mean = candidate_vals.mean()
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:  # 新模型更优
            t, p = ttest_rel(candidate_vals, self.bl_vals)  # 统计检验
            p_val = p / 2  # 单侧 p 值
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:  # p 值小于阈值
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        """获取基线状态字典。
        - 返回: 包含模型、数据集和 epoch 的字典
        """
        return {'model': self.model, 'dataset': self.dataset, 'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        """加载基线状态字典。
        - state_dict: 包含模型、数据集和 epoch 的字典
        """
        load_model = copy.deepcopy(self.model)
        load_model_ = get_inner_model(load_model)
        load_model_.load_state_dict({**load_model_.state_dict(), **state_dict['model'].state_dict()})
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])

# === 基线数据集 ===
class BaselineDataset(Dataset):
    """包装数据集，包含数据和基线值。
    - dataset: 原始数据集
    - baseline: 基线值
    """
    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        """获取单个样本。
        - item: 索引
        - 返回: 包含数据和基线值的字典
        """
        return {'data': self.dataset[item], 'baseline': self.baseline[item]}

    def __len__(self):
        """获取数据集长度。
        - 返回: 数据集大小
        """
        return len(self.dataset)