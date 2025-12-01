from torch.utils.data import Dataset  # 导入PyTorch数据集基类
import torch  # 导入PyTorch库
import os  # 导入操作系统功能，处理文件路径
import pickle  # 导入pickle，用于数据序列化/反序列化
from problems.tsp.state_tsp import StateTSP # 导入TSP问题状态定义
from utils.beam_search import beam_search # 导入束搜索算法


class TSP(object):  # 定义TSP问题相关的类

    NAME = 'tsp'  # 问题名称

    @staticmethod
    def get_costs(dataset, pi):
        """
        计算旅行路径的总成本（距离）。
        验证路径有效性，按路径顺序收集节点坐标，然后计算所有相邻节点距离之和，并加上起点终点距离。
        """
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"  # 检查路径是否有效（包含所有节点且无重复）

        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))  # 按照路径顺序收集节点坐标

        # 计算所有相邻节点距离之和，并加上最后一个节点回到第一个节点的距离
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        """创建并返回TSP数据集实例。"""
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        """创建并初始化TSP问题状态实例。"""
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        """
        使用束搜索算法寻找TSP问题的近似最优解。
        依赖于提供的模型来提议每一步的扩展（选择下一个节点）。
        """
        assert model is not None, "Provide model"  # 确保模型已提供

        fixed = model.precompute_fixed(input)  # 模型预计算固定部分

        def propose_expansions(beam):
            """定义模型如何提议束的扩展（即下一步选择）。"""
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )  # 初始化TSP问题状态

        return beam_search(state, beam_size, propose_expansions)  # 执行束搜索


class TSPDataset(Dataset):  # 定义TSP数据集类，继承自PyTorch的Dataset

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        """
        TSP数据集初始化。
        可以从.pkl文件加载数据，或者随机生成指定数量和大小的TSP实例（节点坐标）。
        """
        super(TSPDataset, self).__init__()

        self.data_set = []  # 冗余列表
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'  # 检查文件类型
            with open(filename, 'rb') as f:
                data = pickle.load(f)  # 从文件加载数据
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]  # 加载指定范围的数据
        else:
            # 随机生成在[0,1]正方形内的节点坐标
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)  # 记录数据集大小

    def __len__(self):
        """返回数据集中的样本数量。"""
        return self.size

    def __getitem__(self, idx):
        """根据索引获取单个样本数据。"""
        return self.data[idx]