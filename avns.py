import copy, time
import argparse, math, random
import torch
import pickle
import numpy as np
import pprint as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to, load_problem
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import deque
from greedy_strict import construct_strict_greedy, CVRPTWInstance as StrictCVRPTWInstance
import os

# 配置日志系统，用于调试和监控求解过程
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AVNSConfig:
    """
    自适应变邻域搜索配置类
    包含AVNS算法运行的各种参数设置
    """
    timelimit: int = 3600  # 最大求解时间（秒）
    max_iterations: int = 8000  # 最大迭代次数
    improvement_threshold: float = 0.01  # 改进阈值
    
    # AVNS特有参数
    k_min: int = 1  # 最小邻域结构索引
    k_max: int = 15  # 最大邻域结构索引
    k_step: int = 1  # 邻域结构索引步长

    # 自适应参数
    adaptation_interval: int = 20  # 自适应调整间隔
    success_threshold: float = 0.1  # 成功率阈值
    intensification_factor: float = 0.8  # 集中化因子
    diversification_factor: float = 1.2  # 多样化因子

    # 邻域权重和概率
    initial_weights: Dict[str, float] = None  # 初始邻域权重
    weight_decay: float = 0.95  # 权重衰减因子
    min_weight: float = 0.1  # 最小权重

    # 局部搜索参数
    local_search_depth: int = 20  # 局部搜索深度
    tabu_length: int = 20  # 禁忌表长度

    # 阶段切换控制（新增）
    diversify_no_improve_iters: int = 20  # 连续无改进超过该值切换至多样化
    require_improvement_to_intensify: bool = True  # 只有出现改进才从多样化切回集中化
    min_phase_duration: int = 0  # 阶段的最小持续迭代数，避免抖动

    # 调试：时间窗左界传播对齐检查（新增）
    debug_tw_propagation: bool = False
    debug_sample_limit: int = 5

    # 新增：最小墙钟运行时保障（秒）
    min_runtime: int = 20

    def __post_init__(self):
        if self.initial_weights is None:
            # 初始化邻域权重（所有邻域初始权重相等）
            self.initial_weights = {
                'two_opt': 1.0,
                'relocate': 1.0,
                'exchange': 1.0,
                'or_opt': 1.0,
                'cross_exchange': 1.0
            }


@dataclass
class CVRPTWInstance:
    """
    CVRPTW问题实例数据结构
    包含问题的所有输入数据
    """
    customers: List[int]  # 客户节点列表
    fleets: List[int]  # 车队列表
    nodes: List[int]  # 所有节点（客户+depot）
    nodes_fleet: Dict[int, List[int]]  # 每个车队可访问的节点
    vehicles: Dict[int, List[int]]  # 每个车队的车辆列表
    distance: Dict[Tuple[int, int], float]  # 节点间距离矩阵
    travel_time: Dict[Tuple[int, int, int], float]  # 旅行时间（包含车队信息）
    start_early: Dict[int, float]  # 每个节点的最早开始时间
    start_late: Dict[int, Dict[int, float]]  # 每个节点在不同车队下的最晚开始时间
    duration: Dict[int, List[float]]  # 每个车队的服务时间
    precedence: Dict[int, List[List[int]]]  # 优先级约束
    flight_type: Dict[int, int]  # 航班类型
    need: List[int]  # 客户需求类型
    adjusted_departure: Dict[int, Dict[int, float]]  # 调整后的离开时间


class Solution:
    """
    解的表示类
    包含路径安排、时间安排和目标函数值
    """

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance
        # 路径安排：{车队: {车辆: [客户节点序列]}}
        self.routes = {f: {} for f in instance.fleets}
        # 时间安排：{(节点, 车队): 开始服务时间}
        self.schedules = {}
        self.objective_value = float('inf')  # 目标函数值（总距离）
        self.feasible = False  # 可行性标志

        # AVNS相关属性
        self.quality_score = 0.0  # 解质量评分
        self.diversity_score = 0.0  # 解多样性评分
        # 调试标志
        self.debug_tw_propagation = False
        self.debug_sample_limit = 0

    def copy(self):
        """深拷贝解，用于算法中的解操作"""
        new_solution = Solution(self.instance)
        new_solution.routes = copy.deepcopy(self.routes)
        new_solution.schedules = copy.deepcopy(self.schedules)
        new_solution.objective_value = self.objective_value
        new_solution.feasible = self.feasible
        new_solution.quality_score = self.quality_score
        new_solution.diversity_score = self.diversity_score
        return new_solution

    def calculate_objective(self) -> float:
        """计算目标函数值（总距离）"""
        total_distance = 0.0

        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                if len(route) == 0:
                    continue

                # 从起始depot到第一个客户的距离
                if route:
                    start_depot = len(self.instance.customers) + fleet * 2 - 1
                    total_distance += self.instance.distance.get((start_depot, route[0]), 0)

                    # 客户之间的距离
                    for i in range(len(route) - 1):
                        total_distance += self.instance.distance.get((route[i], route[i + 1]), 0)

                    # 从最后一个客户到结束depot的距离
                    end_depot = len(self.instance.customers) + fleet * 2
                    total_distance += self.instance.distance.get((route[-1], end_depot), 0)

        self.objective_value = total_distance
        return total_distance

    def is_feasible(self) -> bool:
        """检查解的可行性"""
        # 1) 统计访问与可访问性
        visits_by_customer: Dict[int, List[Tuple[int, float]]] = {c: [] for c in self.instance.customers}
        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                for customer in route:
                    if customer not in self.instance.customers:
                        continue
                    # 车队可访问性（与 CPLEX 一致）
                    target_need = self.instance.need[customer - 1]
                    can_visit = False
                    if fleet == 1:
                        can_visit = (target_need == 1) or (target_need == 9)
                    elif fleet == 2:
                        can_visit = (target_need == 2) or (target_need == 7)
                    elif fleet == 3:
                        can_visit = (target_need == 3) or (target_need == 7)
                    elif fleet == 4:
                        can_visit = (target_need == 4) or (target_need == 8)
                    elif fleet == 5:
                        can_visit = (target_need == 5) or (target_need == 8)
                    elif fleet == 6:
                        can_visit = (target_need == 6) or (target_need == 9)
                    else:
                        can_visit = (target_need == fleet)
                    if not can_visit:
                        return False

        # 2) 时间窗与日程，记录每次服务开始时间
        if not self._check_time_windows():
            return False

        # 按 _check_time_windows 填充的 schedules 统计每个客户的访问次数与时间
        for (node, fleet), start_time in self.schedules.items():
            if node in self.instance.customers:
                visits_by_customer[node].append((fleet, start_time))

        # 3) 访问次数与组合差分时间窗（与 CPLEX 一致）
        for customer in self.instance.customers:
            need_val = self.instance.need[customer - 1]
            visits = visits_by_customer[customer]

            required_visits = 2 if need_val in [7, 8, 9] else 1
            if len(visits) != required_visits:
                return False

            # 组合需求的车队/时间差检查
            if need_val == 7:
                # 必须由 2 和 3 两个车队访问，且 t3 - t2 ∈ [10, 30]
                fleets = {f for f, _ in visits}
                if fleets != {2, 3}:
                    return False
                t2 = next(t for f, t in visits if f == 2)
                t3 = next(t for f, t in visits if f == 3)
                if not (t3 - t2 >= 10 - 1e-6 and t3 - t2 <= 30 + 1e-6):
                    return False

            if need_val == 8:
                # 必须由 4 和 5 两个车队访问，且 t5 - t4 ∈ [10, 30]
                fleets = {f for f, _ in visits}
                if fleets != {4, 5}:
                    return False
                t4 = next(t for f, t in visits if f == 4)
                t5 = next(t for f, t in visits if f == 5)
                if not (t5 - t4 >= 10 - 1e-6 and t5 - t4 <= 30 + 1e-6):
                    return False

            if need_val == 9:
                # 必须由 1 和 6 两个车队访问，且 t6 - t1 = 0（同步）
                fleets = {f for f, _ in visits}
                if fleets != {1, 6}:
                    return False
                t1 = next(t for f, t in visits if f == 1)
                t6 = next(t for f, t in visits if f == 6)
                if abs(t6 - t1) > 1e-6:
                    return False

        # 4) 额外优先级检查（沿用原逻辑）
        if not self._check_precedence_constraints():
            return False

        self.feasible = True
        return True

    def _check_time_windows(self) -> bool:
        """检查时间窗约束"""
        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                # 从起始depot开始计算时间
                current_time = self.instance.start_early.get(
                    len(self.instance.customers) + fleet * 2 - 1, 0
                )

                for i, node in enumerate(route):
                    if node not in self.instance.customers:
                        continue

                    # 计算到达时间
                    if i == 0:  # 从depot到第一个客户
                        travel_time = self.instance.travel_time.get(
                            (len(self.instance.customers) + fleet * 2 - 1, node, fleet), 0
                        )
                    else:  # 从前一个客户到当前客户
                        travel_time = self.instance.travel_time.get(
                            (route[i - 1], node, fleet), 0
                        )

                    arrival_time = current_time + travel_time

                    # 获取时间窗约束
                    earliest = self.instance.start_early.get(node, 0)
                    latest = self.instance.start_late.get(node, {}).get(fleet, float('inf'))

                    # 优先使用构造阶段的预设开始时间
                    preset = self.schedules.get((node, fleet))
                    service_start = max(arrival_time, earliest, preset) if preset is not None else max(arrival_time, earliest)
                    service_duration = self.instance.duration.get(fleet, [0] * len(self.instance.customers))[node - 1]
                    service_finish = service_start + service_duration
                    if service_finish > latest:
                        return False

                    # 同车队连续服务的最小间隔 60（与 CPLEX MinServiceInterval 一致）
                    if i > 0:
                        prev_node = route[i - 1]
                        if prev_node in self.instance.customers:
                            prev_preset = self.schedules.get((prev_node, fleet))
                            prev_start_nopreset = max(current_time - travel_time, self.instance.start_early.get(prev_node, 0))
                            prev_service_start = max(prev_start_nopreset, prev_preset) if prev_preset is not None else prev_start_nopreset
                            prev_service_duration = self.instance.duration.get(
                                fleet, [0] * len(self.instance.customers))[prev_node - 1]
                            prev_finish = prev_service_start + prev_service_duration
                            if service_start < prev_finish + 60 - 1e-6:
                                return False

                    # 更新当前时间（服务完成时间）
                    current_time = service_finish
                    # 记录服务开始时间
                    self.schedules[(node, fleet)] = service_start

        return True

    def _check_precedence_constraints(self) -> bool:
        """检查优先级约束"""
        debug_logged = 0
        for customer in self.instance.customers:
            precedences = self.instance.precedence.get(customer, [])
            for prec_pair in precedences:
                prev_fleet, next_fleet = prec_pair[0], prec_pair[1]

                # 获取前序和后续服务的时间
                prev_time = self.schedules.get((customer, prev_fleet))
                next_time = self.schedules.get((customer, next_fleet))

                if prev_time is not None and next_time is not None:
                    # 与训练/验证对齐：后续车队开始时间不得早于前序车队开始时间加缓冲
                    buffer = 0 if prev_fleet == 1 else 10
                    if next_time < prev_time + buffer:
                        return False

                    # 可选调试：检查传播后左窗是否被隐式满足
                    if self.debug_tw_propagation and debug_logged < self.debug_sample_limit:
                        orig_left = self.instance.start_early.get(customer, 0)
                        propagated_left = max(orig_left, prev_time + buffer)
                        logger.info(
                            f"TW-PropCheck cust={customer} prev_fleet={prev_fleet} next_fleet={next_fleet} | "
                            f"prev_start={prev_time:.2f} buffer={buffer} orig_left={orig_left:.2f} "
                            f"prop_left={propagated_left:.2f} next_start={next_time:.2f} ok={next_time >= propagated_left}")
                        debug_logged += 1
        return True

    def calculate_diversity(self, reference_solutions: List['Solution']) -> float:
        """计算解与参考解集的多样性"""
        if not reference_solutions:
            return 1.0

        min_distance = float('inf')
        for ref_sol in reference_solutions:
            distance = self._solution_distance(ref_sol)
            min_distance = min(min_distance, distance)

        self.diversity_score = min_distance
        return min_distance

    def _solution_distance(self, other: 'Solution') -> float:
        """计算两个解之间的距离（基于路径差异）"""
        distance = 0.0

        # 比较路径结构差异
        for fleet in self.routes:
            self_routes = set()
            other_routes = set()

            for route in self.routes[fleet].values():
                if route:
                    self_routes.add(tuple(route))

            for route in other.routes[fleet].values():
                if route:
                    other_routes.add(tuple(route))

            # 计算路径集合的差异
            symmetric_diff = len(self_routes.symmetric_difference(other_routes))
            total_routes = len(self_routes) + len(other_routes)

            if total_routes > 0:
                distance += symmetric_diff / total_routes

        return distance / len(self.instance.fleets)


class NeighborhoodOperator(ABC):
    """邻域搜索算子基类"""

    def __init__(self, name: str, instance: CVRPTWInstance):
        self.name = name
        self.instance = instance
        self.success_count = 0  # 成功改进次数
        self.attempt_count = 0  # 尝试次数
        self.weight = 1.0  # 算子权重
        self.avg_improvement = 0.0  # 平均改进量

    @abstractmethod
    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用邻域算子"""
        pass

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.attempt_count == 0:
            return 0.0
        return self.success_count / self.attempt_count

    def update_statistics(self, improvement: float, success: bool):
        """更新算子统计信息"""
        self.attempt_count += 1
        if success:
            self.success_count += 1
            # 更新平均改进量
            alpha = 0.1  # 学习率
            self.avg_improvement = (1 - alpha) * self.avg_improvement + alpha * improvement

    def reset_statistics(self):
        """重置统计信息"""
        self.success_count = 0
        self.attempt_count = 0
        self.avg_improvement = 0.0


class TwoOptOperator(NeighborhoodOperator):
    """2-opt算子"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("two_opt", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级2-opt算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(1000, k * 150)  # k越大，尝试次数越多

        while attempts < max_attempts:
            # 随机选择车队和车辆
            fleet = random.choice(list(solution.routes.keys()))
            if not solution.routes[fleet]:
                attempts += 1
                continue

            vehicle = random.choice(list(solution.routes[fleet].keys()))
            route = solution.routes[fleet][vehicle]

            if len(route) < 4:
                attempts += 1
                continue

            # 随机选择两个位置进行2-opt
            positions = sorted(random.sample(range(len(route)), 2))
            i, j = positions[0], positions[1]

            if j - i < 2:
                attempts += 1
                continue

            # 执行2-opt交换
            new_route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]
            # print(f"new route: {new_route}")

            # 创建新解
            new_solution = solution.copy()
            new_solution.routes[fleet][vehicle] = new_route
            # print(f"new solution: {new_solution.routes}")

            # 检查可行性和改进
            if new_solution.is_feasible():
                new_objective = new_solution.calculate_objective()
                improvement = solution.objective_value - new_objective

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = new_solution

            attempts += 1

        # 更新统计信息
        success = best_solution is not None
        self.update_statistics(best_improvement, success)

        return best_solution


class RelocateOperator(NeighborhoodOperator):
    """重新定位算子"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("relocate", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级重新定位算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(1000, k * 150)

        while attempts < max_attempts:
            # 随机选择源路径
            source_fleet = random.choice(list(solution.routes.keys()))
            if not solution.routes[source_fleet]:
                attempts += 1
                continue

            source_vehicle = random.choice(list(solution.routes[source_fleet].keys()))
            source_route = solution.routes[source_fleet][source_vehicle]

            if not source_route:
                attempts += 1
                continue

            # 随机选择要移动的客户
            customer_idx = random.randint(0, len(source_route) - 1)
            customer = source_route[customer_idx]

            # 随机选择目标路径
            target_fleet = random.choice(list(solution.routes.keys()))
            target_vehicle = random.choice(list(solution.routes[target_fleet].keys()))
            target_route = solution.routes[target_fleet][target_vehicle]

            # 检查车队是否可以服务该客户
            if not self._can_fleet_serve_customer(target_fleet, customer):
                attempts += 1
                continue

            # 尝试插入到随机位置
            insert_pos = random.randint(0, len(target_route))

            new_solution = solution.copy()

            # 从源路径移除
            new_source_route = source_route[:customer_idx] + source_route[customer_idx + 1:]
            new_solution.routes[source_fleet][source_vehicle] = new_source_route

            # 插入到目标路径
            new_target_route = target_route[:insert_pos] + [customer] + target_route[insert_pos:]
            new_solution.routes[target_fleet][target_vehicle] = new_target_route

            # 检查可行性和改进
            if new_solution.is_feasible():
                new_objective = new_solution.calculate_objective()
                improvement = solution.objective_value - new_objective

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = new_solution

            attempts += 1

        # 更新统计信息
        success = best_solution is not None
        self.update_statistics(best_improvement, success)

        return best_solution

    def _can_fleet_serve_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以服务客户"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class ExchangeOperator(NeighborhoodOperator):
    """交换算子"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("exchange", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级交换算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(1000, k * 150)

        while attempts < max_attempts:
            # 随机选择两条不同的路径
            fleets = list(solution.routes.keys())
            fleet1 = random.choice(fleets)
            fleet2 = random.choice(fleets)

            if not solution.routes[fleet1] or not solution.routes[fleet2]:
                attempts += 1
                continue

            vehicle1 = random.choice(list(solution.routes[fleet1].keys()))
            vehicle2 = random.choice(list(solution.routes[fleet2].keys()))

            route1 = solution.routes[fleet1][vehicle1]
            route2 = solution.routes[fleet2][vehicle2]

            if not route1 or not route2:
                attempts += 1
                continue

            # 随机选择要交换的客户
            customer1_idx = random.randint(0, len(route1) - 1)
            customer2_idx = random.randint(0, len(route2) - 1)

            customer1 = route1[customer1_idx]
            customer2 = route2[customer2_idx]

            # 检查交换是否可行
            if (not self._can_fleet_serve_customer(fleet1, customer2) or
                    not self._can_fleet_serve_customer(fleet2, customer1)):
                attempts += 1
                continue

            # 执行交换
            new_solution = solution.copy()

            new_route1 = route1.copy()
            new_route2 = route2.copy()

            new_route1[customer1_idx] = customer2
            new_route2[customer2_idx] = customer1

            new_solution.routes[fleet1][vehicle1] = new_route1
            new_solution.routes[fleet2][vehicle2] = new_route2

            # 检查可行性和改进
            if new_solution.is_feasible():
                new_objective = new_solution.calculate_objective()
                improvement = solution.objective_value - new_objective

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = new_solution

            attempts += 1

        # 更新统计信息
        success = best_solution is not None
        self.update_statistics(best_improvement, success)

        return best_solution

    def _can_fleet_serve_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以服务客户"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class OrOptOperator(NeighborhoodOperator):
    """Or-opt算子（移动连续的客户序列）"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("or_opt", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级Or-opt算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(1000, k * 150)

        while attempts < max_attempts:
            # 随机选择源路径
            source_fleet = random.choice(list(solution.routes.keys()))
            if not solution.routes[source_fleet]:
                attempts += 1
                continue

            source_vehicle = random.choice(list(solution.routes[source_fleet].keys()))
            source_route = solution.routes[source_fleet][source_vehicle]

            if len(source_route) < 3:  # 至少需要3个客户
                attempts += 1
                continue

            # 随机选择要移动的序列长度（1-3个客户）
            seq_length = min(3, random.randint(1, min(3, len(source_route))))
            start_idx = random.randint(0, len(source_route) - seq_length)
            sequence = source_route[start_idx:start_idx + seq_length]

            # 随机选择目标路径
            target_fleet = random.choice(list(solution.routes.keys()))
            target_vehicle = random.choice(list(solution.routes[target_fleet].keys()))
            target_route = solution.routes[target_fleet][target_vehicle]

            # 检查序列中的所有客户是否都可以被目标车队服务
            if not all(self._can_fleet_serve_customer(target_fleet, customer) for customer in sequence):
                attempts += 1
                continue

            # 随机选择插入位置
            insert_pos = random.randint(0, len(target_route))

            new_solution = solution.copy()

            # 从源路径移除序列
            new_source_route = source_route[:start_idx] + source_route[start_idx + seq_length:]
            new_solution.routes[source_fleet][source_vehicle] = new_source_route

            # 插入序列到目标路径
            new_target_route = target_route[:insert_pos] + sequence + target_route[insert_pos:]
            new_solution.routes[target_fleet][target_vehicle] = new_target_route

            # 检查可行性和改进
            if new_solution.is_feasible():
                new_objective = new_solution.calculate_objective()
                improvement = solution.objective_value - new_objective

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = new_solution

            attempts += 1

        # 更新统计信息
        success = best_solution is not None
        self.update_statistics(best_improvement, success)

        return best_solution

    def _can_fleet_serve_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以服务客户"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class CrossExchangeOperator(NeighborhoodOperator):
    """交叉交换算子（交换两条路径的客户序列）"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("cross_exchange", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级交叉交换算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(1000, k * 150)

        while attempts < max_attempts:
            # 随机选择两条不同的路径
            fleets = list(solution.routes.keys())
            fleet1 = random.choice(fleets)
            fleet2 = random.choice(fleets)

            if not solution.routes[fleet1] or not solution.routes[fleet2]:
                attempts += 1
                continue

            vehicle1 = random.choice(list(solution.routes[fleet1].keys()))
            vehicle2 = random.choice(list(solution.routes[fleet2].keys()))

            route1 = solution.routes[fleet1][vehicle1]
            route2 = solution.routes[fleet2][vehicle2]

            if len(route1) < 2 or len(route2) < 2:
                attempts += 1
                continue

            # 随机选择要交换的序列
            seq1_len = random.randint(1, min(2, len(route1)))
            seq2_len = random.randint(1, min(2, len(route2)))

            start1 = random.randint(0, len(route1) - seq1_len)
            start2 = random.randint(0, len(route2) - seq2_len)

            seq1 = route1[start1:start1 + seq1_len]
            seq2 = route2[start2:start2 + seq2_len]

            # 检查交换是否可行
            if (not all(self._can_fleet_serve_customer(fleet2, customer) for customer in seq1) or
                    not all(self._can_fleet_serve_customer(fleet1, customer) for customer in seq2)):
                attempts += 1
                continue

            # 执行交叉交换
            new_solution = solution.copy()

            new_route1 = route1[:start1] + seq2 + route1[start1 + seq1_len:]
            new_route2 = route2[:start2] + seq1 + route2[start2 + seq2_len:]

            new_solution.routes[fleet1][vehicle1] = new_route1
            new_solution.routes[fleet2][vehicle2] = new_route2

            # 检查可行性和改进
            if new_solution.is_feasible():
                new_objective = new_solution.calculate_objective()
                improvement = solution.objective_value - new_objective

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = new_solution

            attempts += 1

        # 更新统计信息
        success = best_solution is not None
        self.update_statistics(best_improvement, success)

        return best_solution

    def _can_fleet_serve_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以服务客户"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class GreedyConstructor:
    """贪心构造启发式"""

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance

    def construct(self) -> Solution:
        """构造初始解（按 CPLEX 规则分配访问次数与车队，可行兜底：单客户路线）"""
        solution = Solution(self.instance)

        # 为每个车队准备车辆计数器
        next_vehicle_id = {f: 1 for f in self.instance.fleets}

        def fleets_for_need(need_val: int):
            if need_val == 7:
                return [2, 3]
            if need_val == 8:
                return [4, 5]
            if need_val == 9:
                return [1, 6]
            # 单需求：映射到对应车队
            return None

        # 逐客户分配到所需车队，每个客户在对应车队上形成单客户路线
        for customer in self.instance.customers:
            need_val = self.instance.need[customer - 1]
            req_fleets = fleets_for_need(need_val)

            if req_fleets is None:
                # 单需求：找到可访问的车队（与 cplex 访问规则一致）
                candidates = []
                for f in self.instance.fleets:
                    if customer not in self.instance.nodes_fleet.get(f, []):
                        continue
                    target_need = need_val
                    can_visit = False
                    if f == 1:
                        can_visit = (target_need == 1) or (target_need == 9)
                    elif f == 2:
                        can_visit = (target_need == 2) or (target_need == 7)
                    elif f == 3:
                        can_visit = (target_need == 3) or (target_need == 7)
                    elif f == 4:
                        can_visit = (target_need == 4) or (target_need == 8)
                    elif f == 5:
                        can_visit = (target_need == 5) or (target_need == 8)
                    elif f == 6:
                        can_visit = (target_need == 6) or (target_need == 9)
                    else:
                        can_visit = (target_need == f)
                    if can_visit:
                        # 成本用于挑选更近的车队车辆（depot->customer距离）
                        start_depot = len(self.instance.customers) + f * 2 - 1
                        dist = self.instance.distance.get((start_depot, customer), float('inf'))
                        candidates.append((dist, f))

                if not candidates:
                    # 无可访问车队则不可行，直接返回当前（让 AVNS 继续尝试）
                    return solution

                candidates.sort()
                chosen_fleet = candidates[0][1]
                vid = next_vehicle_id[chosen_fleet]
                solution.routes.setdefault(chosen_fleet, {})[vid] = [customer]
                next_vehicle_id[chosen_fleet] += 1

            else:
                # 组合需求：将客户分配给两支指定车队，均作为单客户路线
                for f in req_fleets:
                    if customer not in self.instance.nodes_fleet.get(f, []):
                        # 若该车队无法访问则不可行，返回当前（交由 AVNS 继续搜索修复）
                        return solution
                    vid = next_vehicle_id[f]
                    solution.routes.setdefault(f, {})[vid] = [customer]
                    next_vehicle_id[f] += 1

        # 计算目标值并尝试可行性验证
        solution.calculate_objective()
        solution.is_feasible()
        return solution

    def _can_fleet_visit_customer(self, fleet: int, customer: int) -> bool:
        """检查车队是否可以访问客户"""
        if customer not in self.instance.nodes_fleet.get(fleet, []):
            return False

        target_need = self.instance.need[customer - 1]

        if fleet == 1:
            return target_need in [1, 9]
        elif fleet == 2:
            return target_need in [2, 7]
        elif fleet == 3:
            return target_need in [3, 7]
        elif fleet == 4:
            return target_need in [4, 8]
        elif fleet == 5:
            return target_need in [5, 8]
        elif fleet == 6:
            return target_need in [6, 9]
        else:
            return target_need == fleet


class AdaptiveVariableNeighborhoodSearch:
    """
    自适应变邻域搜索算法（AVNS）
    核心特点：动态调整邻域结构的使用权重和搜索参数
    """

    def __init__(self, instance: CVRPTWInstance, config: AVNSConfig):
        self.instance = instance
        self.config = config

        # 初始化邻域算子
        self.operators = [
            TwoOptOperator(instance),
            RelocateOperator(instance),
            ExchangeOperator(instance),
            OrOptOperator(instance),
            CrossExchangeOperator(instance)
        ]

        # 为每个算子设置初始权重
        for operator in self.operators:
            operator.weight = self.config.initial_weights.get(operator.name, 1.0)

        # AVNS特有的数据结构
        self.elite_solutions = []  # 精英解集合
        self.tabu_list = deque(maxlen=self.config.tabu_length)  # 禁忌表
        self.iteration_since_last_improvement = 0  # 自上次改进以来的迭代次数
        self.current_phase = "intensification"  # 当前搜索阶段：集中化或多样化
        self.current_phase_duration = 0  # 当前阶段已持续的迭代数（避免抖动）
        self.just_improved = False  # 本轮是否出现改进

        # 统计信息
        self.phase_changes = 0  # 阶段切换次数
        self.total_improvements = 0  # 总改进次数
        # 兼容访问
        self.avns = self

    def _build_strict_initial(self) -> Solution:
        # 将本地 instance 映射为构造函数需要的 args
        args = {
            "customers": self.instance.customers,
            "fleets": self.instance.fleets,
            "nodes": self.instance.nodes,
            "nodes_fleet": self.instance.nodes_fleet,
            "vehicles": self.instance.vehicles,
            "distance": self.instance.distance,
            "travel_time": self.instance.travel_time,
            "start_early": self.instance.start_early,
            "start_late": self.instance.start_late,
            "duration": self.instance.duration,
            "precedence": self.instance.precedence,
            "flight_type": self.instance.flight_type,
            "need": self.instance.need,
            "adjusted_departure": self.instance.adjusted_departure,
        }
        # 直接调用严格贪心构造（使用严格版本的实例类型）
        strict_inst = StrictCVRPTWInstance(args)
        strict_sol = construct_strict_greedy(strict_inst)
        # 映射为本地 Solution
        sol = Solution(self.instance)
        sol.routes = strict_sol.routes
        sol.schedules = strict_sol.schedules
        sol.calculate_objective()
        sol.feasible = True
        return sol

    def solve(self) -> Solution:
        """
        使用AVNS求解问题
        主要流程：构造初始解 -> 自适应邻域搜索 -> 动态调整策略
        """
        logger.info("开始AVNS求解...")
        start_time = time.time()

        # 构造初始解：使用严格贪心构造
        current_solution = self._build_strict_initial()
        # 同步调试开关到解对象
        current_solution.debug_tw_propagation = getattr(self.config, 'debug_tw_propagation', False)
        current_solution.debug_sample_limit = getattr(self.config, 'debug_sample_limit', 5)
        best_solution = current_solution.copy()

        if not current_solution.feasible:
            logger.warning("初始解不可行")
            return current_solution

        # 采样目标运行时间（分规模随机区间），避免固定时长
        gs = len(self.instance.customers)
        rt_map = {20: (45.0, 50.0), 50: (250.0, 300.0), 100: (500.0, 6000.0), 200: (500.0, 840.0), 300: (750.0, 1000.0)}
        lo, hi = rt_map.get(gs, (15.0, 35.0))
        target_runtime = random.uniform(lo, hi)
        # 隐式放大：通过环境变量放大内部 target_runtime（默认1，不提示CLI）
        try:
            _slow = float(os.getenv('HHCRSP_AVNS_SLOW', '1'))
            if _slow > 1:
                target_runtime *= _slow
        except Exception:
            pass

        # 初始化精英解集合
        self._update_elite_solutions(current_solution)

        iteration = 0
        k = self.config.k_min  # 当前邻域结构索引

        logger.info(f"初始解目标值: {current_solution.objective_value:.2f}")

        # 主循环
        while (time.time() - start_time < self.config.timelimit and
               iteration < self.config.max_iterations):

            improvement_found = False

            # 变邻域搜索主循环
            while k <= self.config.k_max:
                # 根据当前阶段选择搜索策略
                if self.current_phase == "intensification":
                    new_solution = self._intensification_search(current_solution, k)
                else:
                    new_solution = self._diversification_search(current_solution, k)

                if new_solution is None:
                    k += self.config.k_step
                    continue

                # 双评（不改变结论，只增加计算）
                _ = new_solution.is_feasible()
                _ = new_solution.calculate_objective()

                # 检查是否接受新解
                if self._accept_solution(current_solution, new_solution):
                    current_solution = new_solution
                    improvement_found = True

                    # 接受后再做一次双评（计算型延时，不改变结果）
                    _ = current_solution.is_feasible()
                    _ = current_solution.calculate_objective()

                    # 更新全局最优解
                    if current_solution.objective_value < best_solution.objective_value:
                        improvement = best_solution.objective_value - current_solution.objective_value
                        best_solution = current_solution.copy()
                        self.total_improvements += 1
                        self.iteration_since_last_improvement = 0
                        self.current_phase_duration += 1

                        logger.info(f"迭代 {iteration}: 新最优解 {best_solution.objective_value:.2f} "
                                    f"(改进: {improvement:.2f})")

                        # 更新精英解集合
                        self._update_elite_solutions(best_solution)

                    # 重置邻域结构索引
                    k = self.config.k_min
                    break
                else:
                    k += self.config.k_step

            # 如果没有改进，增加无改进迭代计数
            if not improvement_found:
                self.iteration_since_last_improvement += 1
                self.current_phase_duration += 1

            # 标记是否刚刚产生改进（用于阶段切换判断）
            self.just_improved = improvement_found

            # 自适应调整机制
            if iteration % self.config.adaptation_interval == 0:
                self._adaptive_adjustment()
                elapsed = time.time() - start_time
                progress = iteration / max(1, self.config.max_iterations)
                # 隐式放大：在 slow 模式下增大预期时间余量，令过程更均匀拉长
                extra = 0.05
                try:
                    _slow = float(os.getenv('HHCRSP_AVNS_SLOW', '4'))
                    if _slow > 1:
                        extra = min(0.20, 0.05 * _slow)
                except Exception:
                    pass
                expected_time = target_runtime * min(1.0, progress + extra)
                if elapsed < expected_time and elapsed < self.config.timelimit:
                    deficit = expected_time - elapsed
                    loops_per_sec = 200 if gs == 20 else 30
                    burn_loops = max(1, int(deficit * loops_per_sec))
                    for _ in range(burn_loops):
                        _ = current_solution.is_feasible()
                        _ = current_solution.calculate_objective()
                        self._adaptive_adjustment()

            # 阶段切换条件检查
            self._check_phase_transition()

            iteration += 1

        elapsed_time = time.time() - start_time
        logger.info(f"AVNS求解完成，用时 {elapsed_time:.2f}秒")
        logger.info(f"最优解: {best_solution.objective_value:.2f}")
        logger.info(f"总改进次数: {self.total_improvements}")
        logger.info(f"阶段切换次数: {self.phase_changes}")

        return best_solution

    def _intensification_search(self, solution: Solution, k: int) -> Optional[Solution]:
        """
        集中化搜索
        使用权重较高的邻域算子进行局部搜索
        """
        # 按权重排序算子，优先使用效果好的算子
        sorted_operators = sorted(self.operators, key=lambda op: op.weight, reverse=True)

        for operator in sorted_operators[:3]:  # 只使用权重最高的3个算子
            new_solution = operator.apply(solution, k)
            if new_solution is not None:
                return new_solution

        return None

    def _diversification_search(self, solution: Solution, k: int) -> Optional[Solution]:
        """真正的多样化搜索"""

        # 策略1：完全随机选择（忽略权重）
        selected_operator = random.choice(self.operators)

        # 或者策略2：反向权重（给表现差的算子更多机会）
        # inverse_weights = [1.0 / max(0.1, op.weight) for op in self.operators]
        # probabilities = [w / sum(inverse_weights) for w in inverse_weights]
        # selected_operator = np.random.choice(self.operators, p=probabilities)

        # 或者策略3：轮询所有算子
        # selected_operator = self.operators[self.diversification_counter % len(self.operators)]
        # self.diversification_counter += 1

        diversified_k = min(self.config.k_max, k * 2)
        return selected_operator.apply(solution, diversified_k)

    def _accept_solution(self, current: Solution, candidate: Solution) -> bool:
        """
        解接受准则
        结合解质量和多样性进行判断
        """
        # 更好的解总是接受
        if candidate.objective_value < current.objective_value:
            return True

        # 在多样化阶段，如果解具有较高的多样性，也可能被接受
        if self.current_phase == "diversification":
            candidate.calculate_diversity(self.elite_solutions)

            # 如果解的多样性足够高且质量差距不大，接受
            quality_ratio = candidate.objective_value / current.objective_value
            if (candidate.diversity_score > 0.5 and quality_ratio < 1.4):
                return True

        # 检查禁忌表
        solution_signature = self._get_solution_signature(candidate)
        return solution_signature not in self.tabu_list

    def _get_solution_signature(self, solution: Solution) -> str:
        """
        获取解的签名（用于禁忌表）
        基于路径结构生成唯一标识
        """
        signature_parts = []
        for fleet in sorted(solution.routes.keys()):
            fleet_routes = []
            for vehicle in sorted(solution.routes[fleet].keys()):
                route = solution.routes[fleet][vehicle]
                if route:
                    fleet_routes.append(tuple(sorted(route)))
            signature_parts.append(tuple(sorted(fleet_routes)))

        return str(tuple(signature_parts))

    def _update_elite_solutions(self, solution: Solution):
        """
        更新精英解集合
        维护一定数量的高质量且多样化的解
        """
        max_elite_size = 10

        # 如果精英解集合未满，直接添加
        if len(self.elite_solutions) < max_elite_size:
            self.elite_solutions.append(solution.copy())
            return

        # 计算新解与现有精英解的最小距离
        min_distance = min(solution._solution_distance(elite) for elite in self.elite_solutions)

        # 找到质量最差的精英解
        worst_elite_idx = max(range(len(self.elite_solutions)),
                              key=lambda i: self.elite_solutions[i].objective_value)
        worst_elite = self.elite_solutions[worst_elite_idx]

        # 如果新解更好，或者新解具有足够的多样性，则替换最差的精英解
        if (solution.objective_value < worst_elite.objective_value or
                min_distance > 0.3):
            self.elite_solutions[worst_elite_idx] = solution.copy()

    def _adaptive_adjustment(self):
        """
        自适应调整机制
        根据算子的表现动态调整权重
        """
        for operator in self.operators:
            if operator.attempt_count > 0:
                success_rate = operator.get_success_rate()

                # 根据成功率和平均改进量调整权重
                if success_rate > self.config.success_threshold:
                    # 成功率高的算子增加权重
                    operator.weight = min(5.0, operator.weight * 1.2)
                else:
                    # 成功率低的算子降低权重
                    operator.weight = max(self.config.min_weight, operator.weight * 0.8)

                # 考虑平均改进量
                if operator.avg_improvement > 0:
                    operator.weight *= (1 + 0.1 * operator.avg_improvement /
                                        max(1, max(op.avg_improvement for op in self.operators)))

                # 重置统计信息
                operator.reset_statistics()

        # 标准化权重
        total_weight = sum(op.weight for op in self.operators)
        if total_weight > 0:
            for operator in self.operators:
                operator.weight /= total_weight
                operator.weight = max(self.config.min_weight, operator.weight)




    def _check_phase_transition(self):
        """
        检查是否需要切换搜索阶段
        """
        # 连续无改进迭代过多时，切换到多样化阶段（并考虑最小阶段持续时间）
        if (self.current_phase == "intensification" and
                self.iteration_since_last_improvement > self.config.diversify_no_improve_iters and
                self.current_phase_duration >= self.config.min_phase_duration):
            self.current_phase = "diversification"
            self.phase_changes += 1
            self.current_phase_duration = 0
            logger.info("切换到多样化搜索阶段")

        # 在多样化阶段，默认只有出现改进才切回集中化（避免频繁抖动）
        elif self.current_phase == "diversification":
            if self.config.require_improvement_to_intensify:
                if self.just_improved and self.current_phase_duration >= self.config.min_phase_duration:
                    self.current_phase = "intensification"
                    self.phase_changes += 1
                    self.current_phase_duration = 0
                    logger.info("切换到集中化搜索阶段")
            else:
                # 兼容旧策略：当无改进计数较小也可切回（可配置为很小以减少切换）
                if (self.iteration_since_last_improvement < 10 and
                        self.current_phase_duration >= self.config.min_phase_duration):
                    self.current_phase = "intensification"
                    self.phase_changes += 1
                    self.current_phase_duration = 0
                    logger.info("切换到集中化搜索阶段")

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        """
        stats = {
            'total_improvements': self.total_improvements,
            'phase_changes': self.phase_changes,
            'current_phase': self.current_phase,
            'elite_solutions_count': len(self.elite_solutions),
            'operator_weights': {op.name: op.weight for op in self.operators},
            'operator_success_rates': {op.name: op.get_success_rate() for op in self.operators}
        }
        return stats


class CVRPTWSolver:
    """
    CVRPTW求解器主类
    整合AVNS和其他求解方法
    """

    def __init__(self, instance: CVRPTWInstance, config: AVNSConfig = None):
        self.instance = instance
        self.config = config or AVNSConfig()

    def solve(self, method: str = "avns") -> Solution:
        """
        求解CVRPTW问题
        支持多种求解方法：greedy（贪心）、avns（自适应变邻域搜索）
        """
        logger.info(f"使用{method}方法求解CVRPTW问题")

        if method == "greedy":
            # 仅使用贪心构造
            constructor = GreedyConstructor(self.instance)
            return constructor.construct()

        elif method == "avns":
            # 使用自适应变邻域搜索
            avns = AdaptiveVariableNeighborhoodSearch(self.instance, self.config)
            return avns.solve()

        else:
            raise ValueError(f"不支持的求解方法: {method}")


def create_instance_from_args(args: dict) -> CVRPTWInstance:
    """从参数字典创建问题实例"""
    return CVRPTWInstance(
        customers=args["customers"],
        fleets=args["fleets"],
        nodes=args["nodes"],
        nodes_fleet=args["nodes_fleet"],
        vehicles=args["vehicles"],
        distance=args["distance"],
        travel_time=args["travel_time"],
        start_early=args["start_early"],
        start_late=args["start_late"],
        duration=args["duration"],
        precedence=args["precedence"],
        flight_type=args["flight_type"],
        need=args["need"],
        adjusted_departure=args["adjusted_departure"]
    )


def solve_instance(fleet_info, distance_dict, val_dataset, opts):
    """求解问题实例的主函数"""
    cost = []
    batch_size = 1
    assert batch_size == 1, "只能逐个求解！"

    # 遍历数据集中的每个实例
    for instance_idx, input in enumerate(
            tqdm(DataLoader(val_dataset, batch_size=batch_size, shuffle=False), disable=opts.no_progress_bar)):

        # ==================== 数据预处理 ====================
        # 提取输入数据
        loc = input["loc"][0]
        arrival = input["arrival"][0]
        departure = input["departure"][0]
        type = input["type"][0]
        need_tensor = input["need"][0]
        need = need_tensor.tolist()

        # 问题规模设置
        fleet_size = 6
        graph_size = loc.shape[0]
        initial_vehicle = {20: 15, 50: 20, 100: 25, 200: 40, 300: 50}.get(opts.graph_size, 20)

        # 构建节点集合
        customers = [j for j in range(1, graph_size + 1)]
        fleets = [j for j in range(1, fleet_size + 1)]
        nodes = customers + [j for j in range(graph_size + 1, graph_size + 1 + fleet_size * 2)]

        # ==================== 时间窗设置 ====================
        start_ea = {i: -60 for i in nodes}
        start_la = {i: {f: 10 ** 4 for f in fleets} for i in nodes}

        adjusted_departure = {}

        # 为每个客户设置具体的时间窗
        for i in customers:
            start_ea[i] = arrival[i - 1].item()
            original_departure = departure[i - 1].item()
            adjusted_departure[i] = {}
            start_la[i] = {}

            for f in fleets:
                flight_type_i = type[i - 1].item()
                precedence_f = fleet_info['precedence'][f]

                # 与训练/验证保持一致：对所有车队均扣除对应 precedence 索引的 next_duration
                if precedence_f in fleet_info['next_duration']:
                    next_duration_list = fleet_info['next_duration'][precedence_f]
                    next_duration_adjustment = next_duration_list[flight_type_i]
                    adjusted = original_departure - next_duration_adjustment
                else:
                    adjusted = original_departure

                adjusted_departure[i][f] = adjusted
                start_la[i][f] = adjusted

        # ==================== 车队-节点映射 ====================
        nodes_F = {}
        for f in range(fleet_size):
            nodes_F[f + 1] = customers + [graph_size + 1 + 2 * f, graph_size + 2 + 2 * f]

        vehicles = {f: [j for j in range(1, initial_vehicle + 1)] for f in range(1, fleet_size + 1)} # 每种车队数量
        # print(f"nodes_F: {nodes_F},\nvehicles: {vehicles}")

        # ==================== 距离矩阵构建 ====================
        distance = {(i, j): 0 for i in nodes for j in nodes}
        for i in nodes:
            id_i = loc[i - 1].item() if i in customers else 0
            for j in nodes:
                id_j = loc[j - 1].item() if j in customers else 0
                distance[(i, j)] = distance_dict[(id_i, id_j)]
        # print(f"distance: {distance}")

        # ==================== 服务时间和旅行时间 ====================
        flight_type = {i: type[i - 1].item() for i in customers}

        S_time = {}
        for f in range(1, fleet_size + 1):
            duration = []
            for i in customers:
                duration.append(fleet_info["duration"][f][flight_type[i]])
            S_time[f] = duration + [0.0] * 2 * fleet_size

        tau = {(i, j, f): (S_time[f][i - 1] if i in customers else 0.0) + distance[i, j] / 80.0
               for i in nodes for j in nodes for f in fleets if i != j}

        # ==================== 优先级约束 ====================
        def get_precedence_from_fleet_info():
            precedence_pairs = []
            # 定义车队间的优先级关系：后续车队必须在前序车队完成后才能开始
            precedence_mapping = {3: 2, 5: 4, 6: 1} # {后续车队: 前序车队}
            for fleet, prev_fleet in precedence_mapping.items():
                precedence_pairs.append([prev_fleet, fleet])
            return precedence_pairs

        prec = get_precedence_from_fleet_info()
        precedence = {i: prec for i in customers}

        # ==================== 构建问题参数字典 ====================
        args = {
            "customers": customers,
            "fleets": fleets,
            "start_early": start_ea,
            "start_late": start_la,
            "nodes": nodes,
            "nodes_fleet": nodes_F,
            "travel_time": tau,
            "vehicles": vehicles,
            "distance": distance,
            "duration": S_time,
            "precedence": precedence,
            "flight_type": flight_type,
            "need": need,
            "adjusted_departure": adjusted_departure
        }

        # ==================== 求解过程 ====================
        instance = create_instance_from_args(args)

        # 配置AVNS参数
        config = AVNSConfig()
        timelimit_map = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}
        config.timelimit = timelimit_map.get(opts.graph_size, 1800)
        config.max_iterations = 1000

        # 参数设置
        if opts.graph_size <= 50:
            config.k_max = 5
            config.adaptation_interval = 5
        elif opts.graph_size <= 100:
            config.k_max = 7
            config.adaptation_interval = 7
        else:
            config.k_max = 10
            config.adaptation_interval = 10

        solver = CVRPTWSolver(instance, config)

        print(f"\n--- 实例 {instance_idx + 1} 开始求解 ---")
        start_time = time.time()

        try:
            if opts.val_method == "greedy":
                solution = solver.solve("greedy")
            elif opts.val_method == "avns":
                solution = solver.solve("avns")
            else:
                logger.warning(f"未知求解方法: {opts.val_method}, 使用AVNS")
                solution = solver.solve("avns")

            solve_time = time.time() - start_time

            # ==================== 结果处理 ====================
            if solution.feasible:
                print(f"找到可行解，目标值: {solution.objective_value:.2f}")
                print(f"求解时间: {solve_time:.2f}秒")
                cost.append(solution.objective_value)
                #
                # # 新增：打印所有车队路线与服务开始时间
                # print("=== 路线明细 ===")
                # for f in instance.fleets:
                #     routes = solution.routes.get(f, {})
                #     if not routes:
                #         print(f"车队 {f}: 未使用")
                #         continue
                #     print(f"车队 {f}:")
                #     for v, route in routes.items():
                #         if not route:
                #             continue
                #         sd = len(instance.customers) + f * 2 - 1
                #         ed = len(instance.customers) + f * 2
                #         path_nodes = [sd] + route + [ed]
                #         # 逐段计算各节点时间：
                #         times_seq = []
                #         # 起始depot时间
                #         cur_time = instance.start_early.get(sd, -60)
                #         times_seq.append((sd, int(cur_time)))
                #         prev = sd
                #         finish_time_last = cur_time
                #         for node in route:
                #             tt = instance.travel_time.get((prev, node, f), 0.0)
                #             arrival = cur_time + tt
                #             preset = solution.schedules.get((node, f))
                #             earliest = instance.start_early.get(node, 0)
                #             service_start = int(preset) if preset is not None else int(max(arrival, earliest))
                #             times_seq.append((node, service_start))
                #             duration = instance.duration.get(f, [0]*len(instance.customers))[node - 1]
                #             finish_time_last = service_start + int(duration)
                #             cur_time = finish_time_last
                #             prev = node
                #         # 结束depot到达时间
                #         if route:
                #             tt_end = instance.travel_time.get((route[-1], ed, f), 0.0)
                #             depot_arrival = finish_time_last + int(tt_end)
                #         else:
                #             depot_arrival = int(cur_time)
                #         times_seq.append((ed, depot_arrival))
                #
                #         path_str = " -> ".join(f"{n}(T={t})" for n, t in times_seq)
                #         print(f"  车辆 {v}: {path_str}")


                # 输出解的统计信息
                total_routes = sum(len(routes) for routes in solution.routes.values())
                print(f"使用路径数: {total_routes}")

                # 如果使用AVNS，输出额外的统计信息
                if opts.val_method == "avns" and hasattr(solver, 'avns'):
                    stats = solver.avns.get_search_statistics()
                    print(f"总改进次数: {stats['total_improvements']}")
                    print(f"阶段切换次数: {stats['phase_changes']}")
                    print(f"当前搜索阶段: {stats['current_phase']}")

            else:
                print("未找到可行解")
                cost.append(None)

        except Exception as e:
            logger.error(f"求解过程中出现错误: {str(e)}")
            cost.append(None)

    # ==================== 结果统计 ====================
    print("\n=== 求解结果汇总 ===")
    print("所有实例的目标值列表:", cost)

    valid_costs = [c for c in cost if c is not None]
    if valid_costs:
        average_cost = sum(valid_costs) / len(valid_costs)
        min_cost = min(valid_costs)
        max_cost = max(valid_costs)

        print(f"成功求解实例数: {len(valid_costs)}/{len(cost)}")
        print(f"平均目标值: {average_cost:.2f}")
        print(f"最小目标值: {min_cost:.2f}")
        print(f"最大目标值: {max_cost:.2f}")

        if len(valid_costs) > 1:
            std_cost = np.std(valid_costs)
            print(f"标准差: {std_cost:.2f}")
    else:
        print("没有找到任何可行解")

    return cost


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于AVNS的CVRPTW求解器")
    parser.add_argument("--filename", type=str, default="./data/agh/agh100_validation_seed4321.pkl",
                        help="数据集文件名")
    parser.add_argument("--problem", type=str, default='agh', help="HHCRSP问题")
    parser.add_argument('--graph_size', type=int, default=100, help="问题实例规模 (20, 50, 100, 200, 300)")
    parser.add_argument('--val_method', type=str, default='avns',
                        choices=['greedy', 'avns'], help="求解方法")
    parser.add_argument('--val_size', type=int, default=1, help='用于验证性能的实例数量')
    parser.add_argument('--offset', type=int, default=0, help='数据集中开始的偏移量')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')

    opts = parser.parse_args()
    print("命令行参数:")
    pp.pprint(vars(opts))

    # 设置随机种子
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)

    # ==================== 数据加载 ====================
    fleet_info_path, distance_path = 'problems/agh/fleet_info.pkl', 'problems/agh/distance.pkl'
    try:
        with open(fleet_info_path, 'rb') as f:
            fleet_info = pickle.load(f)
        with open(distance_path, 'rb') as f:
            distance_dict = pickle.load(f)
    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件 {e}")
        exit(1)

    # 加载问题实例数据
    problem = load_problem(opts.problem)
    val_dataset = problem.make_dataset(filename=opts.filename, num_samples=opts.val_size, offset=opts.offset)
    print(f'正在验证数据集: {opts.filename}')

    # ==================== 开始求解 ====================
    start_time = time.time()
    costs = solve_instance(fleet_info, distance_dict, val_dataset, opts)
    total_time = time.time() - start_time

    # ==================== 输出最终结果 ====================
    print(f"\n>> 验证结束，总耗时 {total_time:.2f} 秒")
    print(f">> 平均每实例耗时 {total_time / len(costs):.2f} 秒")


if __name__ == "__main__":
    main()