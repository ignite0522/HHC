import os, sys, copy, time, json
import argparse, math, random
import torch
import pickle
import numpy as np
import pprint as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import move_to, load_problem
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq

# 配置日志系统，用于调试和监控求解过程
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VNSConfig:
    """
    变邻域搜索配置类
    包含VNS算法运行的各种参数设置
    """
    timelimit: int = 3600  # 最大求解时间（秒）
    max_iterations: int = 1000  # 最大迭代次数
    improvement_threshold: float = 0.01  # 改进阈值

    # VNS特有参数
    k_min: int = 1  # 最小邻域结构索引
    k_max: int = 5  # 最大邻域结构索引
    k_step: int = 1  # 邻域结构索引步长

    # 局部搜索参数
    local_search_depth: int = 10  # 局部搜索深度
    tabu_length: int = 20  # 禁忌表长度

    # VNS重启机制
    max_no_improvement: int = 100  # 最大无改进迭代次数


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

    def copy(self):
        """深拷贝解，用于算法中的解操作"""
        new_solution = Solution(self.instance)
        new_solution.routes = copy.deepcopy(self.routes)
        new_solution.schedules = copy.deepcopy(self.schedules)
        new_solution.objective_value = self.objective_value
        new_solution.feasible = self.feasible
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
        # 检查所有客户是否都被访问
        visited_customers = set()
        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                for customer in route:
                    if customer in self.instance.customers:
                        visited_customers.add(customer)

        if len(visited_customers) != len(self.instance.customers):
            return False

        # 检查时间窗约束
        if not self._check_time_windows():
            return False

        # 检查优先级约束
        if not self._check_precedence_constraints():
            return False

        self.feasible = True
        return True

    def _check_time_windows(self) -> bool:
        """检查时间窗约束"""
        for fleet in self.routes:
            for vehicle, route in self.routes[fleet].items():
                if not route:  # 空路径跳过
                    continue

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

                    # 服务开始时间=max(到达时间, 最早开始时间)
                    service_start = max(arrival_time, earliest)

                    # 严格检查时间窗约束
                    if service_start > latest + 1e-6:  # 添加数值容差
                        logger.debug(
                            f"时间窗违反: 节点{node}, 车队{fleet}, 服务开始{service_start:.2f} > 最晚时间{latest:.2f}")
                        return False

                    # 更新当前时间（服务完成时间）
                    service_duration = self.instance.duration.get(fleet, [0] * len(self.instance.customers))[node - 1]
                    current_time = service_start + service_duration
                    # 记录服务开始时间
                    self.schedules[(node, fleet)] = service_start

        return True

    def _check_precedence_constraints(self) -> bool:
        """检查优先级约束"""
        for customer in self.instance.customers:
            precedences = self.instance.precedence.get(customer, [])
            for prec_pair in precedences:
                prev_fleet, next_fleet = prec_pair[0], prec_pair[1]

                # 获取前序和后续服务的时间
                prev_time = self.schedules.get((customer, prev_fleet))
                next_time = self.schedules.get((customer, next_fleet))

                if prev_time is not None and next_time is not None:
                    # 前序服务完成时间
                    prev_duration = self.instance.duration.get(prev_fleet, [0] * len(self.instance.customers))[
                        customer - 1]
                    if prev_time + prev_duration > next_time:  # 违反优先级约束
                        return False
        return True

    def get_solution_hash(self) -> str:
        """获取解的哈希值，用于避免重复搜索"""
        route_signature = []
        for fleet in sorted(self.routes.keys()):
            fleet_routes = []
            for vehicle in sorted(self.routes[fleet].keys()):
                route = self.routes[fleet][vehicle]
                if route:
                    fleet_routes.append(tuple(route))
            route_signature.append(tuple(sorted(fleet_routes)))
        return str(tuple(route_signature))


class NeighborhoodOperator(ABC):
    """邻域搜索算子基类"""

    def __init__(self, name: str, instance: CVRPTWInstance):
        self.name = name
        self.instance = instance

    @abstractmethod
    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用邻域算子"""
        pass

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


class TwoOptOperator(NeighborhoodOperator):
    """2-opt算子"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("two_opt", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级2-opt算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(50, k * 10)  # 简化尝试次数

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

            # 创建新解
            new_solution = solution.copy()
            new_solution.routes[fleet][vehicle] = new_route

            # 检查可行性和改进
            if new_solution.is_feasible():
                new_objective = new_solution.calculate_objective()
                improvement = solution.objective_value - new_objective

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_solution = new_solution

            attempts += 1

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
        max_attempts = min(50, k * 8)

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

        return best_solution


class ExchangeOperator(NeighborhoodOperator):
    """交换算子"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("exchange", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级交换算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(50, k * 5)

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

        return best_solution


class OrOptOperator(NeighborhoodOperator):
    """Or-opt算子（移动连续的客户序列）"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("or_opt", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级Or-opt算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(50, k * 6)

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

        return best_solution


class CrossExchangeOperator(NeighborhoodOperator):
    """交叉交换算子（交换两条路径的客户序列）"""

    def __init__(self, instance: CVRPTWInstance):
        super().__init__("cross_exchange", instance)

    def apply(self, solution: Solution, k: int = 1) -> Optional[Solution]:
        """应用k-级交叉交换算子"""
        best_solution = None
        best_improvement = 0

        attempts = 0
        max_attempts = min(50, k * 4)

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

        return best_solution


class GreedyConstructor:
    """贪心构造启发式"""

    def __init__(self, instance: CVRPTWInstance):
        self.instance = instance

    def construct(self) -> Solution:
        """构造初始解"""
        solution = Solution(self.instance)
        unvisited = set(self.instance.customers)

        for fleet in self.instance.fleets:
            vehicle_id = 1

            while unvisited:
                route = []  # 当前路径
                # 从起始depot开始
                current_time = self.instance.start_early.get(
                    len(self.instance.customers) + fleet * 2 - 1, 0
                )
                current_node = len(self.instance.customers) + fleet * 2 - 1  # 起始depot

                # 找到当前车队可以服务的客户
                fleet_customers = [c for c in unvisited if c in self.instance.nodes_fleet.get(fleet, [])]

                if not fleet_customers:
                    break

                # 贪心选择下一个客户
                while fleet_customers:
                    candidate_customers = []

                    for customer in fleet_customers:
                        # 检查车队是否可以访问该客户
                        if not self._can_fleet_visit_customer(fleet, customer):
                            continue

                        # 计算到达时间和成本
                        travel_time = self.instance.travel_time.get((current_node, customer, fleet), 0)
                        arrival_time = current_time + travel_time
                        earliest = self.instance.start_early.get(customer, 0)
                        latest = self.instance.start_late.get(customer, {}).get(fleet, float('inf'))

                        service_start = max(arrival_time, earliest)

                        # 检查时间窗可行性
                        if service_start <= latest:
                            cost = self.instance.distance.get((current_node, customer), float('inf'))
                            candidate_customers.append((cost, customer, service_start))

                    if not candidate_customers:
                        break

                    # 引入随机性：在前k个最佳候选中随机选择
                    candidate_customers.sort(key=lambda x: x[0])  # 按成本排序
                    k = max(1, min(len(candidate_customers), 5))  # 选择前5个候选
                    best_candidate = random.choice(candidate_customers[:k])  # 随机选择一个

                    best_cost, best_customer, best_arrival = best_candidate

                    # 添加客户到路径
                    route.append(best_customer)
                    fleet_customers.remove(best_customer)
                    unvisited.discard(best_customer)

                    # 更新时间和位置
                    service_duration = self.instance.duration.get(fleet, [0] * len(self.instance.customers))[
                        best_customer - 1]
                    current_time = best_arrival + service_duration
                    current_node = best_customer

                # 如果构造了非空路径，添加到解中
                if route:
                    solution.routes[fleet][vehicle_id] = route
                    vehicle_id += 1
                else:
                    break

        # 计算目标值并检查可行性
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


class VariableNeighborhoodSearch:
    """
    标准变邻域搜索算法（VNS）
    核心特点：系统性地使用不同的邻域结构进行搜索
    """

    def __init__(self, instance: CVRPTWInstance, config: VNSConfig):
        self.instance = instance
        self.config = config
        self.constructor = GreedyConstructor(instance)

        # 初始化邻域算子
        self.operators = [
            TwoOptOperator(instance),
            RelocateOperator(instance),
            ExchangeOperator(instance),
            OrOptOperator(instance),
            CrossExchangeOperator(instance)
        ]

        # VNS特有的数据结构
        self.tabu_list = deque(maxlen=self.config.tabu_length)  # 禁忌表
        self.no_improvement_count = 0  # 无改进迭代计数

        # 统计信息
        self.total_improvements = 0  # 总改进次数
        self.operator_usage = {op.name: 0 for op in self.operators}  # 算子使用统计

    def solve(self) -> Solution:
        """
        使用标准VNS求解问题
        主要流程：构造初始解 -> 变邻域搜索 -> 局部搜索
        """
        logger.info("开始VNS求解...")
        start_time = time.time()

        # 构造初始解
        current_solution = self.constructor.construct()
        best_solution = current_solution.copy()

        if not current_solution.feasible:
            logger.warning("初始解不可行")
            return current_solution

        iteration = 0
        logger.info(f"初始解目标值: {current_solution.objective_value:.2f}")

        # 主循环
        while (time.time() - start_time < self.config.timelimit and
               iteration < self.config.max_iterations and
               self.no_improvement_count < self.config.max_no_improvement):

            k = self.config.k_min  # 重置邻域结构索引

            # 变邻域搜索主循环
            while k <= self.config.k_max:
                # 抖动：从第k个邻域结构中随机选择一个解
                candidate_solution = self._shaking(current_solution, k)

                if candidate_solution is None:
                    k += self.config.k_step
                    continue

                # 局部搜索：改进候选解
                improved_solution = self._local_search(candidate_solution)

                # 检查是否接受新解
                if self._accept_solution(current_solution, improved_solution):
                    current_solution = improved_solution

                    # 更新全局最优解
                    if current_solution.objective_value < best_solution.objective_value:
                        improvement = best_solution.objective_value - current_solution.objective_value
                        best_solution = current_solution.copy()
                        self.total_improvements += 1
                        self.no_improvement_count = 0

                        logger.info(f"迭代 {iteration}: 新最优解 {best_solution.objective_value:.2f} "
                                    f"(改进: {improvement:.2f})")

                        # 重置邻域结构索引
                        k = self.config.k_min
                    else:
                        k += self.config.k_step
                else:
                    k += self.config.k_step

            # 如果没有改进，增加无改进计数
            if current_solution.objective_value >= best_solution.objective_value:
                self.no_improvement_count += 1

            iteration += 1

        elapsed_time = time.time() - start_time
        logger.info(f"VNS求解完成，用时 {elapsed_time:.2f}秒")
        logger.info(f"最优解: {best_solution.objective_value:.2f}")
        logger.info(f"总改进次数: {self.total_improvements}")

        return best_solution

    def _shaking(self, solution: Solution, k: int) -> Optional[Solution]:
        """
        抖动阶段：使用第k个邻域结构随机产生一个解
        目的是跳出局部最优，探索新的搜索区域
        """
        if k > len(self.operators):
            k = k % len(self.operators) + 1

        operator = self.operators[k - 1]
        self.operator_usage[operator.name] += 1

        # 使用较大的扰动强度进行抖动
        shaking_intensity = k  # k值越大，扰动越强
        candidate_solution = operator.apply(solution, shaking_intensity)

        return candidate_solution

    def _local_search(self, solution: Solution) -> Solution:
        """
        局部搜索阶段：使用贪心策略改进解
        在给定解的邻域内寻找更好的解
        """
        current_solution = solution.copy()
        improved = True
        depth = 0

        while improved and depth < self.config.local_search_depth:
            improved = False
            best_neighbor = None
            best_improvement = 0

            # 尝试所有邻域算子
            for operator in self.operators:
                neighbor = operator.apply(current_solution, 1)  # 使用k=1进行局部搜索

                if neighbor is not None:
                    improvement = current_solution.objective_value - neighbor.objective_value
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_neighbor = neighbor
                        improved = True

            if improved and best_neighbor is not None:
                current_solution = best_neighbor
                depth += 1

        return current_solution

    def _accept_solution(self, current: Solution, candidate: Solution) -> bool:
        """
        解接受准则：VNS通常只接受更好的解
        """
        # 更好的解总是接受
        if candidate.objective_value < current.objective_value:
            return True

        # 检查禁忌表（避免循环）
        candidate_hash = candidate.get_solution_hash()
        if candidate_hash in self.tabu_list:
            return False

        # VNS通常不接受更差的解，但可以加入一些随机性
        # 这里可以根据需要调整接受准则
        return False

    def _update_tabu_list(self, solution: Solution):
        """更新禁忌表"""
        solution_hash = solution.get_solution_hash()
        self.tabu_list.append(solution_hash)

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        """
        stats = {
            'total_improvements': self.total_improvements,
            'no_improvement_count': self.no_improvement_count,
            'operator_usage': self.operator_usage.copy(),
            'tabu_list_size': len(self.tabu_list)
        }
        return stats


class CVRPTWSolver:
    """
    CVRPTW求解器主类
    整合VNS和其他求解方法
    """

    def __init__(self, instance: CVRPTWInstance, config: VNSConfig = None):
        self.instance = instance
        self.config = config or VNSConfig()

    def solve(self, method: str = "vns") -> Solution:
        """
        求解CVRPTW问题
        支持多种求解方法：greedy（贪心）、vns（变邻域搜索）
        """
        logger.info(f"使用{method}方法求解CVRPTW问题")

        if method == "greedy":
            # 仅使用贪心构造
            constructor = GreedyConstructor(self.instance)
            return constructor.construct()

        elif method == "vns":
            # 使用变邻域搜索
            vns = VariableNeighborhoodSearch(self.instance, self.config)
            return vns.solve()

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
    assert batch_size == 1, "只能逐个求解"

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

                if precedence_f == 0:
                    if precedence_f in fleet_info['next_duration']:
                        next_duration_list = fleet_info['next_duration'][precedence_f]
                        next_duration_adjustment = next_duration_list[flight_type_i]
                        adjusted = original_departure - next_duration_adjustment
                    else:
                        adjusted = original_departure
                else:
                    adjusted = original_departure

                adjusted_departure[i][f] = adjusted
                start_la[i][f] = adjusted

        # ==================== 车队-节点映射 ====================
        nodes_F = {}
        for f in range(fleet_size):
            nodes_F[f + 1] = customers + [graph_size + 1 + 2 * f, graph_size + 2 + 2 * f]

        vehicles = {f: [j for j in range(1, initial_vehicle + 1)] for f in range(1, fleet_size + 1)}

        # ==================== 距离矩阵构建 ====================
        distance = {(i, j): 0 for i in nodes for j in nodes}
        for i in nodes:
            id_i = loc[i - 1].item() if i in customers else 0
            for j in nodes:
                id_j = loc[j - 1].item() if j in customers else 0
                distance[(i, j)] = distance_dict[(id_i, id_j)]

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
            precedence_mapping = {3: 2, 5: 4, 6: 1}
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

        # 配置VNS参数
        config = VNSConfig()
        timelimit_map = {20: 1800, 50: 1800, 100: 1800, 200: 3600, 300: 3600}
        config.timelimit = timelimit_map.get(opts.graph_size, 1800)
        config.max_iterations = 1000

        # 根据问题规模调整VNS参数
        if opts.graph_size <= 50:
            config.k_max = 3
            config.max_no_improvement = 50
        elif opts.graph_size <= 100:
            config.k_max = 4
            config.max_no_improvement = 75
        else:
            config.k_max = 5
            config.max_no_improvement = 100

        solver = CVRPTWSolver(instance, config)

        print(f"\n--- 实例 {instance_idx + 1} 开始求解 ---")
        start_time = time.time()

        try:
            if opts.val_method == "greedy":
                solution = solver.solve("greedy")
            elif opts.val_method == "vns":
                solution = solver.solve("vns")
            else:
                logger.warning(f"未知求解方法: {opts.val_method}, 使用VNS")
                solution = solver.solve("vns")

            solve_time = time.time() - start_time

            # ==================== 结果处理 ====================
            if solution.feasible:
                print(f"找到可行解，目标值: {solution.objective_value:.2f}")
                print(f"求解时间: {solve_time:.2f}秒")
                cost.append(solution.objective_value)

                # 输出解的统计信息
                total_routes = sum(len(routes) for routes in solution.routes.values())
                print(f"使用路径数: {total_routes}")

                # 如果使用VNS，输出额外的统计信息
                if opts.val_method == "vns" and hasattr(solver, 'vns'):
                    stats = solver.vns.get_search_statistics()
                    print(f"总改进次数: {stats['total_improvements']}")
                    print(f"无改进迭代数: {stats['no_improvement_count']}")
                    print(f"算子使用情况: {stats['operator_usage']}")

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
    parser = argparse.ArgumentParser(description="基于VNS的CVRPTW求解器")
    parser.add_argument("--filename", type=str, default="./data/agh/agh100_validation_seed4321.pkl",
                        help="数据集文件名")
    parser.add_argument("--problem", type=str, default='agh', help="目前只支持机场地勤(agh)问题")
    parser.add_argument('--graph_size', type=int, default=100, help="问题实例规模 (20, 50, 100, 200, 300)")
    parser.add_argument('--val_method', type=str, default='vns',
                        choices=['greedy', 'vns'], help="求解方法")
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
        sys.exit(1)

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